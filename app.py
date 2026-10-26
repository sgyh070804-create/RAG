"""Minimal UI adapter around the first edition's unchanged retrieval/generation code."""
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
runtime_config = ROOT / 'runtime.local.json'
RUNTIME = Path(json.loads(runtime_config.read_text(encoding='utf-8-sig'))['runtime_root']).expanduser() if runtime_config.exists() else ROOT
if not RUNTIME.is_absolute():
    RUNTIME = (ROOT / RUNTIME).resolve()
sys.path[:0] = [str(RUNTIME / '.pylibs'), str(ROOT / 'src')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))
import config
# Keep first-edition sources intact; only resolve the read-only runtime location.
config.MODEL_NAME = str(RUNTIME / '.models' / 'bge-small-zh-v1.5')
config.LLM_MODEL = 'qwen2.5-3b-instruct'
import ask as rag
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

STATE = {'ready': False, 'error': None, 'started': time.monotonic()}
LOCK = threading.Lock()


def warmup():
    try:
        rag.get_model()
        import chromadb
        coll = chromadb.PersistentClient(path=str(config.CHROMA_DIR)).get_collection(config.COLLECTION_NAME)
        if not coll.count():
            raise RuntimeError('第一版索引为空')
        STATE['ready'] = True
    except Exception as exc:
        STATE['error'] = str(exc)


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=warmup, daemon=True).start()
    yield


app = FastAPI(title='第一版 RAG · 最小演示', lifespan=lifespan)


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.get('/', response_class=HTMLResponse)
def home():
    return (ROOT / 'index.html').read_text(encoding='utf-8')


@app.get('/api/health')
def health():
    return {'ready': STATE['ready'], 'error': STATE['error'], 'waited': round(time.monotonic()-STATE['started'])}


@app.post('/api/ask')
def answer(req: Question):
    question = req.question.strip()
    if not question:
        raise HTTPException(422, '请输入问题')
    if not STATE['ready']:
        raise HTTPException(503, STATE['error'] or '知识库还在加载，请稍后再试')
    if not LOCK.acquire(blocking=False):
        raise HTTPException(429, '上一个问题还在处理，请稍后再试')
    started = time.monotonic()
    try:
        hits = rag.retrieve(question, config.TOP_K)
        retrieved = time.monotonic()
        refused = not hits or hits[0]['distance'] > rag.DISTANCE_THRESHOLD
        result = None if refused else rag.generate(question, hits)
        message = '资料中没有提到相关内容。' if refused else result['text'] if result and result['text'].strip() else '生成服务暂不可用，请查看下方检索资料。'
        return {'answer': message, 'refused': refused,
                'generation_error': not refused and (not result or not result['text'].strip()),
                'hits': [{'source': h['source'], 'distance': round(h['distance'], 4), 'text': h['text']} for h in hits],
                'timing': {'retrieve': round(retrieved-started,2),
                           'generate': round(time.monotonic()-retrieved,2),
                           'total': round(time.monotonic()-started,2)}}
    finally:
        LOCK.release()

