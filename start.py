"""Start the first-edition UI; reuse an existing matching local 3B service."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from urllib.request import urlopen
import webbrowser

ROOT = Path(__file__).resolve().parent
runtime_config = ROOT / 'runtime.local.json'
RUNTIME = Path(json.loads(runtime_config.read_text(encoding='utf-8-sig'))['runtime_root']).expanduser() if runtime_config.exists() else ROOT
if not RUNTIME.is_absolute():
    RUNTIME = (ROOT / RUNTIME).resolve()
sys.path.insert(0, str(RUNTIME/'.pylibs'))


def available(port):
    with socket.socket() as sock:
        sock.settimeout(.3)
        return sock.connect_ex(('127.0.0.1',port)) != 0


def compatible(port):
    try:
        with urlopen(f'http://127.0.0.1:{port}/v1/models',timeout=2) as resp:
            return 'qwen2.5-3b-instruct' in [m['id'] for m in json.load(resp).get('data',[])]
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8024)
    parser.add_argument('--no-browser',action='store_true')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not available(args.port):
        parser.error('网页端口不可用，请用 --port 指定其他端口')
    model_port = next((p for p in (11518,11434) if compatible(p)),None)
    proc = None
    try:
        if model_port is None:
            model_port = next((p for p in range(11521,11531) if available(p)),None)
            if model_port is None:
                parser.error('没有可用的本地模型端口')
            env = {**os.environ,'PYTHONPATH':str(RUNTIME/'.pylibs'),'HF_HUB_OFFLINE':'1'}
            proc = subprocess.Popen([sys.executable,str(ROOT/'tools/llm_server.py'),
                '--model',str(RUNTIME/'.models/qwen2.5-3b-instruct'),'--port',str(model_port)],env=env,cwd=ROOT)
            print('正在加载本地模型，请稍等……',flush=True)
            deadline = time.monotonic()+180
            while not compatible(model_port):
                if proc.poll() is not None or time.monotonic()>deadline:
                    raise RuntimeError('模型未能启动，请检查 runtime.local.json 的原项目目录')
                time.sleep(1)
        os.environ['OLLAMA_BASE_URL']=f'http://127.0.0.1:{model_port}/v1'
        os.environ['LLM_MODEL']='qwen2.5-3b-instruct'
        import app
        import uvicorn
        url=f'http://127.0.0.1:{args.port}/'
        print('第一版最小界面：'+url,flush=True)
        if not args.no_browser:
            timer=threading.Timer(2,lambda:webbrowser.open(url))
            timer.daemon=True
            timer.start()
        uvicorn.run(app.app,host='127.0.0.1',port=args.port,log_level='warning')
    finally:
        if proc and proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


if __name__=='__main__':
    main()

