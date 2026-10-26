# -*- coding: utf-8 -*-
"""建索引：本地文档 -> 切块 -> embedding -> 存入 Chroma。

用法：
    python src/ingest.py                          # 阶段 1 默认：data/docs，A 策略
    python src/ingest.py --strategy B             # 换内容感知切块
    python src/ingest.py --docs-dir data/docs-long --collection long_a \
                         --chroma-dir data/index/chroma_A --strategy A

每次运行都会重建目标集合（幂等）。build_index() 抽出来给 A/B 对比脚本复用。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import config
from chunkers import get_strategy, hard_split
from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DOC_DIR,
    MODEL_NAME,
)

# 兼容阶段 1 的旧名字
split_text = hard_split


def load_documents(doc_dir: Path = DOC_DIR) -> list[tuple[str, str]]:
    """读取 doc_dir 下所有 .md / .txt，返回 [(文件名, 全文)]，按文件名排序。"""
    doc_dir = Path(doc_dir)
    files = sorted(
        p for p in doc_dir.rglob("*") if p.suffix.lower() in {".md", ".txt"}
    )
    if not files:
        raise SystemExit(f"没有找到文档，请把 .md/.txt 放到 {doc_dir}")
    docs = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        docs.append((path.name, text))
    return docs


def build_chunks(
    docs: list[tuple[str, str]],
    strategy: str = "A",
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    name, fn = get_strategy(strategy)
    records = []
    for doc_name, text in docs:
        pieces = fn(text, size, overlap)
        for i, piece in enumerate(pieces):
            records.append(
                {
                    "id": f"{Path(doc_name).stem}-{i:03d}",
                    "text": piece,
                    "source": doc_name,
                    "chunk_index": i,
                }
            )
    return records


def chunk_stats(records: list[dict]) -> dict:
    lens = [len(r["text"]) for r in records] or [0]
    return {
        "chunks": len(records),
        "min_len": min(lens),
        "avg_len": round(sum(lens) / len(lens), 1),
        "max_len": max(lens),
    }


def build_index(
    docs_dir: Path = DOC_DIR,
    chroma_dir: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    strategy: str = "A",
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    model=None,
    verbose: bool = True,
) -> dict:
    docs_dir, chroma_dir = Path(docs_dir), Path(chroma_dir)
    t0 = time.time()
    docs = load_documents(docs_dir)
    records = build_chunks(docs, strategy, size, overlap)
    st = chunk_stats(records)
    if verbose:
        print(
            f"[1/3] {docs_dir}：{len(docs)} 个文档，切出 {st['chunks']} 块"
            f"（{st['min_len']}~{st['max_len']} 字，均 {st['avg_len']}）"
        )

    if model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                f"依赖没装好：{exc}\n先执行 pip install -r requirements.txt，"
                "或按 README 用 .pylibs 方式安装。"
            )
        if verbose:
            print(f"[2/3] 加载 embedding 模型：{MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME, device="cpu")

    vectors = model.encode(
        [r["text"] for r in records],
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    import chromadb

    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(name=collection_name)
    collection.add(
        ids=[r["id"] for r in records],
        documents=[r["text"] for r in records],
        metadatas=[
            {"source": r["source"], "chunk_index": r["chunk_index"]} for r in records
        ],
        embeddings=[v.tolist() for v in vectors],
    )
    if verbose:
        print(f"[3/3] 集合 {collection_name} 现有 {collection.count()} 条 → {chroma_dir}")

    meta = {
        "model": MODEL_NAME,
        "strategy": strategy,
        "chunk_size": size,
        "chunk_overlap": overlap,
        "docs_dir": str(docs_dir.relative_to(config.ROOT)) if docs_dir.is_relative_to(config.ROOT) else str(docs_dir),
        "collection": collection_name,
        "chroma_dir": str(chroma_dir.relative_to(config.ROOT)) if chroma_dir.is_relative_to(config.ROOT) else str(chroma_dir),
        "docs": len(docs),
        **st,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_name = (
        "index_meta.json"
        if collection_name == COLLECTION_NAME
        else f"index_meta_{collection_name}.json"
    )
    (chroma_dir.parent / meta_name).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if verbose:
        print(f"完成，用时 {time.time() - t0:.1f}s")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="建向量索引（阶段 1/2）")
    parser.add_argument("--docs-dir", default=str(DOC_DIR))
    parser.add_argument("--chroma-dir", default=str(CHROMA_DIR))
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--strategy", default="A", choices=["A", "B", "a", "b"])
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    args = parser.parse_args()

    size = args.chunk_size or (CHUNK_SIZE if args.strategy.upper() == "A" else 500)
    build_index(
        docs_dir=Path(args.docs_dir),
        chroma_dir=Path(args.chroma_dir),
        collection_name=args.collection,
        strategy=args.strategy.upper(),
        size=size,
        overlap=args.overlap if args.overlap is not None else CHUNK_OVERLAP,
    )


if __name__ == "__main__":
    main()
