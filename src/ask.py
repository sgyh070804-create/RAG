# -*- coding: utf-8 -*-
"""阶段 1 · 问答：问题 -> 向量化 -> top3 检索 -> 拼 prompt -> LLM 生成。

用法：
    python src/ask.py "工单附件最大多大？"
    python src/ask.py                 # 进入交互模式，逐条提问
    python src/ask.py --no-llm "..."  # 只验证检索，不调 LLM
"""
import argparse
import sys

import config
from config import CHROMA_DIR, COLLECTION_NAME, LLM_MODEL, OLLAMA_BASE_URL, TOP_K

# 检索距离阈值（L2 距离，向量已归一化；越小越像）。
# 实测：库里相关问题的 top1 距离 0.62~1.09，文档外问题 1.23，
# 所以取 1.2 作为「硬拒答」门槛。这个阈值的余量很小，
# 阶段 4（rerank / 混合召回）会把它替换成更稳的判据。
DISTANCE_THRESHOLD = 1.2

SYSTEM_PROMPT = """你是「云帆工单系统」的产品支持助手。
规则：
1. 只能依据下面给出的【资料】回答，不要使用资料以外的知识；
2. 资料里没有的信息，直接回答「资料中没有提到」，不要猜；
3. 回答简洁，中文，必要时分点；
4. 最后单独一行写「来源：」+ 你实际用到的文件名（只写文件名本身）。"""


_MODEL = None


def get_model():
    """模型只加载一次（一次加载约 1-2 秒，批量验收时差别明显）。"""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(config.MODEL_NAME, device="cpu")
    return _MODEL


def retrieve(question: str, top_k: int):
    qvec = get_model().encode([question], normalize_embeddings=True)[0].tolist()
    return query_collection(qvec, top_k)


def query_collection(qvec, top_k: int):
    """用现成的查询向量检索（--fast 演示档复用它，避免为了 6 个问题加载模型）。"""
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    res = collection.query(
        query_embeddings=[qvec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for text, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append(
            {
                "text": text,
                "source": meta.get("source", "?"),
                "chunk_index": meta.get("chunk_index", -1),
                "distance": float(dist),
            }
        )
    return hits


def print_hits(hits, full: bool = False) -> None:
    print("\n--- 召回 top%d ---" % len(hits))
    for i, h in enumerate(hits, 1):
        print(f"[{i}] 距离={h['distance']:.4f}  来源={h['source']}  块#{h['chunk_index']}")
        if full:
            print(h["text"])
        else:
            preview = h["text"].replace("\n", " ")
            if len(preview) > 70:
                preview = preview[:70] + "…"
            print(f"    {preview}")
        print()


def build_prompt(question: str, hits) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(f"[{i}]（来源：{h['source']}）\n{h['text']}")
    return "【资料】\n" + "\n\n".join(blocks) + f"\n\n【问题】{question}"


def generate(question: str, hits, max_tokens: int = 192) -> dict | None:
    """调用本地服务（OpenAI 兼容）；连不上就返回 None，让调用方走降级路径。

    返回 {"text", "seconds", "completion_tokens", "tokens_per_second"}，
    这几个数字直接进评估表（阶段 1 的延迟/成本指标从这里来）。
    """
    import time

    from openai import OpenAI

    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", timeout=180)
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, hits)},
            ],
            temperature=0.1,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        print(f"\n[!] 调用 LLM 失败（{type(exc).__name__}: {exc}）")
        print("    服务地址 " + OLLAMA_BASE_URL)
        print("    Ollama 路径：ollama serve / ollama pull " + LLM_MODEL)
        print("    本地自架服务：python tools/llm_server.py --port 11434")
        return None

    seconds = time.time() - t0
    usage = getattr(resp, "usage", None)
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
    tps = getattr(usage, "tokens_per_second", None) if usage else None
    if tps is None and completion_tokens:
        tps = round(completion_tokens / seconds, 2)
    return {
        "text": resp.choices[0].message.content or "",
        "seconds": round(seconds, 2),
        "completion_tokens": completion_tokens,
        "tokens_per_second": round(tps, 2) if tps else None,
    }


def answer_once(question: str, top_k: int, use_llm: bool, full: bool = False) -> None:
    print(f"\n========== 问题：{question}")
    hits = retrieve(question, top_k)
    print_hits(hits, full)

    if not hits or hits[0]["distance"] > DISTANCE_THRESHOLD:
        print("\n--- 回答 ---")
        print("资料中没有提到相关内容。")
        return

    if not use_llm:
        print("\n（--no-llm：跳过生成，只看召回）")
        return

    result = generate(question, hits)
    if result is None:
        print("\n--- 回答（降级：仅召回）---")
        print("LLM 不可用，请先启动本地服务。以下是最相关的一段资料：")
        print(hits[0]["text"])
        return

    print("\n--- 回答 ---")
    print(result["text"])
    print(
        f"\n（生成 {result['seconds']}s，{result['completion_tokens']} tokens，"
        f"{result['tokens_per_second']} tok/s）"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="云帆工单知识库问答（阶段 1）")
    parser.add_argument("question", nargs="*", help="要问的问题；省略则进入交互模式")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--no-llm", action="store_true", help="只检索、不生成")
    parser.add_argument("--full", action="store_true", help="打印召回块的完整原文")
    args = parser.parse_args()

    if not (CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir())):
        raise SystemExit(f"向量库不存在：{CHROMA_DIR}\n先运行 python src/ingest.py")

    if args.question:
        answer_once(" ".join(args.question), args.top_k, not args.no_llm, args.full)
        return

    print("交互模式，输入问题回车；输入 q 退出。")
    while True:
        try:
            q = input("\n问> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in {"q", "quit", "exit"}:
            break
        if not q:
            continue
        answer_once(q, args.top_k, not args.no_llm, args.full)


if __name__ == "__main__":
    main()
