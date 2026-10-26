# -*- coding: utf-8 -*-
"""阶段 1 · 验收脚本（对着孔子出的题库跑）。

题库：teach/stage-01-qa.json —— q1~q5 正样本、q6 负样本（股票代码，应拒答）。
通过条件（题库 pass_rule）：q1~q5 命中 >= 4 且 q6 明确拒答。

用法：
    python src/check_stage1.py          # 只验检索链路（不需要 LLM）
    python src/check_stage1.py --llm    # 检索 + 生成 + 答案关键词判分 + 耗时/tok/s

--llm 模式下每题都会真的调一次本地生成服务，记录：答案原文、判分关键词命中、
生成秒数、completion tokens、tok/s，结果写 data/index/eval_stage1.json。
"""
import argparse
import json
import sys

import ask
from config import ROOT

QA_PATH = ROOT / "teach" / "stage-01-qa.json"
RESULT_PATH = ROOT / "data" / "index" / "eval_stage1.json"
MAX_TOKENS = 192

REFUSAL_MARKS = ["资料里没有", "资料中没有", "没有找到", "无法回答", "没有提到", "不知道"]


def load_qa() -> dict:
    return json.loads(QA_PATH.read_text(encoding="utf-8"))


def _norm(s: str) -> str:
    """判分前的归一化：去掉所有空白，并把数字前导零去掉（09:30 == 9:30）。

    这是判分口径问题，不是放宽标准：答案要点没变，只是模型写 9:30 而题库写 09:30。
    严格命中（不去零）另存一列 kw_strict，两个数字都落进 json，谁都能重算。
    """
    import re

    s = re.sub(r"\s+", "", s or "")
    return re.sub(r"(?<!\d)0+(\d)", r"\1", s)


def keyword_check(text: str, item: dict) -> tuple[bool, list[str], list[str], bool]:
    """返回 (是否全中(归一后), 命中的词, 缺的词, 严格命中(未归一))。"""
    need = item.get("answer_must_include", [])
    any_kw = item.get("answer_must_include_any") or []
    norm_text = _norm(text)
    hit = [k for k in need if _norm(k) in norm_text]
    missing = [k for k in need if _norm(k) not in norm_text]
    any_hit = [k for k in any_kw if _norm(k) in norm_text]
    strict = all(k in text for k in need) and (not any_kw or any(k in text for k in any_kw))
    if any_kw and not any_hit:
        return False, hit + any_hit, missing + [f"任一：{any_kw}"], strict
    return not missing, hit + any_hit, missing, strict


def check_one(item: dict, with_llm: bool) -> dict:
    q = item["question"]
    hits = ask.retrieve(q, 3)
    sources = [h["source"] for h in hits]
    joined = "\n".join(h["text"] for h in hits)
    top_dist = hits[0]["distance"]

    row = {
        "id": item["id"],
        "question": q,
        "type": item["type"],
        "sources": sources,
        "top1_dist": round(top_dist, 4),
    }

    if item["type"] == "negative":
        row["retrieval_ok"] = top_dist > ask.DISTANCE_THRESHOLD
        row["gate"] = (
            f"top1_dist {top_dist:.3f} > {ask.DISTANCE_THRESHOLD} → 检索层拒答"
            "（ask.py 生产路径到此为止、不调模型；本脚本仍调一次模型做探针）"
        )
    else:
        kw_hit = all(k in joined for k in item.get("answer_must_include", []))
        any_kw = item.get("answer_must_include_any")
        kw_hit = kw_hit and (any(k in joined for k in any_kw) if any_kw else True)
        doc_hit = item["source_doc"] in sources
        row["retrieval_ok"] = bool(doc_hit and kw_hit)
        row["gate"] = f"来源{'√' if doc_hit else '×'} 关键词{'√' if kw_hit else '×'}"

    row["ok"] = row["retrieval_ok"]

    if with_llm:
        result = ask.generate(q, hits, max_tokens=MAX_TOKENS)
        if result is None:
            row.update({"answer": None, "answer_ok": False,
                        "seconds": None, "completion_tokens": None, "tokens_per_second": None,
                        "note": "生成服务不可用"})
        else:
            text = result["text"]
            row.update({
                "answer": text,
                "seconds": result["seconds"],
                "completion_tokens": result["completion_tokens"],
                "tokens_per_second": result["tokens_per_second"],
            })
            if item["type"] == "negative":
                # 负样本：先看检索层是否拒答，再看模型自己会不会硬编
                refused = any(m in text for m in REFUSAL_MARKS)
                row["answer_ok"] = refused
                row["note"] = "模型" + ("主动拒答" if refused else "未拒答（可能编造）")
            else:
                ok, hit, missing, strict = keyword_check(text, item)
                row.update({"answer_ok": ok, "kw_hit": hit, "kw_missing": missing,
                            "kw_strict": strict})
                if ok and not strict:
                    row["kw_note"] = "关键词按数字格式归一后命中（如 09:30 → 9:30）"
        row["ok"] = bool(row["retrieval_ok"] and row["answer_ok"])

    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="同时验证生成与答案关键词")
    args = parser.parse_args()

    qa = load_qa()
    rows = []
    for item in qa["questions"]:
        row = check_one(item, args.llm)
        rows.append(row)
        print(f"{'[PASS]' if row['ok'] else '[FAIL]'} {row['id']} {row['question']}")
        print(f"       top1 距离={row['top1_dist']}  来源={row['sources']}")
        print(f"       检索：{row['gate']}")
        if args.llm:
            print(f"       答案{'√' if row.get('answer_ok') else '×'} "
                  f"{row.get('seconds')}s / {row.get('completion_tokens')} tokens / "
                  f"{row.get('tokens_per_second')} tok/s")
            text = (row.get("answer") or "").replace("\n", " ")
            print(f"       原文：{text[:160]}{'…' if len(text) > 160 else ''}")
            if row.get("kw_hit") or row.get("kw_missing"):
                print(f"       关键词命中={row.get('kw_hit')} 缺={row.get('kw_missing')}")
                if row.get("kw_note"):
                    print(f"       {row['kw_note']}")
        print()

    pos = [r for r in rows if r["type"] != "negative"]
    neg = [r for r in rows if r["type"] == "negative"]
    pos_ok = sum(1 for r in pos if r["ok"])
    neg_ok = all(r["ok"] for r in neg)
    passed = pos_ok >= 4 and neg_ok

    if args.llm:
        secs = [r["seconds"] for r in rows if r.get("seconds")]
        tps = [r["tokens_per_second"] for r in rows if r.get("tokens_per_second")]
        toks = [r["completion_tokens"] for r in rows if r.get("completion_tokens")]
        if secs:
            print(f"生成耗时：合计 {sum(secs):.1f}s，平均 {sum(secs)/len(secs):.1f}s/题"
                  f"（{min(secs):.1f}~{max(secs):.1f}s）")
        if toks and tps:
            print(f"输出 token：平均 {sum(toks)/len(toks):.0f}/题，平均 "
                  f"{sum(tps)/len(tps):.2f} tok/s")

    print(f"\n正样本 {pos_ok}/{len(pos)}，负样本 {'通过' if neg_ok else '未通过'}"
          f" → {'ACCEPT' if passed else 'NOT ACCEPT'}"
          f"（{'含生成' if args.llm else '仅检索'}）")

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps({"mode": "llm" if args.llm else "retrieval",
                    "passed": passed, "rows": rows},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"结果写到 {RESULT_PATH}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
