# -*- coding: utf-8 -*-
"""阶段 2 · 切块 A/B 对比（只做检索，不调 LLM）。

A  = hard_split(400, 80)  ：固定 400 字窗口 + 句末收口 + 80 字重叠（阶段 1 现状）
A0 = hard_split(400, 0)   ：同一算法但零重叠 = 教科书式「硬切」（孔子 m00499 要求的口径）
B  = structure_split(500, 80)：标题/段落/表格/代码块切成原子块，打包到 500 字上限 + 80 字重叠

每题在 A、A0、B 三个索引上检索 top-k，记录：
- top1 来源、top1 距离、top3 是否包含应命中文档
- top1 / top3 的块里是否**完整**包含答案关键词（被劈开的题主要看这个）
- 答案在原文里存在，但没有任何一个单块完整包含全部关键词 → 判为「被切块劈开」
- 旧题（阶段 1 题库）的 source_doc 不在长语料里时，改按「关键词是否完整出现在块里」判命中，
  不再整题 SKIP——前提是这些事实确实写进了长文档。

输出 data/index/eval_stage2.json，A/B 两列并排。

用法：
    python src/stage2_ab.py
    python src/stage2_ab.py --qa teach/stage-02-qa.json --skip-short
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import config
from ask import DISTANCE_THRESHOLD, get_model
from ingest import build_chunks, build_index, load_documents

ROOT = config.ROOT
LONG_DIR = ROOT / "data" / "docs-long"
SHORT_DIR = ROOT / "data" / "docs"
QA_SAVE = ROOT / "data" / "index" / "eval_stage2.json"

STRATEGY_PARAMS = {"A": ("A", 400, 80), "A0": ("A", 400, 0), "B": ("B", 500, 80)}
KEYS = tuple(STRATEGY_PARAMS)
WS_RE = re.compile(r"\s+")


def norm(text: str) -> str:
    """去掉空白、统一大小写、去掉数字前导零，判关键词用。"""
    text = WS_RE.sub("", str(text)).lower()
    return re.sub(r"(?<!\d)0+(?=\d)", "", text)


def question_keywords(q: dict) -> tuple[list[str], list[str]]:
    """返回 (必须全含, 至少含一个)。优先 answer_span，它视为唯一必须项。"""
    if q.get("answer_span"):
        return [q["answer_span"]], []
    required = [k for k in (q.get("answer_must_include") or []) if k]
    any_group = [k for k in (q.get("answer_must_include_any") or []) if k]
    return required, any_group


def covered(text: str, required: list[str], any_group: list[str] = ()) -> bool:
    """这段文本是否满足：必须项全含 + any 组里至少含一个（any 组为空则只判必须项）。"""
    if not required and not any_group:
        return False
    n = norm(text)
    if not all(norm(k) in n for k in required):
        return False
    if not any_group:
        return True
    return any(norm(k) in n for k in any_group)


def split_state(
    doc_text: str, records: list[dict], required: list[str], any_group: list[str] = ()
) -> bool | None:
    """True=被劈开，False=某一块完整包含，None=原文里就凑不齐、判不了。"""
    if not required:
        return None
    doc_norm = norm(doc_text)
    if not all(norm(k) in doc_norm for k in required):
        return None
    if any_group and not any(norm(k) in doc_norm for k in any_group):
        return None
    return not any(covered(r["text"], required, any_group) for r in records)


def retrieve(chroma_dir: Path, collection: str, question: str, top_k: int, model):
    import chromadb

    qvec = model.encode([question], normalize_embeddings=True)[0].tolist()
    client = chromadb.PersistentClient(path=str(chroma_dir))
    coll = client.get_collection(collection)
    res = coll.query(
        query_embeddings=[qvec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "text": text,
            "source": meta.get("source", "?"),
            "chunk_index": meta.get("chunk_index", -1),
            "distance": float(dist),
        }
        for text, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


def reused_index(chroma_dir: Path, collection: str, records: list[dict]) -> dict | None:
    """集合已存在且块数对得上就直接复用，省掉重新 embed 的十几秒。"""
    if not (chroma_dir / "chroma.sqlite3").exists():
        return None
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(chroma_dir))
        coll = client.get_collection(collection)
        count = coll.count()
    except Exception:  # noqa: BLE001
        return None
    if count != len(records):
        return None
    lens = [len(r["text"]) for r in records] or [0]
    return {
        "chunks": count,
        "min_len": min(lens),
        "avg_len": round(sum(lens) / len(lens), 1),
        "max_len": max(lens),
        "reused": True,
    }


def build_side(docs, docs_dir: Path, label: str, key: str, model, rebuild: bool = False) -> dict:
    strategy, size, overlap = STRATEGY_PARAMS[key]
    chroma_dir = ROOT / "data" / "index" / f"chroma_{label}_{key}"
    collection = f"{label}_{key}".lower()
    records = build_chunks(docs, strategy, size, overlap)
    meta = None if rebuild else reused_index(chroma_dir, collection, records)
    if meta is not None:
        print(f"  [{label}_{key}] 复用已有索引（{meta['chunks']} 块）")
    else:
        meta = build_index(
            docs_dir=docs_dir,
            chroma_dir=chroma_dir,
            collection_name=collection,
            strategy=strategy,
            size=size,
            overlap=overlap,
            model=model,
            verbose=True,
        )
    return {
        "meta": meta,
        "chroma_dir": str(chroma_dir.relative_to(ROOT)),
        "collection": collection,
        "records": records,
    }


def run_one(qa_sets, docs_dir: Path, label: str, model, top_k: int, remap: dict | None = None,
            rebuild: bool = False) -> dict:
    remap = remap or {}
    docs_dir = docs_dir.resolve()
    docs = load_documents(docs_dir)
    doc_map = {name: text for name, text in docs}
    corpus_text = "\n".join(doc_map.values())
    corpus_norm = norm(corpus_text)
    print(f"\n===== {label}：{docs_dir.name}（{len(docs)} 篇）")

    sides = {key: build_side(docs, docs_dir, label, key, model, rebuild) for key in KEYS}

    rows = []
    for qa in qa_sets:
        for q in qa.get("questions", []):
            required, any_group = question_keywords(q)
            raw_src = q.get("source_doc")
            # 只有重映射目标确实在本语料里才换名，短文档语料仍用原名
            expected = raw_src
            if raw_src and remap.get(raw_src) in doc_map:
                expected = remap[raw_src]
            in_src = bool(expected) and expected in doc_map
            # any 组（如「一人操作/一人确认」）若在本语料换了措辞，就退回只判必须项，并在行里记明
            eff_any = [k for k in any_group if norm(k) in corpus_norm]
            kw_ok = bool(required) and all(norm(k) in corpus_norm for k in required)
            applicable = expected is None or in_src or kw_ok

            row = {
                "qa": qa.get("corpus") or f"stage{qa.get('stage')}",
                "id": q.get("id"),
                "type": q.get("type"),
                "question": q.get("question"),
                "raw_source": raw_src,
                "expected_source": expected,
                "source_in_corpus": in_src,
                "keywords": required,
                "keywords_any": any_group,
                "keywords_any_used": eff_any,
                "applicable": applicable,
            }
            if required and any_group and not eff_any:
                row["kw_note"] = (
                    f"any 组 {any_group} 在 {docs_dir.name} 没有对应措辞，本题按必须项判"
                )
            if not applicable:
                row["note"] = f"答案关键词在 {docs_dir.name} 里找不到，跳过"
                rows.append(row)
                continue

            for key, st in sides.items():
                item = {}
                hits = retrieve(st["chroma_dir"], st["collection"], q["question"], top_k, model)
                sources = [h["source"] for h in hits]
                item["top1_source"] = hits[0]["source"] if hits else None
                item["top1_dist"] = round(hits[0]["distance"], 4) if hits else None
                item["top3_sources"] = sources
                item["answer_in_top1"] = bool(hits) and covered(hits[0]["text"], required, eff_any)
                item["answer_in_top3"] = any(covered(h["text"], required, eff_any) for h in hits)

                if expected is None:
                    item["refused"] = bool(hits) and hits[0]["distance"] > DISTANCE_THRESHOLD
                elif in_src:
                    item["hit_top1"] = bool(hits) and hits[0]["source"] == expected
                    item["hit_top3"] = expected in sources
                else:
                    item["hit_top1"] = item["answer_in_top1"]
                    item["hit_top3"] = item["answer_in_top3"]

                recs = (
                    [r for r in st["records"] if r["source"] == expected]
                    if in_src
                    else st["records"]
                )
                doc_text = doc_map[expected] if in_src else corpus_text
                item["split"] = split_state(doc_text, recs, required, eff_any)
                row[key] = item

            if raw_src and raw_src != expected:
                row["note"] = f"映射 {raw_src} → {expected}"
            elif expected and not in_src:
                row["note"] = "source_doc 不在本语料，按关键词是否完整出现在块里判命中"
            rows.append(row)

    summary = summarize(rows)
    for key in KEYS:
        summary[key]["chunk_count"] = sides[key]["meta"]["chunks"]
        summary[key]["avg_chunk_len"] = sides[key]["meta"]["avg_len"]

    # 预埋陷阱核验：出题人说「某事实必被 A 劈开」，这里按切块结果实测一次
    trap_checks = []
    for qa in qa_sets:
        trap = qa.get("trap")
        if not trap or not trap.get("fact") or trap.get("doc") not in doc_map:
            continue
        fact = trap["fact"]
        check = {"doc": trap["doc"], "fact": fact}
        for key, st in sides.items():
            recs = [r for r in st["records"] if r["source"] == trap["doc"]]
            hit = [r["chunk_index"] for r in recs if covered(r["text"], [fact])]
            check[key] = {
                "fact_in_single_chunk": bool(hit),
                "chunk_index": hit,
                "chunks": len(recs),
            }
        trap_checks.append(check)

    print_summary(label, summary)
    for c in trap_checks:
        detail = "、".join(
            f"{k} 完整={c[k]['fact_in_single_chunk']}(块{c[k]['chunk_index']})" for k in KEYS
        )
        print(f"  [trap] {c['fact']}：{detail}")
    return {
        "docs_dir": str(docs_dir.relative_to(ROOT)),
        "docs": len(docs),
        **{key: {k: sides[key][k] for k in ("meta", "chroma_dir", "collection")} for key in KEYS},
        "rows": rows,
        "summary": summary,
        "trap_check": trap_checks,
    }


def _pct(a: int, b: int):
    return round(a / b, 4) if b else None


def summarize(rows: list[dict]) -> dict:
    out = {}
    for key in KEYS:
        live = [r for r in rows if r["applicable"]]
        pos = [r for r in live if r["expected_source"] is not None]
        src_pos = [r for r in pos if r["source_in_corpus"]]
        neg = [r for r in live if r["expected_source"] is None]
        splits = [r for r in live if r[key].get("split") is True]
        out[key] = {
            "evaluated": len(live),
            "positive": len(pos),
            "hit_top1": sum(1 for r in pos if r[key].get("hit_top1")),
            "hit_top1_rate": _pct(sum(1 for r in pos if r[key].get("hit_top1")), len(pos)),
            "hit_top3": sum(1 for r in pos if r[key].get("hit_top3")),
            "hit_top3_rate": _pct(sum(1 for r in pos if r[key].get("hit_top3")), len(pos)),
            "src_hit_top1": sum(1 for r in src_pos if r[key].get("hit_top1")),
            "src_positive": len(src_pos),
            "answer_in_top1": sum(1 for r in pos if r[key].get("answer_in_top1")),
            "answer_in_top1_rate": _pct(
                sum(1 for r in pos if r[key].get("answer_in_top1")), len(pos)
            ),
            "mean_top1_dist": round(
                sum(r[key]["top1_dist"] for r in pos if r[key].get("top1_dist") is not None)
                / max(1, len([r for r in pos if r[key].get("top1_dist") is not None])),
                4,
            )
            if pos
            else None,
            "split": len(splits),
            "split_ids": [r["id"] for r in splits],
            "split_answer_in_top1": sum(1 for r in splits if r[key].get("answer_in_top1")),
            "negative": len(neg),
            "refused": sum(1 for r in neg if r[key].get("refused")),
            "skipped": [r["id"] for r in rows if not r["applicable"]],
        }
    a, b = out["A"], out["B"]
    a0 = out["A0"]
    out["verdict"] = {
        "b_hit_top1_ge_a": (b["hit_top1"] >= a["hit_top1"]) if a["positive"] else None,
        "b_answer_in_top1_ge_a": (b["answer_in_top1"] >= a["answer_in_top1"])
        if a["positive"]
        else None,
        "b_hit_top1_ge_a0": (b["hit_top1"] >= a0["hit_top1"]) if a0["positive"] else None,
        "b_answer_in_top1_ge_a0": (b["answer_in_top1"] >= a0["answer_in_top1"])
        if a0["positive"]
        else None,
        "split_fixed_vs_a": sorted(set(a["split_ids"]) - set(b["split_ids"])),
        "split_fixed_vs_a0": sorted(set(a0["split_ids"]) - set(b["split_ids"])),
        "split_new_in_b": sorted(set(b["split_ids"]) - set(a["split_ids"]) - set(a0["split_ids"])),
        "split_ids_by_side": {k: out[k]["split_ids"] for k in KEYS},
        "answer_recovered_in_b": {
            ref: sorted(
                r["id"]
                for r in rows
                if r["applicable"]
                and r.get(ref, {}).get("split") is True
                and r.get("B", {}).get("answer_in_top1") is True
            )
            for ref in ("A", "A0")
        },
        "refused_note": f"负样本 A {a['refused']}/{a['negative']}、A0 {a0['refused']}/{a0['negative']}、"
        f"B {b['refused']}/{b['negative']} 被拒答（阈值 {DISTANCE_THRESHOLD}）",
        "note": "只做检索对比；如果 B 没提升就照实写，不说「效果更好」。",
    }
    return out


def print_summary(label: str, s: dict) -> None:
    print(f"\n----- {label} 汇总 -----")
    for key in KEYS:
        x = s[key]
        print(
            f"{key}: 块数 {x['chunk_count']}（均长 {x['avg_chunk_len']}）, "
            f"top1 {x['hit_top1']}/{x['positive']}（{x['hit_top1_rate']}）, "
            f"含完整答案 {x['answer_in_top1']}/{x['positive']}（{x['answer_in_top1_rate']}）, "
            f"均距 {x['mean_top1_dist']}, 被劈开 {x['split']} {x['split_ids']}"
        )
    print("verdict:", json.dumps(s["verdict"], ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段 2 切块 A/B 对比（仅检索）")
    parser.add_argument("--docs-dir", default=str(LONG_DIR), help="主对比语料目录")
    parser.add_argument("--qa", nargs="*", default=None, help="题集 json，默认自动收集")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--out", default=str(QA_SAVE))
    parser.add_argument("--skip-short", action="store_true", help="不跑短文档对照")
    parser.add_argument("--rebuild", action="store_true", help="强制重建索引（默认复用已有集合）")
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir).resolve()
    if not docs_dir.exists() or not any(docs_dir.rglob("*")):
        raise SystemExit(f"长语料还没就位：{docs_dir}（等孔子产出）")

    qa_files = args.qa or sorted(
        str(p) for p in (ROOT / "teach").glob("stage-0*-qa.json")
    )
    qa_sets = [json.loads(Path(p).read_text(encoding="utf-8")) for p in qa_files]
    print(f"题集：{[Path(p).name for p in qa_files]}")

    # 旧题 source_doc → 长文档 的映射由孔子写在 stage-02-qa.json（old_question_source_map）
    remap = {}
    for qa in qa_sets:
        remap.update(qa.get("old_question_source_map") or {})
    if remap:
        print(f"旧题重映射：{json.dumps(remap, ensure_ascii=False)}")

    model = get_model()
    t0 = time.time()
    result = {
        "stage": 2,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": config.MODEL_NAME,
        "top_k": args.top_k,
        "distance_threshold": DISTANCE_THRESHOLD,
        "qa_files": [Path(p).name for p in qa_files],
        "old_question_source_map": remap,
        "comparisons": {},
    }
    result["comparisons"]["long"] = run_one(
        qa_sets, docs_dir, "long", model, args.top_k, remap, args.rebuild
    )
    if not args.skip_short and SHORT_DIR.exists():
        result["comparisons"]["short"] = run_one(
            qa_sets, SHORT_DIR, "short", model, args.top_k, remap, args.rebuild
        )

    result["elapsed_seconds"] = round(time.time() - t0, 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {out}（{result['elapsed_seconds']}s）")


if __name__ == "__main__":
    main()
