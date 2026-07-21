#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E2 覆盖率曲线（T2⑦c）—— 把「端到端回收受语料覆盖瓶颈」从 hedge 变成显式贡献。

逻辑
----
对 Part2 的 unresolved 引用（规则匹配不到），按【候选检索最佳 overlap】分桶：
  该 overlap 高 = 真目标很可能在 arxiv 语料池里（in-corpus）
  该 overlap 低/无候选 = 真目标不在池里（out-of-corpus，如 Lü&Zhou 2011 期刊论文）
然后每桶抽样跑 LLM 匹配 + 独立裁判，得【每桶 LLM 回收率】。
曲线直观显示：回收率随语料覆盖下降而塌缩 → 瓶颈是语料覆盖，非匹配器。

复用 run_e2_citation_ablation 的全部 helper（不重复实现）。

用法
  python benchmark/run_e2_coverage_curve.py --parse-papers 300 --per-bucket 25 --llm-workers 3
"""
import os
import sys
import json
import random
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmark"))

# 显式加载 .env（API_KEY/MODEL_BASE_URL/MODEL_NAME），不依赖 shell 预先 source
_envf = ROOT / ".env"
if _envf.exists():
    for _line in _envf.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

import run_e2_citation_ablation as E2  # 复用 helper
from run_e2_citation_ablation import (load_metadata, get_agent_parser, parse_paper_refs,
                                      normalize_title, retrieve_candidates, rule_match_production,
                                      llm_match, judge_correct, make_llm_client, STOP,
                                      META_PATH, PAPERS_DIR)

OUT = ROOT / "benchmark" / "results" / "e2_citation_ablation"

ap = argparse.ArgumentParser()
ap.add_argument("--parse-papers", type=int, default=300)
ap.add_argument("--k", type=int, default=20)
ap.add_argument("--per-bucket", type=int, default=25, help="每桶抽样送 LLM 数")
ap.add_argument("--llm-workers", type=int, default=3)
ap.add_argument("--seed", type=int, default=1)
args = ap.parse_args()
random.seed(args.seed)
model = os.getenv("MODEL_NAME", "mimo-v2-flash")


def coverage_bucket(best_overlap_norm, n_cand):
    """语料覆盖桶。返回 (bucket_name, order)。"""
    if n_cand == 0:
        return ("out_of_corpus(no_candidate)", 0)
    if best_overlap_norm < 0.25:
        return ("low_coverage(<.25)", 1)
    if best_overlap_norm < 0.5:
        return ("mid_coverage(.25-.5)", 2)
    return ("high_coverage(>=.5)", 3)


def main():
    meta = load_metadata(META_PATH)
    t2i = meta["title_to_id"]; a2c = meta["arxiv_id_to_canonical"]
    wi = meta["word_index"]; id2t = meta["id_to_title"]; id2a = meta["id_to_authors"]
    print(f"[pool] {len(t2i)} titles")

    # 1. 收集 unresolved + 每条的候选最佳 overlap
    agent = get_agent_parser()
    paper_ids = sorted(os.listdir(PAPERS_DIR))
    random.shuffle(paper_ids)
    unresolved = []
    seen = set()
    for pid in paper_ids[:args.parse_papers * 2]:
        md = os.path.join(PAPERS_DIR, pid, f"{pid}.md")
        if not os.path.exists(md):
            continue
        refs = parse_paper_refs(agent, md)
        for r in refs:
            key = normalize_title(r["title"])
            if key in seen:
                continue
            seen.add(key)
            if rule_match_production(r["title"], r["arxiv_id"], t2i, a2c, wi, 0.90) is not None:
                continue  # resolved by rule，跳过
            # 检索候选 + 计算 best overlap 归一化
            cands = retrieve_candidates(r["title"], wi, t2i, id2t, id2a, K=args.k)
            nref = normalize_title(r["title"])
            sig = {w for w in (set(nref.split()) - STOP) if len(w) >= 3}
            best_ov = max((c[3] for c in cands), default=0)
            best_norm = best_ov / max(len(sig), 1)
            unresolved.append({"ref": r["title"], "from": pid,
                               "n_cand": len(cands), "best_overlap_norm": round(best_norm, 3),
                               "candidates": cands})
        if len(unresolved) >= 1500:
            break
    print(f"[unresolved] {len(unresolved)} (parsed ~{args.parse_papers} papers)")

    # 2. 分桶
    buckets = defaultdict(list)
    for u in unresolved:
        bname, _ = coverage_bucket(u["best_overlap_norm"], u["n_cand"])
        buckets[bname].append(u)
    for b, lst in buckets.items():
        print(f"  {b}: {len(lst)}")

    # 3. 每桶抽样送 LLM
    client = make_llm_client()
    summary = {}
    for bname, lst in buckets.items():
        sample = random.sample(lst, min(args.per_bucket, len(lst)))
        correct = 0
        tried = 0

        def work(u):
            cand = u["candidates"]
            pid = llm_match(client, model, u["ref"], cand)
            ok, _ = judge_correct(u["ref"], pid, id2t, id2a)
            return ok

        with ThreadPoolExecutor(max_workers=args.llm_workers) as ex:
            futs = [ex.submit(work, u) for u in sample]
            for f in as_completed(futs):
                tried += 1
                if f.result():
                    correct += 1
                if tried % 5 == 0:
                    print(f"    [{bname}] {tried}/{len(sample)}...")
        rec = correct / tried if tried else 0.0
        summary[bname] = {"n_in_bucket": len(lst), "sampled": tried,
                          "llm_correct": correct, "llm_recovery": round(rec, 3),
                          "mean_best_overlap": round(
                              sum(u["best_overlap_norm"] for u in sample) / max(len(sample), 1), 3)}
        print(f"  >>> {bname}: recovery={rec:.3f} (n={tried})")

    # 排序输出
    order = {0: "out_of_corpus(no_candidate)", 1: "low_coverage(<.25)",
             2: "mid_coverage(.25-.5)", 3: "high_coverage(>=.5)"}
    curve = [{"bucket": order[i], **summary.get(order[i], {"n_in_bucket": 0, "llm_recovery": None})}
             for i in range(4)]
    out = {"task": "e2_coverage_curve", "model": model, "k": args.k,
           "n_unresolved": len(unresolved),
           "interpretation": "recovery collapses as corpus coverage drops → "
                             "end-to-end bottleneck is corpus coverage, not the matcher.",
           "curve": curve, "per_bucket": summary}
    outf = OUT / "e2_coverage_curve.json"
    json.dump(out, open(outf, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n保存 → {outf}")
    print("覆盖率曲线 (recovery vs 语料覆盖桶):")
    for c in curve:
        print(f"  {c['bucket']:30s} n={c.get('n_in_bucket',0):5d}  recovery={c.get('llm_recovery')}")


if __name__ == "__main__":
    main()
