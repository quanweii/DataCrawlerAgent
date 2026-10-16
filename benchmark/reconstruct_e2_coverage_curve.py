#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 Jun20 已有的 E2 Part2 数据(146 条 unresolved-with-candidates,LLM 已跑过)
重建【真实】的 coverage 曲线——不花一次 LLM 调用。

Jun23 的 e2_coverage_curve.json 4 桶全 0 是 artifact(RateLimitError + 解析退化),
真实 recovery 由 Jun20 的 part2_rows 给出(总体 4/146=2.7%, precision 4/31=13%,
人工复核见 MANUAL_VERIFICATION_part2.md)。

逻辑:对每条 ref 重跑 retrieve_candidates 取 best_overlap_norm → 分桶 → 每桶 LLM 回收率。
桶内 n 小、correct 稀疏(总 4 对),曲线必然噪,但这是【真实的】数据,不是 fabricated。
"""
import os, sys, json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmark"))

from run_e2_citation_ablation import (load_metadata, retrieve_candidates, normalize_title,
                                      STOP, META_PATH)

OUT = ROOT / "benchmark" / "results" / "e2_citation_ablation"
JUN20 = OUT / "e2_results.json"


def coverage_bucket(best_overlap_norm, n_cand):
    if n_cand == 0:
        return ("out_of_corpus(no_candidate)", 0)
    if best_overlap_norm < 0.25:
        return ("low_coverage(<.25)", 1)
    if best_overlap_norm < 0.5:
        return ("mid_coverage(.25-.5)", 2)
    return ("high_coverage(>=.5)", 3)


def main():
    meta = load_metadata(META_PATH)
    t2i = meta["title_to_id"]; wi = meta["word_index"]
    id2t = meta["id_to_title"]; id2a = meta["id_to_authors"]

    d = json.load(open(JUN20))
    rows = d["part2_rows"]
    print(f"[load] {len(rows)} part2 rows (Jun20, LLM 已跑)")

    # 重算每条 best_overlap + 分桶,带 Jun20 的 llm_correct
    buckets = defaultdict(lambda: {"n": 0, "llm_proposed": 0, "llm_correct": 0,
                                   "overlaps": []})
    for r in rows:
        cands = retrieve_candidates(r["ref"], wi, t2i, id2t, id2a, K=20)
        nref = normalize_title(r["ref"])
        sig = {w for w in (set(nref.split()) - STOP) if len(w) >= 3}
        best_ov = max((c[3] for c in cands), default=0)
        best_norm = best_ov / max(len(sig), 1)
        bname, _ = coverage_bucket(best_norm, len(cands))
        buckets[bname]["n"] += 1
        buckets[bname]["overlaps"].append(best_norm)
        if r.get("llm"):
            buckets[bname]["llm_proposed"] += 1
        if r.get("llm_correct"):
            buckets[bname]["llm_correct"] += 1

    order = {0: "out_of_corpus(no_candidate)", 1: "low_coverage(<.25)",
             2: "mid_coverage(.25-.5)", 3: "high_coverage(>=.5)"}
    curve = []
    for i in range(4):
        b = order[i]; s = buckets.get(b, {"n":0,"llm_proposed":0,"llm_correct":0,"overlaps":[]})
        ov = s["overlaps"]
        rec = s["llm_correct"] / s["n"] if s["n"] else 0.0
        prec = s["llm_correct"] / s["llm_proposed"] if s["llm_proposed"] else 0.0
        mean_ov = sum(ov)/len(ov) if ov else 0.0
        curve.append({"bucket": b, "n_in_bucket": s["n"],
                      "llm_proposed": s["llm_proposed"], "llm_correct": s["llm_correct"],
                      "llm_recovery": round(rec, 3), "llm_precision": round(prec, 3),
                      "mean_best_overlap": round(mean_ov, 3)})

    out = {"task": "e2_coverage_curve_reconstructed",
           "source": "Jun20 e2_results.json part2_rows (no new LLM calls)",
           "model": d.get("model", "mimo-v2-flash"),
           "n_unresolved_with_candidates": len(rows),
           "note": ("Jun23 全 0 是 artifact(RateLimitError+解析)。真实总体 recovery=4/146=2.7%, "
                    "precision=4/31=13%。低回收根因=unresolved 被非-arxiv 经典/期刊引用主导"
                    "(见 MANUAL_VERIFICATION_part2.md),非匹配器能力不足。"),
           "interpretation": ("端到端引用边回收 ~3%,瓶颈是语料覆盖(非-arxiv 引用)+ OCR, "
                              "非 LLM 匹配器。→ agent 在引用边构建上仅边际增益,不构成 "
                              "'为何必须多智能体'的证据;作为诚实的数据系统发现保留。"),
           "curve": curve}
    outf = OUT / "e2_coverage_curve_reconstructed.json"
    json.dump(out, open(outf, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n真实 coverage 曲线 (Jun20 数据重建):")
    print(f"  {'bucket':30s} {'n':>5} {'prop':>5} {'corr':>5} {'rec':>6} {'prec':>6} {'meanOV':>7}")
    for c in curve:
        print(f"  {c['bucket']:30s} {c['n_in_bucket']:>5} {c['llm_proposed']:>5} "
              f"{c['llm_correct']:>5} {c['llm_recovery']:>6} {c['llm_precision']:>6} "
              f"{c['mean_best_overlap']:>7}")
    tot_n = sum(c["n_in_bucket"] for c in curve)
    tot_c = sum(c["llm_correct"] for c in curve)
    tot_p = sum(c["llm_proposed"] for c in curve)
    print(f"\n总体 recovery={tot_c}/{tot_n}={tot_c/tot_n:.3f}  precision={tot_c}/{tot_p}={tot_c/max(tot_p,1):.3f}")
    print(f"保存 → {outf}")


if __name__ == "__main__":
    main()
