#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Option 1 杀手图 —— LP-ceiling 泄露诊断(三域三分制)。

每域一个面板:
  - 柱: 每条关系【单独】成图的 LP-ceiling(未剪枝 = 结构泄露信号)
  - 参考线:
      majority   (多数类基线;LP-ceiling ≤ 此即"无泄露")
      all-raw    (全关系未剪枝 = 构造图原貌)
      all-pruned (全关系跨切分剪枝后 = leak-free 协议修好的)
  - 标题: 域 + regime 分类 + 同质性 + 剪枝边比例

三域 = 三种泄露 regime(干净且诚实):
  tmdb     Suppressible      —— 关系选择即可修(0.413 → 选 shared_genre 0.255 ≈ oracle)
  wikiart  Structural        —— 每条关系都泄露,选择无解;剪枝才修(0.921 → 0.024)
  steam    Nothing-to-do     —— 本就 ≈ 多数类(0.557 ≈ 0.504)

数据全部现算(与 igc_all.json 同 split_seed=1),不硬编码。CPU-only。
"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.igc.graph_subset import (load_domain_base, measure_subset,
                                        subset_edge_index, per_rel_lp_profile)
from benchmark.data.splits import stratified_split, prune_cross_split_edges

OUT = PROJECT_ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
SPLIT_SEED = 1

# 与 make_figures.py 一致的域配色; regime 沿用同色(每域=一种 regime)
REGIME = {
    "tmdb":     ("Suppressible",  "#d1495b",
                 "relation selection fixes it:\nall 0.413 → shared_genre 0.255 (≈ oracle)"),
    "wikiart":  ("Structural",    "#2e8b57",
                 "every relation leaks;\ncross-split pruning fixes it (0.921 → 0.024)"),
    "steam":    ("Near-floor", "#3d7ea6",
                 "leakage below 0.08 threshold\n(0.557 vs majority 0.504) → constructor no-ops"),
}
DOMAIN_LABEL = {
    "tmdb": "TMDB\n(movies, 18 cls)",
    "steam": "Steam\n(games, 8 cls)",
    "wikiart": "WikiArt\n(art, 64 cls)",
}
DOMAIN_ORDER = ["tmdb", "steam", "wikiart"]   # 按 regime 叙事顺序


def diagnose(domain):
    """返回单域诊断 dict(全确定性, split_seed=1)。"""
    base = load_domain_base(domain, modality="text+image")
    all_rels = set(base["all_relations"])
    solo = per_rel_lp_profile(base, split_seed=SPLIT_SEED)          # {rel: LP-ceiling 未剪枝}
    raw = measure_subset(base, all_rels, split_seed=SPLIT_SEED, prune=False)
    pruned = measure_subset(base, all_rels, split_seed=SPLIT_SEED, prune=True)
    # 剪枝边比例
    ei, _, n_raw = subset_edge_index(base, all_rels)
    masks = stratified_split(base["y"], ratios=(0.6, 0.2, 0.2), seed=SPLIT_SEED)
    ei_p = prune_cross_split_edges(ei, masks)
    n_pruned = int(ei_p.size(1) // 2)
    prune_frac = 1.0 - n_pruned / max(n_raw, 1)
    # 关系按 solo LP-ceiling 升序(低泄露→高泄露)排,便于读图
    rels_sorted = sorted(solo, key=lambda r: solo[r])
    return {
        "domain": domain,
        "relations_sorted": rels_sorted,
        "solo_lp": {r: solo[r] for r in rels_sorted},
        "all_raw_lp": round(raw["lp_ceiling"], 3),
        "all_pruned_lp": round(pruned["lp_ceiling"], 3),
        "majority": round(raw["majority_acc"], 3),
        "homophily": round(raw["homophily"], 3),
        "n_edges_raw": n_raw,
        "n_edges_pruned": n_pruned,
        "prune_frac": round(prune_frac, 3),
    }


# ---------- 计算 + dump ----------
diag = {d: diagnose(d) for d in DOMAIN_ORDER}
with open(OUT / "leakage_diagnostic_data.json", "w") as f:
    json.dump(diag, f, indent=2, ensure_ascii=False)
print("wrote leakage_diagnostic_data.json")
for d in DOMAIN_ORDER:
    r = diag[d]
    print(f"\n[{d}] regime={REGIME[d][0]}  homo={r['homophily']}  "
          f"majority={r['majority']}  all_raw={r['all_raw_lp']}  "
          f"all_pruned={r['all_pruned_lp']}  prune={r['prune_frac']*100:.0f}%")
    for rel in r["relations_sorted"]:
        print(f"    solo {rel:18s} LP={r['solo_lp'][rel]}")


# ---------- 画图 ----------
fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4), sharey=True)
for ax, dom in zip(axes, DOMAIN_ORDER):
    r = diag[dom]
    regime_name, regime_color, regime_note = REGIME[dom]
    rels = r["relations_sorted"]
    vals = [r["solo_lp"][x] for x in rels]
    x = np.arange(len(rels))
    bars = ax.bar(x, vals, width=0.55, color=regime_color, edgecolor="black",
                  linewidth=0.6, alpha=0.92, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold", zorder=4)
    # 参考线
    ax.axhline(r["majority"], color="#888", ls="--", lw=1.6, zorder=2)
    ax.axhline(r["all_raw_lp"], color="#222", ls=":", lw=1.8, zorder=2)
    ax.axhline(r["all_pruned_lp"], color="#1b7837", ls="-.", lw=1.8, zorder=2)
    # 轴
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("_", "\n") for r in rels], fontsize=8.5)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-0.6, len(rels) - 0.4)
    ax.grid(True, axis="y", alpha=0.2, zorder=0)
    ax.set_title(f"{dom}  —  {regime_name}\n"
                 f"homo {r['homophily']} · prune {r['prune_frac']*100:.0f}% edges",
                 fontsize=10.5, fontweight="bold", color=regime_color)
    # regime 注释(面板内右下/左下)
    ax.text(0.98, 0.04, regime_note, transform=ax.transAxes, fontsize=8,
            ha="right", va="bottom", color="#333", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=regime_color, alpha=0.85))

axes[0].set_ylabel("LP-ceiling  (label propagation, no features)\n"
                   "← lower = less label leakage", fontsize=10.5)

# 共享图例
legend_elems = [
    Line2D([0], [0], color="#888", ls="--", lw=1.6, label="majority baseline (no leakage)"),
    Line2D([0], [0], color="#222", ls=":", lw=1.8, label="all-relations, raw construction graph"),
    Line2D([0], [0], color="#1b7837", ls="-.", lw=1.8, label="all-relations, after cross-split pruning"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#999", markeredgecolor="black",
           markersize=11, label="per-relation solo LP-ceiling (raw)"),
]
fig.legend(handles=legend_elems, loc="lower center", ncol=4, fontsize=9.5,
           bbox_to_anchor=(0.5, -0.02), frameon=True)
fig.suptitle("LP-ceiling leakage diagnostic across three construction regimes\n"
             "bars = each relation alone; lines = full-graph references",
             fontsize=12.5, fontweight="bold", y=1.02)
plt.tight_layout(rect=(0, 0.06, 1, 0.98))
plt.savefig(OUT / "leakage_diagnostic.png", dpi=300, bbox_inches="tight")
plt.close()
print("\nwrote leakage_diagnostic.png")
