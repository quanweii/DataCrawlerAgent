#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 5 (C4) — Steering 受控实验: GCN-MLP gap vs edge homophily (n=12).

数据来源(权威, 唯一): benchmark/results/steering/steering_causal_analysis_2026-07-12.json
设计: 同节点 + 同特征(MLP bit-identical), 只换边集 -> 同质性当旋钮.
12 点 = steam(5) + tmdb(4) + wikiart(3). 仅 3 域 -> cluster-robust effective n≈3.

正确性保证: 脚本内硬编码 12 个真实点, 并断言算出的 r / slope / intercept 与
权威 JSON 自洽 (r=0.846, slope=0.364, intercept=-0.332). 不一致则报错退出.
配色 Okabe-Ito 3 色, 已过 dataviz CVD 验证 (ΔE>=11).
输出: fig5_steering_gap_vs_homophily.{pdf,png} (300 dpi).
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 12 真实点 (homophily, gcn_f1, mlp_f1) —— 逐字抄自 steering_causal_analysis JSON
# ---------------------------------------------------------------------------
POINTS = [
    # steam (mlp_f1 = 0.5373379928174398)
    ("steam", "all",            0.5294906658826988, 0.30676131243241983, 0.5373379928174398),
    ("steam", "same_developer", 0.8166645908581392, 0.5683076159721048,  0.5373379928174398),
    ("steam", "same_publisher", 0.5947315938175171, 0.3845124086511829,  0.5373379928174398),
    ("steam", "shared_category",0.4400444380733945, 0.27394993732333794, 0.5373379928174398),
    ("steam", "shared_genre",   0.4373548087347065, 0.4734448941647778,  0.5373379928174398),
    # tmdb (mlp_f1 = 0.5038098552759842)
    ("tmdb", "all",             0.23708274285477848,0.2198102325722344,  0.5038098552759842),
    ("tmdb", "same_director",   0.32910222222222224,0.32536406801667916, 0.5038098552759842),
    ("tmdb", "shared_actor",    0.21712958152101303,0.22460703293453047, 0.5038098552759842),
    ("tmdb", "shared_genre",    0.24036727332349567,0.3462050123820315,  0.5038098552759842),
    # wikiart (mlp_f1 = 0.908768948605389)
    ("wikiart", "all",          0.8941487215936378, 0.9033615648220529,  0.908768948605389),
    ("wikiart", "same_artist",  0.8757693539005398, 0.8656359681515262,  0.908768948605389),
    ("wikiart", "shared_style", 1.0,                0.958162987373178,   0.908768948605389),
]

# 权威 JSON 报告值 (自洽断言用)
JSON_R_GAP   = 0.8460255488684291
JSON_SLOPE   = 0.3643947541530317
JSON_INTERC  = -0.3322874805865808
JSON_P       = 0.0005231914861170858

homo = np.array([p[2] for p in POINTS])
gap  = np.array([p[3] - p[4] for p in POINTS])     # GCN - MLP

# ---------------------------------------------------------------------------
# 正确性自洽断言
# ---------------------------------------------------------------------------
r = np.corrcoef(homo, gap)[0, 1]
A = np.vstack([homo, np.ones_like(homo)]).T
slope, interc = np.linalg.lstsq(A, gap, rcond=None)[0]
crossover = -interc / slope

assert abs(r - JSON_R_GAP) < 1e-3, f"r mismatch: {r:.4f} vs {JSON_R_GAP}"
assert abs(slope - JSON_SLOPE) < 1e-3, f"slope mismatch: {slope:.4f} vs {JSON_SLOPE}"
assert abs(interc - JSON_INTERC) < 1e-3, f"intercept mismatch: {interc:.4f} vs {JSON_INTERC}"
print(f"[OK] 自洽: r={r:.4f} (json {JSON_R_GAP:.4f}) | "
      f"slope={slope:.4f} (json {JSON_SLOPE:.4f}) | "
      f"intercept={interc:.4f} (json {JSON_INTERC:.4f}) | crossover={crossover:.3f}")

# 95% CI band of the regression mean response
from scipy import stats
n = len(homo)
xbar = homo.mean()
sxx = np.sum((homo - xbar) ** 2)
resid = gap - (slope * homo + interc)
s = np.sqrt(np.sum(resid ** 2) / (n - 2))
tval = stats.t.ppf(0.975, n - 2)
xs = np.linspace(0.15, 1.02, 200)
fit = slope * xs + interc
se = s * np.sqrt(1.0 / n + (xs - xbar) ** 2 / sxx)

# ---------------------------------------------------------------------------
# 画图
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5,
    "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
})
fig, ax = plt.subplots(figsize=(5.3, 3.8))

# 域配色 (Okabe-Ito, CVD-safe, 已验证)
DOMAIN = {"tmdb": ("#0072B2", "o"), "steam": ("#D55E00", "s"), "wikiart": ("#009E73", "^")}
for dom in ("tmdb", "steam", "wikiart"):
    color, mk = DOMAIN[dom]
    m = np.array([p[0] == dom for p in POINTS])
    ax.scatter(homo[m], gap[m], s=62, c=color, marker=mk, edgecolors="white",
               linewidths=0.7, zorder=4, label=f"{dom}")

# 回归 + CI band
ax.fill_between(xs, fit - tval * se, fit + tval * se, color="#888888",
                alpha=0.16, zorder=1, linewidth=0)
ax.plot(xs, fit, color="#333333", lw=2.0, zorder=2,
        label=f"OLS fit  (gap = {slope:.3f}·homo {interc:+.3f})")

# 零线 (GCN = MLP)
ax.axhline(0, color="#9aa0a6", lw=1.0, ls="--", zorder=1)
ax.text(0.158, 0.012, "GCN = MLP", color="#6b7178", fontsize=8, va="bottom")

# 交叉点标注: 贴在虚线顶端的纯文字(不穿点云, 无箭头)
ax.axvline(crossover, color="#c4c4c4", lw=0.9, ls=":", zorder=0)
ax.text(crossover, 0.082, f"crossover ≈ {crossover:.2f}", fontsize=7.8,
        color="#666", ha="center", va="bottom")

# 统计标注框 (ink 色, 非 series 色)
ax.text(0.975, 0.05,
        f"Pearson r = {r:.3f}\np = {JSON_P:.1e}\nn = {n}",
        transform=ax.transAxes, fontsize=9, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", lw=0.7))

ax.set_xlabel("Edge homophily", fontsize=10)
ax.set_ylabel("GCN − MLP  (NC macro-F1 gap)", fontsize=10)
ax.set_xlim(0.15, 1.02)
ax.set_ylim(-0.31, 0.10)
ax.grid(axis="y", color="#e6e6e6", lw=0.7, zorder=0)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.tick_params(length=3)

leg = ax.legend(loc="upper left", fontsize=8.2, frameon=True, framealpha=0.95,
                edgecolor="#dddddd", borderpad=0.5, labelspacing=0.35)
leg.get_frame().set_linewidth(0.6)

fig.tight_layout()
out = "/home/dev_qw/DataCrawlerAgent/docs/figures/fig5_steering_gap_vs_homophily"
fig.savefig(out + ".pdf", bbox_inches="tight")
fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
print(f"[OK] saved: {out}.pdf / .png")
