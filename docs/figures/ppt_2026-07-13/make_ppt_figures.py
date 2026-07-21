#!/usr/bin/env python3
"""PPT figures v2 (2026-07-13) — fixes overlap + missing labels.
Rules enforced this version:
  1. ALL bar value labels go ABOVE the bar (outside, dark text) — never inside.
     → no white-text-on-white overflow on short bars.
  2. EVERY bar gets a value label (no unlabeled bars).
  3. Dense groups use rotated labels to avoid horizontal collision.
  4. Fig 1 metadata moved to two-line y-ticks (no text inside short bars).
Output: docs/figures/ppt_2026-07-13/*.png  (300 DPI, paste-ready)
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 16,
    "axes.titlesize": 19,
    "axes.titleweight": "bold",
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "savefig.pad_inches": 0.3,
})

OUT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]

C_TMDB, C_STEAM, C_SPOT, C_WIKI, C_ARXIV = "#E45756", "#F2A93B", "#4C78A8", "#54A24B", "#9D6AC9"
BLUE, RED, GREY, GREEN, ORANGE = "#4C78A8", "#E45756", "#9A9A9A", "#54A24B", "#F2A93B"


def save(fig, name):
    p = OUT / name
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p)


def label_bars(ax, bars, values, fmt="{:.3f}", fs=11, dy=0.012, rot=0, color="#222"):
    """Put a value label ABOVE every bar (outside). Never inside."""
    ymax = ax.get_ylim()[1]
    for b, v in zip(bars, values):
        if v is None or v == 0:
            continue
        ax.text(b.get_x() + b.get_width() / 2, v + dy * ymax,
                fmt.format(v), ha="center", va="bottom",
                fontsize=fs, fontweight="bold", rotation=rot, color=color)


# =====================================================================
# Fig 0 — overview cards (C1–C4)
# =====================================================================
def fig_overview():
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.axis("off")
    cards = [
        ("C1  Four+1 MAG Datasets", "arxiv · steam · tmdb · wikiart\n(+spotify)\nhomophily 0.24 → 0.93\nunique formula modality\nleak-profile metadata", C_ARXIV),
        ("C2  Leak-free Protocol\n(moat)", "cross-split edge pruning\n+ LP-ceiling double-report\ngreedy = no-LLM = oracle\nde-leaking = measurement,\nnot LLM", RED),
        ("C3  Declarative Framework\n+ Skill", "1 YAML = 1 domain\n94.6% reuse (3 domains)\nspotify: 0 Python LOC\nbuild-mag-dataset skill\n7 stages, zero-code", GREEN),
        ("C4  Benchmark Value\n(steering crown)", "same nodes/features,\nswap edges only\nGCN−MLP gap vs homophily\nr = 0.846, p = 0.0005\nLP / cluster / few-shot", BLUE),
    ]
    for i, (title, body, col) in enumerate(cards):
        x0 = 0.012 + i * 0.247
        ax.add_patch(plt.Rectangle((x0, 0.06), 0.232, 0.86, fc="#F7F8FA", ec=col, lw=2.4))
        ax.add_patch(plt.Rectangle((x0, 0.805), 0.232, 0.115, fc=col, ec=col, lw=2.4))
        ax.text(x0 + 0.116, 0.862, title, ha="center", va="center", color="white",
                fontweight="bold", fontsize=11)
        ax.text(x0 + 0.116, 0.74, body, ha="center", va="top", fontsize=10.2, linespacing=1.45)
    ax.text(0.5, 0.975, "DataCrawlerAgent — NeurIPS D&B Contribution Map",
            ha="center", fontsize=14.5, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(fig, "00_overview.png")


# =====================================================================
# Fig 1 — homophily spectrum (metadata in 2-line y-ticks, value at bar end)
# =====================================================================
def fig_homophily_spectrum():
    domains = ["tmdb", "steam", "spotify", "wikiart", "arxiv"]
    homo = [0.237, 0.530, 0.741, 0.894, 0.933]
    meta = ["13.0k nodes · 18 classes · text+img",
            "2.8k nodes · 8 classes · text+img",
            "10.0k nodes · 6 classes · text",
            "22.0k nodes · 64 classes · text+img",
            "45.7k nodes · 4 classes · text+img+formula"]
    colors = [C_TMDB, C_STEAM, C_SPOT, C_WIKI, C_ARXIV]
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    y = np.arange(len(domains))
    bars = ax.barh(y, homo, color=colors, height=0.6, edgecolor="white")
    # value at bar end (outside)
    for b, h in zip(bars, homo):
        ax.text(h + 0.014, b.get_y() + b.get_height() / 2, f"{h:.2f}",
                va="center", ha="left", fontweight="bold", fontsize=13)
    # two-line y-tick: domain (bold) + metadata
    ax.set_yticks(y)
    ax.set_yticklabels([f"{d.upper()}\n{m}" for d, m in zip(domains, meta)], fontsize=11.5)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Edge homophily  (fraction of edges connecting same-label nodes)")
    ax.axvline(0.5, ls=":", color=GREY, lw=1.2)
    ax.text(0.5, len(domains) - 0.3, "homo=0.5 (neutral)", color=GREY, fontsize=10, ha="center")
    ax.set_title("Four+1 MAG benchmarks span a wide homophily spectrum (natural sources, not tuned)")
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    save(fig, "01_homophily_spectrum.png")


# =====================================================================
# Fig 2 — LP-ceiling collapse (all bars labeled on top)
# =====================================================================
def fig_lpceiling_collapse():
    domains = ["tmdb\n(0.24)", "steam\n(0.53)", "arxiv\n(0.93)", "wikiart\n(0.89)"]
    orig = [0.413, 0.557, 0.673, 0.921]
    pruned = [0.154, 0.504, 0.261, 0.024]
    majority = [0.205, 0.504, 0.288, 0.025]
    x = np.arange(len(domains)); w = 0.26
    fig, ax = plt.subplots(figsize=(11, 5.2))
    b1 = ax.bar(x - w, orig, w, label="LP-ceiling BEFORE pruning (leaky)", color=RED, alpha=0.92)
    b2 = ax.bar(x, pruned, w, label="LP-ceiling AFTER pruning (leak-free)", color=GREEN, alpha=0.95)
    b3 = ax.bar(x + w, majority, w, label="Majority-class baseline", color=GREY, alpha=0.85)
    label_bars(ax, b1, orig, fs=10.5)
    label_bars(ax, b2, pruned, fs=10.5)
    label_bars(ax, b3, majority, fs=10.5)
    ax.annotate("0.921 → 0.024\n(structural leak eliminated)",
                xy=(3 - w, 0.921), xytext=(1.7, 1.0),
                fontsize=11, color=RED, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.8))
    ax.set_xticks(x); ax.set_xticklabels(domains)
    ax.set_ylabel("Label-propagation accuracy (no features)")
    ax.set_ylim(0, 1.18)
    ax.set_title("C2 leak-free protocol: pruned LP-ceiling ≤ majority class on all 4 domains")
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    save(fig, "02_lpceiling_collapse.png")


# =====================================================================
# Fig 3 — steering crown (scatter + regression)
# =====================================================================
def fig_steering_crown():
    d = json.load(open(ROOT / "benchmark/results/steering/steering_causal_analysis_2026-07-12.json"))
    pts = d["points"]
    homo = np.array([p["homophily"] for p in pts])
    gap = np.array([p["gcn_f1"] - p["mlp_f1"] for p in pts])
    gcn = np.array([p["gcn_f1"] for p in pts])
    dom = [p["domain"] for p in pts]
    cmap = {"steam": C_STEAM, "tmdb": C_TMDB, "wikiart": C_WIKI}
    fit = d["linear_fit_gap_vs_homophily"]
    xs = np.linspace(0.2, 0.95, 50)
    ys = fit["slope"] * xs + fit["intercept"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    # left: gap vs homophily
    for h, g, dm in zip(homo, gap, dom):
        ax1.scatter(h, g, c=cmap[dm], s=110, edgecolor="black", lw=0.7, zorder=3)
    ax1.plot(xs, ys, color=BLUE, lw=2.4, label="linear fit (r=0.846)")
    ax1.axhline(0, color=GREY, ls="--", lw=1)
    ax1.axvline(fit["crossover_homo"], color=RED, ls=":", lw=1.4)
    ax1.text(fit["crossover_homo"] + 0.01, -0.33, f"crossover\nh≈{fit['crossover_homo']:.2f}",
             color=RED, fontsize=10)
    ax1.set_xlabel("Edge homophily (edges swapped, same nodes/features)")
    ax1.set_ylabel("GCN − MLP  macro-F1 gap")
    ax1.set_title("Same nodes/features, only edges change → gap tracks homophily")
    ax1.text(0.04, 0.06, "r = 0.846  [boot 95% CI 0.60, 0.96]\np = 0.0005, n = 12",
             transform=ax1.transAxes, va="bottom", fontsize=11,
             bbox=dict(boxstyle="round", fc="#F0F4FA", ec=BLUE))
    handles = [Line2D([0], [0], marker='o', ls='', mfc=c, mec='k', markersize=9, label=dm)
               for dm, c in cmap.items()]
    handles.append(Line2D([0], [0], color=BLUE, lw=2.4, label="linear fit (r=0.846)"))
    ax1.legend(handles=handles, loc="upper left", frameon=False, fontsize=10.5)
    ax1.set_ylim(-0.42, 0.18)

    # right: GCN F1 vs homophily + flat MLP reference per domain
    for h, g, dm in zip(homo, gcn, dom):
        ax2.scatter(h, g, c=cmap[dm], s=110, edgecolor="black", lw=0.7, zorder=3)
    seen = {}
    for p in pts:
        seen.setdefault(p["domain"], p["mlp_f1"])
    for dm, mlpv in seen.items():
        hs = [p["homophily"] for p in pts if p["domain"] == dm]
        ax2.plot([min(hs), max(hs)], [mlpv, mlpv], color=cmap[dm], ls="--", lw=1.6, alpha=0.8)
    ax2.set_xlabel("Edge homophily")
    ax2.set_ylabel("macro-F1")
    ax2.set_title("GNN benefit rises with homophily; MLP flat (ignores edges)")
    ax2.text(0.04, 0.04, "dashed = MLP (edge-independent)\nGCN-F1 vs homo r = 0.913",
             transform=ax2.transAxes, fontsize=10.5, color="#444")
    save(fig, "03_steering_crown.png")


# =====================================================================
# Fig 4 — code reuse (all bars labeled on top)
# =====================================================================
def fig_code_reuse():
    domains = ["spotify\n(zero-code)", "wikiart", "tmdb", "steam", "arxiv\n(legacy)"]
    reuse = [100.0, 95.0, 94.8, 94.6, 67.7]
    colors = [C_SPOT, C_WIKI, C_TMDB, C_STEAM, C_ARXIV]
    fig, ax = plt.subplots(figsize=(11, 5.0))
    x = np.arange(len(domains))
    bars = ax.bar(x, reuse, color=colors, width=0.6, edgecolor="white")
    label_bars(ax, bars, reuse, fmt="{:.1f}%", fs=12, dy=0.01)
    ax.axhline(94.0, ls="--", color=GREEN, lw=1.4)
    ax.text(0.2, 95.4, "headline 94% (3 co-attribute domains)", color=GREEN, fontsize=10.5)
    ax.set_xticks(x); ax.set_xticklabels(domains)
    ax.set_ylabel("Shared-skeleton reuse  =  shared / (shared + per-domain incremental)")
    ax.set_ylim(0, 112)
    ax.set_title("C3 reusability: 1 YAML = 1 domain (spotify 0 Python LOC); arxiv footnoted")
    save(fig, "04_code_reuse.png")


# =====================================================================
# Fig 5 — fair modality ablation (ALL bars labeled, rotated to avoid overlap)
# =====================================================================
def fig_modality_ablation():
    domains = ["steam", "tmdb", "wikiart", "arxiv"]
    text = [0.662, 0.667, 0.916, 0.906]
    image = [0.640, 0.554, 0.769, 0.819]
    ti = [0.683, 0.647, 0.906, 0.895]
    tif = [0.0, 0.0, 0.0, 0.896]   # formula exists only on arxiv
    homo_lbl = [0.53, 0.24, 0.89, 0.93]
    x = np.arange(len(domains)); w = 0.2
    fig, ax = plt.subplots(figsize=(12, 5.4))
    b1 = ax.bar(x - 1.5 * w, text, w, label="text", color=BLUE)
    b2 = ax.bar(x - 0.5 * w, image, w, label="image", color=ORANGE)
    b3 = ax.bar(x + 0.5 * w, ti, w, label="text+image", color=GREEN)
    b4 = ax.bar(x + 1.5 * w, tif, w, label="text+image+formula (arxiv only)", color=C_ARXIV)
    # label EVERY bar on top, rotated 90° so dense groups don't collide
    label_bars(ax, b1, text, fmt="{:.3f}", fs=9.5, dy=0.006, rot=90)
    label_bars(ax, b2, image, fmt="{:.3f}", fs=9.5, dy=0.006, rot=90)
    label_bars(ax, b3, ti, fmt="{:.3f}", fs=9.5, dy=0.006, rot=90)
    label_bars(ax, b4, tif, fmt="{:.3f}", fs=9.5, dy=0.006, rot=90)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n(homo {h:.2f})" for d, h in zip(domains, homo_lbl)])
    ax.set_ylabel("Node-classification accuracy (GCNII, leak-free)")
    ax.set_ylim(0, 1.08)
    ax.set_title("Fair modality ablation (all projected to 768-d): text alone is strongest")
    ax.legend(loc="lower left", frameon=False, fontsize=10.5, ncol=2)
    save(fig, "05_modality_ablation.png")


# =====================================================================
# Fig 6 — leak-free main NC table (ALL 3 bars labeled)
# =====================================================================
def fig_main_nc_table():
    order = ["tmdb", "steam", "wikiart", "arxiv"]
    homo = [0.24, 0.53, 0.89, 0.93]
    mlp = [0.662, 0.694, 0.933, 0.899]
    gcn = [0.487, 0.611, 0.900, 0.904]
    bestgnn = [0.643, 0.681, 0.922, 0.908]
    x = np.arange(len(order)); w = 0.26
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    b1 = ax.bar(x - w, mlp, w, label="MLP (no edges)", color=BLUE)
    b2 = ax.bar(x, gcn, w, label="GCN", color=RED)
    b3 = ax.bar(x + w, bestgnn, w, label="best GNN", color=GREEN)
    # value on top of every bar (rotated to avoid collision)
    label_bars(ax, b1, mlp, fmt="{:.3f}", fs=9.5, dy=0.006, rot=90)
    label_bars(ax, b2, gcn, fmt="{:.3f}", fs=9.5, dy=0.006, rot=90)
    # best-GNN: value + delta
    ymax = 1.04
    for b, v, m in zip(b3, bestgnn, mlp):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006 * ymax,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9.5,
                fontweight="bold", rotation=90)
        delta = v - m
        ax.text(b.get_x() + b.get_width() / 2 + 0.012, v + 0.006 * ymax,
                f"Δ{delta:+.3f}", ha="left", va="bottom", fontsize=9.5, fontweight="bold",
                color="black" if delta >= 0 else RED, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels([f"{d}\nhomo {h:.2f}" for d, h in zip(order, homo)])
    ax.set_ylabel("Accuracy (leak-free pruned graph)")
    ax.set_ylim(0, 1.10)
    ax.set_title("Leak-free NC table: MLP competitive (leak removed); GNN helps only at highest homophily")
    ax.legend(loc="lower right", frameon=False, fontsize=11)
    save(fig, "06_main_nc_table.png")


# =====================================================================
# Fig 7 — robustness (all bars labeled)
# =====================================================================
def fig_robustness():
    domains = ["steam\n(homo 0.53)", "tmdb\n(homo 0.24)", "wikiart\n(homo 0.89)"]
    feat = [0.083, 0.212, 0.090]
    edge = [0.108, 0.234, 0.327]
    x = np.arange(len(domains)); w = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    b1 = ax.bar(x - w / 2, feat, w, label="feature noise (rows→Gaussian)", color=ORANGE)
    b2 = ax.bar(x + w / 2, edge, w, label="edge noise (add random edges)", color=RED)
    label_bars(ax, b1, feat, fmt="{:.3f}", fs=11)
    label_bars(ax, b2, edge, fmt="{:.3f}", fs=11)
    ax.set_xticks(x); ax.set_xticklabels(domains)
    ax.set_ylabel("|Δ accuracy| at ρ=0.8 noise  (lower = more robust)")
    ax.set_ylim(0, 0.40)
    ax.set_title("Robustness: edge noise hurts more than feature noise; tmdb most sensitive")
    ax.legend(loc="upper left", frameon=False, fontsize=10.5)
    save(fig, "07_robustness.png")


if __name__ == "__main__":
    fig_overview()
    fig_homophily_spectrum()
    fig_lpceiling_collapse()
    fig_steering_crown()
    fig_code_reuse()
    fig_modality_ablation()
    fig_main_nc_table()
    fig_robustness()
    print("\nALL FIGURES v2 DONE →", OUT)
