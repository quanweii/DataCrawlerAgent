#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
画 4 域设计矩阵图: 模态轴(Y) × 同质性轴(X).
诚实设计: X 用实测边同质性, Y 用模态构成的离散带(非杜撰标量),
每个点旁标真实模态 pip + 节点/边/类别规模.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#444"})

# ---- 数据(实测) ----
# x = edge homophily, y band = modality, mods = present modalities
DOMAINS = [
    # name, homophily, y_band, modalities, nnodes, nedges, ncls, note
    dict(name="tmdb",    x=0.237, y=1, mods=["image","text"],            n=13037, e=193346, c=18, note="constructed"),
    dict(name="steam",   x=0.526, y=1, mods=["image","text"],            n=2760,  e=54424,  c=8,  note="constructed"),
    dict(name="wikiart", x=0.894, y=0, mods=["image"],                   n=21968, e=520570, c=64, note="constructed"),
    dict(name="arxiv",   x=0.933, y=2, mods=["text","image","formula"],  n=45672, e=57590,  c=4,  note="natural citation"),
]
MOD_COLOR = {"text":"#2b6cb0", "image":"#dd6b20", "formula":"#805ad5"}
DOM_COLOR = {"tmdb":"#3182ce", "steam":"#38a169", "wikiart":"#e53e3e", "arxiv":"#805ad5"}
BAND_LABEL = {0:"image", 1:"image + text", 2:"text + image + formula"}

fig, ax = plt.subplots(figsize=(10.5, 6.2))

# 背景: 异质/同质分区
ax.axvspan(0.0, 0.5, color="#f7fafc", zorder=0)
ax.axvspan(0.5, 1.0, color="#edf2f7", zorder=0)
ax.axvline(0.5, ls="--", color="#a0aec0", lw=1, zorder=1)
ax.text(0.27, 2.62, "heterophilic", ha="center", color="#718096", fontsize=10, style="italic")
ax.text(0.78, 2.62, "homophilic",   ha="center", color="#718096", fontsize=10, style="italic")

# y 带
for yy in (0,1,2):
    ax.axhline(yy, color="#e2e8f0", lw=0.8, zorder=1)

# 画点 (Y 轴本身即模态带, 不再画 pip, 避免被误读为数据点)
for d in DOMAINS:
    # steam/tmdb 同带, 轻微竖向错开避免标签重叠
    yj = d["y"] + (0.10 if d["name"]=="steam" else (-0.10 if d["name"]=="tmdb" else 0))
    ax.scatter(d["x"], yj, s=520, color=DOM_COLOR[d["name"]],
               edgecolor="white", linewidth=2.0, zorder=5)
    # 标签左右放置
    right_side = d["x"] < 0.85
    dx = 0.020 if right_side else -0.020
    ha = "left" if right_side else "right"
    # 域名(粗体) + natural/constructed
    tag = "natural" if d["note"].startswith("natural") else "constructed"
    ax.text(d["x"]+dx, yj+0.13, d["name"], fontsize=14, fontweight="bold",
            color=DOM_COLOR[d["name"]], ha=ha, va="bottom", zorder=6)
    # 规模 + 边类型
    ax.text(d["x"]+dx, yj-0.13,
            f"{d['n']:,} nodes · {d['e']:,} edges · {d['c']} classes",
            fontsize=9, color="#4a5568", ha=ha, va="top", zorder=6)
    ax.text(d["x"]+dx, yj-0.30, f"[{tag}]", fontsize=8, color="#718096",
            ha=ha, va="top", style="italic", zorder=6)

# 注解: steam 与 tmdb 同模态不同同质性 -> 孤立同质性轴
ax.annotate("", xy=(0.526,1.10), xytext=(0.237,0.90),
            arrowprops=dict(arrowstyle="<->", color="#a0aec0", lw=1.2))
ax.text(0.38, 1.55, "same modality,\ndifferent homophily\n→ isolates the axis",
        fontsize=8.5, color="#718096", ha="center", style="italic")

ax.set_xlim(0.12, 1.04)
ax.set_ylim(-0.8, 3.1)
ax.set_yticks([0,1,2])
ax.set_yticklabels([BAND_LABEL[0], BAND_LABEL[1], BAND_LABEL[2]], fontsize=10.5)
ax.set_xticks([0.237,0.5,0.526,0.894,0.933])
ax.set_xticklabels(["0.24\n(tmdb)","0.50","0.53\n(steam)","0.89\n(wikiart)","0.93\n(arxiv)"], fontsize=8.5)
ax.set_xlabel("Edge homophily  (same-label edges / total)", fontsize=12, labelpad=8)
ax.set_ylabel("Modality profile", fontsize=12, labelpad=8)

ax.set_title("4-Domain MAG Design Matrix\nmodality × homophily — a controllable 2-axis benchmark",
             fontsize=13.5, fontweight="bold", pad=12)

ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
p = os.path.join(OUT, "design_matrix.png")
plt.savefig(p, dpi=300, bbox_inches="tight")
print("wrote", p)
