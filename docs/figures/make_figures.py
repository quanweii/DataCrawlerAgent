#!/usr/bin/env python3
"""
TRACK D — the three money figures from existing JSON results.
CPU-only. Reads:
  benchmark/results/newdomain_nc/nc_<domain>_<model>_<modality>_supervised.json
  benchmark/results/newdomain_nc/nc_<domain>_summary.json
  benchmark/results/newdomain_lp/lp_<domain>_summary.json
  benchmark/configs/newdomains.yaml  (homophily)
Dumps docs/figures/figure_data.json + 3 PNGs at 300 dpi.
Nothing is fabricated: missing -> None, recorded in PROBLEMS.
"""
import json, os, glob
from collections import OrderedDict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

NC_DIR = "benchmark/results/newdomain_nc"
LP_DIR = "benchmark/results/newdomain_lp"
OUT = "docs/figures"

# ---------------------------------------------------------------------------
# Ground-truth config values (from newdomains.yaml, verified 2026-06-19)
# ---------------------------------------------------------------------------
DOMAINS = OrderedDict([
    ("tmdb",    {"homo": 0.237, "label": "TMDB\n(movies, 18 cls)", "short": "tmdb"}),
    ("steam",   {"homo": 0.526, "label": "Steam\n(games, 8 cls)",  "short": "steam"}),
    ("wikiart", {"homo": 0.894, "label": "WikiArt\n(art, 64 cls)", "short": "wikiart"}),
])

MODELS = ["mlp", "gcn", "gat", "graphsage", "gin", "gcnii"]
GNN_MODELS = ["gcn", "gat", "graphsage", "gin", "gcnii"]
MODALITIES = ["text", "image", "text+image"]
NEG_TYPES = ["random", "hard", "hard_degree"]

PROBLEMS = []

# ---------------------------------------------------------------------------
# 1. NC grid: domain -> model -> modality -> test_acc (and std)
# ---------------------------------------------------------------------------
def load_nc(domain):
    grid = {}
    for model in MODELS:
        for mod in MODALITIES:
            f = os.path.join(NC_DIR, f"nc_{domain}_{model}_{mod}_supervised.json")
            if not os.path.exists(f):
                PROBLEMS.append(f"NC missing file: {f}")
                grid.setdefault(model, {})[mod] = None
                continue
            d = json.load(open(f))
            acc = d.get("test_acc_mean", d.get("acc"))
            std = d.get("test_acc_std", d.get("acc_std"))
            grid.setdefault(model, {})[mod] = {"acc": acc, "std": std}
    return grid

NC = {dom: load_nc(dom) for dom in DOMAINS}

# sanity note: steam summary.json is truncated (only gcnii+text present), but
# per-model JSONs are complete -> we rely on per-model JSONs, not the summary.
if list(json.load(open(os.path.join(NC_DIR, "nc_steam_summary.json"))).keys()) != ["gcnii+text"]:
    pass
else:
    PROBLEMS.append(
        "nc_steam_summary.json is TRUNCATED (only gcnii+text entry). "
        "Per-model JSONs were used instead to reconstruct the full grid."
    )

# ---------------------------------------------------------------------------
# 2. LP grid: domain -> model -> neg_type -> auc
#    LP only ran with text+image modality (per config).
# ---------------------------------------------------------------------------
def load_lp(domain):
    f = os.path.join(LP_DIR, f"lp_{domain}_summary.json")
    d = json.load(open(f))
    grid = {}
    # Per newdomains.yaml, LP config models = [mlp, gcn, gat, graphsage, gin];
    # gcnii is intentionally NOT in the LP model list. So a missing gcnii entry
    # is by-design, not a data gap. We record all present models.
    lp_run_models = [m for m in MODELS if f"{m}+text+image" in d]
    for model in MODELS:
        key = f"{model}+text+image"
        if key not in d:
            if model != "gcnii":
                PROBLEMS.append(f"LP UNEXPECTED missing model entry: {domain}/{key}")
            continue
        grid[model] = {}
        for nt in NEG_TYPES:
            if nt in d[key]:
                grid[model][nt] = d[key][nt].get("auc_mean")
            else:
                grid[model][nt] = None
                PROBLEMS.append(f"LP missing neg type: {domain}/{key}/{nt}")
    return grid

LP = {dom: load_lp(dom) for dom in DOMAINS}

# ---------------------------------------------------------------------------
# Derived quantities for the figures
# ---------------------------------------------------------------------------

# --- Fig 1: GNN value vs homophily ---
# y = best-GNN-text acc - MLP-text acc, for each domain
fig1_points = OrderedDict()
for dom in DOMAINS:
    mlp_text = NC[dom]["mlp"]["text"]["acc"]
    gnn_text = [NC[dom][m]["text"]["acc"] for m in GNN_MODELS if NC[dom][m]["text"]["acc"] is not None]
    best_gnn = max(gnn_text) if gnn_text else None
    best_gnn_model = GNN_MODELS[int(np.argmax(gnn_text))] if gnn_text else None
    delta = (best_gnn - mlp_text) if (best_gnn is not None and mlp_text is not None) else None
    fig1_points[dom] = {
        "homophily": DOMAINS[dom]["homo"],
        "mlp_text_acc": mlp_text,
        "best_gnn_text_acc": best_gnn,
        "best_gnn_text_model": best_gnn_model,
        "delta": delta,
    }

# --- Fig 2: desaturation. Best LP model per domain (by random AUC) shown across neg types,
#     plus grouped bars for ALL models so the collapse is visible.
fig2_best_model = {}
for dom in DOMAINS:
    rand_aucs = {m: LP[dom][m]["random"] for m in LP[dom] if LP[dom][m].get("random") is not None}
    best = max(rand_aucs, key=rand_aucs.get) if rand_aucs else None
    fig2_best_model[dom] = best

# desaturation drop: random - hard_degree for best model (magnitude of collapse)
fig2_drop = {}
for dom in DOMAINS:
    m = fig2_best_model[dom]
    if m is None:
        fig2_drop[dom] = None
        continue
    r = LP[dom][m]["random"]; hd = LP[dom][m]["hard_degree"]
    fig2_drop[dom] = (r - hd) if (r is not None and hd is not None) else None

# --- Fig 3: modality heatmap. cell = best NC acc over all models, per domain x modality ---
fig3_grid = OrderedDict()
for dom in DOMAINS:
    fig3_grid[dom] = {}
    for mod in MODALITIES:
        accs = [NC[dom][m][mod]["acc"] for m in MODELS
                if NC[dom][m][mod]["acc"] is not None]
        fig3_grid[dom][mod] = max(accs) if accs else None

# ---------------------------------------------------------------------------
# Dump figure_data.json
# ---------------------------------------------------------------------------
figure_data = OrderedDict([
    ("source", {
        "nc_per_model": NC_DIR,
        "nc_summary": os.path.join(NC_DIR, "nc_<domain>_summary.json"),
        "lp_summary": os.path.join(LP_DIR, "lp_<domain>_summary.json"),
        "homophily": "benchmark/configs/newdomains.yaml (verified 2026-06-19)",
        "note": "LP runs used text+image modality only (per newdomains.yaml).",
    }),
    ("homophily", {dom: DOMAINS[dom]["homo"] for dom in DOMAINS}),
    ("nc_grid", NC),
    ("lp_grid", LP),
    ("fig1_homophily_curve", fig1_points),
    ("fig2_desaturation", {
        "best_model_per_domain": fig2_best_model,
        "collapse_drop_random_minus_hard_degree": fig2_drop,
        "neg_types_order": NEG_TYPES,
        "lp_models": [m for m in MODELS if m != "gcnii"],  # LP config: mlp,gcn,gat,graphsage,gin
        "note": "gcnii is NOT in LP model list (config lp.models = mlp,gcn,gat,graphsage,gin).",
    }),
    ("fig3_modality_heatmap", fig3_grid),
    ("problems", PROBLEMS),
])
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "figure_data.json"), "w") as f:
    json.dump(figure_data, f, indent=2)
print("wrote figure_data.json")

# ===========================================================================
# FIGURE 1 — homophily_curve.png
# ===========================================================================
fig, ax = plt.subplots(figsize=(7.5, 5.2))
xs = [fig1_points[d]["homophily"] for d in DOMAINS]
ys = [fig1_points[d]["delta"] for d in DOMAINS]
colors = {"tmdb": "#d1495b", "steam": "#3d7ea6", "wikiart": "#2e8b57"}
# sort by homophily for a clean curve
order = sorted(DOMAINS, key=lambda d: fig1_points[d]["homophily"])
xs = [fig1_points[d]["homophily"] for d in order]
ys = [fig1_points[d]["delta"] for d in order]

ax.plot(xs, ys, "-o", color="#444", lw=2, ms=2, zorder=2, alpha=0.5)
for d in order:
    p = fig1_points[d]
    ax.scatter(p["homophily"], p["delta"], s=180, color=colors[d], zorder=3,
               edgecolor="black", linewidth=1.0)
    offy = 0.012
    ax.annotate(
        f"{d}\n({p['best_gnn_text_model']}+text)\nΔ={p['delta']:+.3f}",
        xy=(p["homophily"], p["delta"]),
        xytext=(0, 26 if d != "steam" else -42),
        textcoords="offset points",
        ha="center", fontsize=9, fontweight="bold",
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.8),
    )
ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
ax.set_xlabel("Edge homophily", fontsize=12)
ax.set_ylabel("GNN value:  best-GNN-text acc  −  MLP-text acc", fontsize=11)
ax.set_title("Where does message passing help?\nGNN vs MLP as homophily varies", fontsize=12, fontweight="bold")
ax.set_xlim(0.15, 1.0)
ax.set_ylim(min(ys) - 0.03, max(ys) + 0.05)
ax.grid(True, alpha=0.25)
ax.text(0.27, ax.get_ylim()[1]*0.9, "heterophilic", fontsize=8, style="italic", color="gray")
ax.text(0.82, ax.get_ylim()[1]*0.9, "homophilic", fontsize=8, style="italic", color="gray")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "homophily_curve.png"), dpi=300, bbox_inches="tight")
plt.close()
print("wrote homophily_curve.png")

# ===========================================================================
# FIGURE 2 — desaturation_curve.png  (grouped bars: model x neg-type per domain)
# ===========================================================================
lp_models = ["mlp", "gcn", "gat", "graphsage", "gin"]  # as run
fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), sharey=True)
neg_colors = {"random": "#4c72b0", "hard": "#dd8452", "hard_degree": "#c44e52"}
best_marker = {d: fig2_best_model[d] for d in DOMAINS}

for ax, dom in zip(axes, DOMAINS):
    x = np.arange(len(lp_models))
    width = 0.26
    for i, nt in enumerate(NEG_TYPES):
        vals = [LP[dom][m][nt] for m in lp_models]
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      label=nt, color=neg_colors[nt], edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, vals):
            if v is not None:
                ax.text(b.get_x() + b.get_width()/2, v + 0.006, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=6.5, rotation=90)
    # mark best model with asterisk under x label
    labels = []
    for m in lp_models:
        lab = m + ("\n★" if m == best_marker[dom] else "")
        labels.append(lab)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    drop = fig2_drop[dom]
    title_extra = f"\nΔ(random−hard_deg)={drop:+.3f}" if drop is not None else ""
    ax.set_title(f"{dom}  (homo {DOMAINS[dom]['homo']}){title_extra}", fontsize=10, fontweight="bold")
    ax.set_ylim(0.65, 1.01)
    ax.grid(True, axis="y", alpha=0.25)
    if dom == "tmdb":
        ax.set_ylabel("Link-prediction test AUC", fontsize=11)

axes[0].legend(title="negatives", fontsize=8, title_fontsize=9, loc="lower left")
fig.suptitle("LP desaturation: AUC under random → hard → hard-degree negatives\n"
             "(★ = best model by random-AUC; wikiart shows the saturation collapse, "
             "hard negatives reveal the real gap)", fontsize=11, fontweight="bold", y=1.04)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "desaturation_curve.png"), dpi=300, bbox_inches="tight")
plt.close()
print("wrote desaturation_curve.png")

# ===========================================================================
# FIGURE 3 — modality_heatmap.png  rows=domain cols=modality, cell=best NC acc
# ===========================================================================
mat = np.full((len(DOMAINS), len(MODALITIES)), np.nan)
for i, dom in enumerate(DOMAINS):
    for j, mod in enumerate(MODALITIES):
        v = fig3_grid[dom][mod]
        if v is not None:
            mat[i, j] = v

fig, ax = plt.subplots(figsize=(6.5, 4.2))
im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0.4, vmax=1.0)
ax.set_xticks(range(len(MODALITIES)))
ax.set_xticklabels(MODALITIES, fontsize=11)
rowlabels = []
for dom in DOMAINS:
    rowlabels.append(f"{dom}\nhomo={DOMAINS[dom]['homo']}")
ax.set_yticks(range(len(DOMAINS)))
ax.set_yticklabels(rowlabels, fontsize=10)
for i in range(len(DOMAINS)):
    for j in range(len(MODALITIES)):
        v = mat[i, j]
        if not np.isnan(v):
            txtcol = "white" if v < 0.78 else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color=txtcol, fontsize=11, fontweight="bold")
        else:
            ax.text(j, i, "n/a", ha="center", va="center", color="red", fontsize=9)
ax.set_title("Best NC accuracy by domain × modality\n"
             "(cell = max over {mlp,gcn,gat,graphsage,gin,gcnii}; rows annotated with homophily)",
             fontsize=11, fontweight="bold")
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("best test accuracy", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "modality_heatmap.png"), dpi=300, bbox_inches="tight")
plt.close()
print("wrote modality_heatmap.png")

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print("\n===== FIG 1: homophily curve (GNN value) =====")
for d in order:
    p = fig1_points[d]
    print(f"  {d:8s} homo={p['homophily']:.3f}  MLP-text={p['mlp_text_acc']:.4f}  "
          f"best-GNN-text={p['best_gnn_text_model']}={p['best_gnn_text_acc']:.4f}  "
          f"Δ={p['delta']:+.4f}")

print("\n===== FIG 2: desaturation (best model per domain) =====")
for d in DOMAINS:
    m = fig2_best_model[d]
    r = LP[d][m]["random"]; h = LP[d][m]["hard"]; hd = LP[d][m]["hard_degree"]
    print(f"  {d:8s} best={m:10s}  random={r:.4f}  hard={h:.4f}  hard_degree={hd:.4f}  "
          f"drop(r-hd)={r-hd:+.4f}")

print("\n===== FIG 3: modality heatmap (best NC acc) =====")
for d in DOMAINS:
    print(f"  {d:8s}  " + "  ".join(f"{mod}={fig3_grid[d][mod]:.4f}" for mod in MODALITIES))

print("\nPROBLEMS:")
for p in PROBLEMS:
    print("  -", p)
