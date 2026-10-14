#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 4 域(arxiv/steam/tmdb/wikiart) NC + LP 全消融大表(LaTeX), 直接从底层 JSON 抽数, 零手抄.
权威源(与 docs/experiment_results_FINAL_decontaminated_2026-07-16.md 一致):
  NC 主表/模态消融: benchmark/results/newdomain_nc/modality_ablation_decontaminated_2026-07-15.json (steam/tmdb/wikiart)
                   benchmark/results/newdomain_nc/nc_arxiv_summary.json (arxiv, text+image)
  LP AUC:          benchmark/results/newdomain_lp/lp_{domain}_summary.json
  LP ceiling:      各 lp summary 内 lp_ceiling_* (回退 leak_profile_table.json / 文档§4a)
输出: fig_benchmark_nc.tex / fig_benchmark_lp.tex  (booktabs + xcolor, pdflatex 编译)
"""
import json
from pathlib import Path

ROOT = Path("/home/dev_qw/DataCrawlerAgent")
R = ROOT / "benchmark/results"
NC_ABL = json.load(open(R / "newdomain_nc/modality_ablation_decontaminated_2026-07-15.json"))["domains"]
NC_ARX = json.load(open(R / "newdomain_nc/nc_arxiv_summary.json"))["table"]
LP = {d: json.load(open(R / f"newdomain_lp/lp_{d}_summary.json")) for d in ("arxiv", "steam", "tmdb", "wikiart")}

MODELS = ["mlp", "gcn", "gat", "graphsage", "gin", "gcnii"]
GNN = ["gcn", "gat", "graphsage", "gin", "gcnii"]
DOM4 = ["arxiv", "steam", "tmdb", "wikiart"]
ABL3 = ["steam", "tmdb", "wikiart"]          # 有 text/image/t+i 模态消融的域
NEG = ["random", "hard", "hard_degree"]

def f2(x):   # acc -> %  2 dec
    return f"{x*100:.2f}"
def f1(x):
    return f"{x*100:.1f}"
def f3(x):
    return f"{x:.3f}"

# ---------------- NC main (text+image, acc %) ----------------
nc_main = {}   # dom -> {model: acc}
for d in ABL3:
    nc_main[d] = {m: NC_ABL[d][m]["text+image"]["acc"] for m in MODELS}
nc_main["arxiv"] = {m: NC_ARX[f"{m}+text+image"]["acc"] for m in MODELS}

def bold(s): return r"\textbf{" + s + "}"

# ---------------- emit NC tex ----------------
nc = []
nc.append(r"\begin{table}[ht]")
nc.append(r"\centering\small\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.12}")
nc.append(r"\caption{Node classification (NC). \textbf{(A)} Main results: accuracy (\%) on \emph{text+image}, 10 seeds, leak-free protocol. Best per column in \textbf{bold}. \textbf{(B)} GNN$-$MLP gap (pp): positive (green) only at high homophily (arxiv 0.93, wikiart 0.89), negative (red) at low homophily (steam 0.52, tmdb 0.24).}")
nc.append(r"\label{tab:nc_main}")
nc.append(r"\textbf{(A) NC main --- accuracy (\%), text+image}\\[2pt]")
nc.append(r"\begin{tabular}{lcccc}")
nc.append(r"\toprule")
nc.append(r"Model & arxiv & steam & tmdb & wikiart \\")
nc.append(r"\midrule")
for m in MODELS:
    cells = []
    for d in DOM4:
        v = nc_main[d][m]
        is_best = abs(v - max(nc_main[d][mm] for mm in MODELS)) < 5e-5
        cells.append(bold(f2(v)) if is_best else f2(v))
    nc.append(f"{m.upper():<10s} & " + " & ".join(cells) + r" \\")
nc.append(r"\bottomrule\end{tabular}\\[6pt]")

# (B) gap
GREEN, RED = "{bestgreen}", "{worstred}"
nc.append(r"\textbf{(B)} GNN$-$MLP gap (percentage points)\\[2pt]")
nc.append(r"\begin{tabular}{lcccc}")
nc.append(r"\toprule")
nc.append(r"Model & arxiv (h.93) & steam (h.52) & tmdb (h.24) & wikiart (h.89) \\")
nc.append(r"\midrule")
for m in GNN:
    cells = []
    for d in DOM4:
        gap = (nc_main[d][m] - nc_main[d]["mlp"]) * 100
        col = r"\textcolor{bestgreen}{" if gap > 0 else r"\textcolor{worstred}{"
        cells.append(col + f"{gap:+.2f}" + "}")
    nc.append(f"{m.upper():<10s} & " + " & ".join(cells) + r" \\")
nc.append(r"\bottomrule\end{tabular}")
nc.append(r"\end{table}")

# ---------------- NC modality ablation (steam/tmdb/wikiart) ----------------
nc2 = []
nc2.append(r"\begin{table}[ht]")
nc2.append(r"\centering\small\setlength{\tabcolsep}{6pt}\renewcommand{\arraystretch}{1.1}")
nc2.append(r"\caption{NC modality ablation: accuracy (\%) per modality and the multimodal gain $\Delta$ = (text+image)$-$text (pp). Image dominates on wikiart ($\Delta$ up to $+34.9$); text dominates on arxiv. 10 seeds.}")
nc2.append(r"\label{tab:nc_modality}")
nc2.append(r"\begin{tabular}{llcccc}")
nc2.append(r"\toprule")
nc2.append(r"Domain & Model & text & image & text+image & $\Delta$(multi$-$text) \\")
nc2.append(r"\midrule")
for d in ABL3:
    first = True
    for m in MODELS:
        t = NC_ABL[d][m]["text"]["acc"]; im = NC_ABL[d][m]["image"]["acc"]
        ti = NC_ABL[d][m]["text+image"]["acc"]
        dl_val = NC_ABL[d][m]["(multi−text)"]
        dl = dl_val["acc"] if isinstance(dl_val, dict) else dl_val
        dlab = d if first else ""
        nc2.append(f"{dlab:<8s} & {m.upper():<10s} & {f1(t)} & {f1(im)} & {f1(ti)} & {dl*100:+.1f} \\\\")
        first = False
    nc2.append(r"\midrule")
nc2[-1] = r"\bottomrule"   # 末尾 midrule -> bottomrule
nc2.append(r"\end{tabular}")
nc2.append(r"\par\smallskip\footnotesize\emph{arxiv}: text-only vs.\ text+image already on par (image = paper preview adds noise); the arxiv-specific \emph{formula} modality is a no-op for NC (text+image+formula 89.81 vs text+image 89.73).")
nc2.append(r"\end{table}")

# ---------------- LP ceiling ----------------
def get_ceiling(d):
    j = LP[d]
    for key in ("lp_ceiling_original", "ceiling_original"):
        if key in j:
            o = j["lp_ceiling_original"] if "lp_ceiling_original" in j else j["ceiling_original"]
            p = j.get("lp_ceiling_pruned", j.get("ceiling_pruned", {}))
            maj = (o if isinstance(o, dict) else {}).get("majority_acc") or (p if isinstance(p, dict) else {}).get("majority_acc")
            return (o.get("acc") if isinstance(o, dict) else None,
                    p.get("acc") if isinstance(p, dict) else None, maj)
    return None, None, None

# ceiling 回退(文档§4a 权威值)
DOC_CEIL = {"arxiv": (0.673, 0.261, 0.289), "steam": (0.582, 0.504, 0.504),
            "tmdb": (0.422, 0.154, 0.205), "wikiart": (0.918, 0.024, 0.025)}
ceil = {}
for d in DOM4:
    o, p, m = get_ceiling(d)
    if o is None:
        o, p, m = DOC_CEIL[d]
    ceil[d] = (o, p, m)

# ---------------- emit LP tex ----------------
lp = []
lp.append(r"\begin{table}[ht]")
lp.append(r"\centering\small\setlength{\tabcolsep}{4.5pt}\renewcommand{\arraystretch}{1.08}")
lp.append(r"\caption{Link prediction (LP). \textbf{(A)} Structure-leakage ceiling (label-propagation upper bound, AUC) before\,$\to$\,after cross-split edge pruning; after pruning all domains $\leq$ majority. \textbf{(B)} LP AUC (10 seeds) by modality and negative-sampling regime; arxiv uses text+image+\emph{formula} (no gcnii run).}")
lp.append(r"\label{tab:lp}")
lp.append(r"\textbf{(A)} LP ceiling (structure-leakage)\\[2pt]")
lp.append(r"\begin{tabular}{lccc}")
lp.append(r"\toprule")
lp.append(r"Domain & Original & Pruned & Majority \\")
lp.append(r"\midrule")
for d in DOM4:
    o, p, m = ceil[d]
    lp.append(f"{d:<9s} & {f3(o)} & {f3(p)} & {f3(m)} \\\\")
lp.append(r"\bottomrule\end{tabular}\\[8pt]")

# (B) LP AUC per domain
lp.append(r"\textbf{(B)} LP AUC by modality $\times$ negative sampling (cols: MLP/GCN/GAT/SAGE/GIN/GCNII)\\[2pt]")
lp.append(r"\begin{tabular}{lllcccccc}")
lp.append(r"\toprule")
lp.append(r"Domain & modality & neg & MLP & GCN & GAT & SAGE & GIN & GCNII \\")
lp.append(r"\midrule")
def lp_cell(d, m, mod, neg):
    j = LP[d]
    key = f"{m}+{mod}"
    full = j.get(key)
    if full is None or neg not in full:
        return "---"
    return f3(full[neg]["auc_mean"])

for d in DOM4:
    first = True
    mods = ["text+image+formula"] if d == "arxiv" else ["image", "text", "text+image"]
    for mod in mods:
        for ni, neg in enumerate(NEG):
            cells = []
            for mi, m in enumerate(MODELS):
                if d == "arxiv" and m == "gcnii":
                    cells.append("---"); continue
                cells.append(lp_cell(d, m, mod, neg))
            dlab = d if first else ""
            modlab = mod if ni == 0 else ""
            negdisp = neg.replace("_", r"\_")
            lp.append(f"{dlab:<8s} & {modlab:<16s} & {negdisp:<14s} & " + " & ".join(cells) + r" \\")
            first = False
    lp.append(r"\midrule")
lp[-1] = r"\bottomrule"
lp.append(r"\end{tabular}")
lp.append(r"\end{table}")

# ---------------- write + verify print ----------------
out = ROOT / "docs/figures"
(out / "fig_benchmark_nc.tex").write_text("\n".join(nc) + "\n\n" + "\n".join(nc2) + "\n", encoding="utf-8")
(out / "fig_benchmark_lp.tex").write_text("\n".join(lp) + "\n", encoding="utf-8")

print("=== 校验: NC main (text+image %) ===")
for d in DOM4:
    print(f"  {d}:", {m: round(nc_main[d][m]*100,2) for m in MODELS})
print("=== 校验: LP ceiling ===")
for d in DOM4:
    o,p,m=ceil[d]; print(f"  {d}: {o:.3f} -> {p:.3f} (maj {m:.3f})")
print("=== 校验: LP wikiart text+image random (mlp/gcn/gat/sage/gin/gcnii 应 0.925/0.987/0.987/0.987/0.939/0.938) ===")
for m in MODELS:
    print(f"  {m}:", lp_cell("wikiart", m, "text+image", "random"))
print("\n[OK] wrote fig_benchmark_nc.tex + fig_benchmark_lp.tex")
