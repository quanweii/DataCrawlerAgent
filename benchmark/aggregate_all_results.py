#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全域结果聚合（T2⑥⑨⑫ + 论文头条表）—— 一键生成统一表 + 同质性拟合。

读取 newdomain_{nc,lp}/steering/leakage_free_nc/fusion_ablation/
newdomain_crossdomain/edge_relation_prediction 全部结果，产出：
  benchmark/results/aggregated_report.md   (人读)
  benchmark/results/figure_data.json        (画图用)
含：4 域 NC 统一表(acc/macroF1±std)、LP 主表(hard_degree 头条+饱和 gap)、
跨域迁移、融合消融、泄露修复对比、边关系预测、GNN-MLP gain vs homophily 拟合。
"""
import json
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[1] / "benchmark" / "results"
DOM = ["arxiv", "steam", "tmdb", "wikiart"]
HOMO = {"arxiv": 0.933, "steam": 0.529, "tmdb": 0.237, "wikiart": 0.894}
MODELS = ["mlp", "gcn", "gat", "graphsage", "gin", "gcnii"]


def L(path):
    p = R / path
    return json.load(open(p)) if p.exists() else None


def nc_table():
    """4 域 × 6 模型 × (text+image) acc/macroF1±std。arxiv 额外 text+image+formula。"""
    out = {}
    for d in DOM:
        t = L(f"newdomain_nc/nc_{d}_summary.json")
        if not t:
            continue
        out[d] = {}
        for m in MODELS:
            k = f"{m}+text+image"
            if k in t:
                out[d][m] = {"acc": t[k]["acc"], "f1": t[k]["f1"], "f1_std": t[k]["f1_std"],
                             "acc_std": t[k]["acc_std"]}
        # arxiv 三模态 bonus
        if "gcn+text+image+formula" in t:
            out[d]["gcn_tif"] = {kk: t["gcn+text+image+formula"][kk] for kk in ["acc", "f1", "f1_std", "acc_std"]}
    return out


def lp_table():
    """LP 主表：每域每模型 random/hard/hard_degree AUC + 饱和 gap(random-hard_degree)。"""
    out = {}
    for d in DOM:
        t = L(f"newdomain_lp/lp_{d}_summary.json")
        if not t:
            continue
        out[d] = {}
        for k, v in t.items():
            if "+" not in k:
                continue
            out[d][k] = {
                "random": v["random"]["auc_mean"], "random_std": v["random"]["auc_std"],
                "hard": v["hard"]["auc_mean"],
                "hard_degree": v["hard_degree"]["auc_mean"], "hard_degree_std": v["hard_degree"]["auc_std"],
                "sat_gap": v["random"]["auc_mean"] - v["hard_degree"]["auc_mean"],
            }
    return out


def homophily_fit():
    """收集 (homophily, gnn_mlp_gain) 点：E1 steering per-relation + 4 域 NC。"""
    pts = []
    st = L("steering/steering_all.json")
    if st:
        # 结构: steering_all[domain][relation] = {homophily, mlp+text_f1, gcn+text_f1, ...}
        for dom, slices in st.items():
            if not isinstance(slices, dict):
                continue
            for rel, m in slices.items():
                if not isinstance(m, dict) or "homophily" not in m:
                    continue
                mlp = m.get("mlp+text_f1")
                if mlp is None:
                    continue
                gnns = [m.get(f"{g}+text_f1") for g in ["gcn", "gat", "graphsage", "gin", "gcnii"]]
                gnns = [g for g in gnns if g is not None]
                if not gnns:
                    continue
                best_gnn = max(gnns)
                pts.append((float(m["homophily"]), float(best_gnn - mlp)))
    # 4 域 NC 域级点（gcn-mlp gain vs 域同质性；arxiv 补为 E1 第4点）
    nc = nc_table()
    for d in DOM:
        if d in nc and "gcn" in nc[d] and "mlp" in nc[d]:
            pts.append((HOMO[d], nc[d]["gcn"]["f1"] - nc[d]["mlp"]["f1"]))
    if len(pts) >= 3:
        xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
        coef = np.polyfit(xs, ys, 1)
        r = float(np.corrcoef(xs, ys)[0, 1])
        return {"points": pts, "n": len(pts), "slope": float(coef[0]),
                "intercept": float(coef[1]), "pearson_r": r}
    return {"points": pts, "n": len(pts)}


def leakfree_table():
    out = {}
    for d in DOM:
        t = L(f"leakage_free_nc/leakfree_{d}_text_image.json")
        if t:
            out[d] = {"lp_ceiling_original": t["lp_ceiling_original"]["acc"],
                      "lp_ceiling_pruned": t["lp_ceiling_pruned"]["acc"],
                      "majority": t["lp_ceiling_pruned"]["majority_acc"],
                      "nc": {m: {"acc": v["acc_mean"], "f1": v["f1_mean"]}
                             for m, v in t["nc"].items()}}
    return out


def fusion_table():
    out = {}
    for d in DOM:
        suf = "text+image+formula" if d == "arxiv" else "text+image"
        t = L(f"fusion_ablation/fusion_{d}_{suf}.json")
        if t:
            out[d] = {f: {"acc": v["acc_mean"], "f1": v["f1_mean"]}
                      for f, v in t["fusion_results"].items()}
            out[d]["_attn_weights"] = t.get("attention_weights", {})
    return out


def crossdomain_table():
    t = L("newdomain_crossdomain/transfer_ablation.json")
    if t:
        return t  # 完整结构
    t0 = L("newdomain_crossdomain/transfer_matrix.json")
    return t0


def main():
    fig = {"nc": nc_table(), "lp": lp_table(), "homophily_fit": homophily_fit(),
           "leakfree": leakfree_table(), "fusion": fusion_table(),
           "crossdomain": crossdomain_table(),
           "edgerel": {d: L(f"edge_relation_prediction/edgerel_{d}.json") for d in ["steam", "tmdb", "wikiart"]}}
    with open(R / "figure_data.json", "w") as f:
        json.dump(fig, f, indent=2)
    # ---- markdown 报告 ----
    md = ["# AMMAG 4-Domain Aggregated Results (auto-generated)\n",
          f"Domains: {DOM}\n"]
    # NC 表
    md.append("\n## 1. Node Classification (text+image, 5-seed mean±std; acc / macro-F1)\n")
    md.append("| domain(homo) | mlp | gcn | gat | graphsage | gin | gcnii |")
    md.append("|---|---|---|---|---|---|---|")
    nc = fig["nc"]
    for d in DOM:
        if d not in nc:
            continue
        row = [f"{d}({HOMO[d]})"]
        for m in MODELS:
            v = nc[d].get(m)
            row.append(f"{v['acc']:.3f}/{v['f1']:.3f}±{v['f1_std']:.3f}" if v else "-")
        md.append("| " + " | ".join(row) + " |")
    # LP 主表
    md.append("\n## 2. Link Prediction 主表 (AUC: random / hard / hard_degree + 饱和gap)\n")
    md.append("头条指标 = hard_degree；sat_gap = random - hard_degree（越大越饱和）\n")
    md.append("| domain | model | random | hard | **hard_degree** | sat_gap |")
    md.append("|---|---|---|---|---|---|")
    lp = fig["lp"]
    for d in DOM:
        for k, v in lp.get(d, {}).items():
            md.append(f"| {d} | {k} | {v['random']:.3f} | {v['hard']:.3f} | **{v['hard_degree']:.3f}±{v['hard_degree_std']:.3f}** | {v['sat_gap']:.3f} |")
    # 泄露修复
    md.append("\n## 3. wikiart 泄露修复 (T0②): LP-ceiling 无特征 原始图→剪枝图\n")
    lf = fig["leakfree"]
    for d in DOM:
        if d in lf:
            md.append(f"- {d}: LP-ceiling {lf[d]['lp_ceiling_original']:.3f} → "
                      f"{lf[d]['lp_ceiling_pruned']:.3f} (多数类 {lf[d]['majority']:.3f}); "
                      f"剪枝后 GCN {lf[d]['nc'].get('gcn',{}).get('f1',0):.3f} vs MLP {lf[d]['nc'].get('mlp',{}).get('f1',0):.3f}")
    # 同质性拟合
    md.append("\n## 4. GNN-MLP gain vs homophily 拟合 (T3⑫)\n")
    hf = fig["homophily_fit"]
    if "pearson_r" in hf:
        md.append(f"Pearson r = {hf['pearson_r']:.3f}, slope = {hf['slope']:.3f}, "
                  f"n = {hf['n']} 点 (E1 per-relation + 4 域 NC)")
    # 融合
    md.append("\n## 5. 模态融合消融 (T1④): concat/attention/mean/sum macro-F1\n")
    fu = fig["fusion"]
    for d in DOM:
        if d in fu:
            w = fu[d].get("_attn_weights", {})
            ws = " ".join(f"{k}={v:.2f}" for k, v in (w.get("attention") or {}).items()) if isinstance(w, dict) else ""
            cells = " ".join(f"{f}={fu[d][f]['f1']:.3f}" for f in ["concat", "attention", "mean", "sum"] if f in fu[d])
            md.append(f"- {d}: {cells}  attn_weights={ws}")
    # 边关系
    md.append("\n## 6. 边关系类型预测 (T3⑬)\n")
    for d in ["steam", "tmdb", "wikiart"]:
        e = fig["edgerel"].get(d)
        if e:
            md.append(f"- {d}: acc={e['acc_mean']:.3f}±{e['acc_std']:.3f} macro-F1={e['macro_f1_mean']:.3f} "
                      f"(majority={e['majority_acc']:.3f}, {e['n_relations']} relations)")
    with open(R / "aggregated_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("wrote", R / "aggregated_report.md")
    print("wrote", R / "figure_data.json")
    print("homophily fit:", hf.get("pearson_r"), "n=", hf.get("n"))


if __name__ == "__main__":
    main()
