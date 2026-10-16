#!/usr/bin/env python3
"""提取全部实验结果为清洗后的汇总文件 extract_summary.json，供报告写作使用。防御式解析各种 schema。"""
import json, os, numpy as np
OM = "benchmark/results/openmag"
F = "benchmark/results/lp_ablation_final"
S = {}

def L(p):
    return json.load(open(p)) if os.path.exists(p) else None

def rows_with(d, *keys):
    """递归收集含指定字段的 dict。"""
    out = []
    def w(x):
        if isinstance(x, dict):
            if all(k in x for k in keys):
                out.append(x)
            for v in x.values(): w(v)
        elif isinstance(x, list):
            for v in x: w(v)
    w(d)
    return out

# ===== NC ablation (4x7) =====
d = L(f"{OM}/q1_full_modality_ablation.json")
nc = {}
for r in d["results"]:
    if "acc_mean" in r:
        nc[f"{r['model']}+{r['modality']}"] = (round(r["acc_mean"]*100,2), round(r["acc_std"]*100,2))
S["nc_ablation"] = nc

# ===== LP ablation (9x7) =====
S["lp_ablation"] = L(f"{F}/_lp_grid_clean.json")

# ===== Q5 effectiveness (9 models NC) =====
d = L(f"{OM}/q5_effectiveness.json")
eff = {}
for r in rows_with(d, "model", "acc_mean"):
    eff[r["model"]] = (round(r["acc_mean"]*100,2), round(r.get("acc_std",0)*100,2),
                       round(r.get("f1_mean",0)*100,2))
S["q5_effectiveness"] = eff

# ===== Q5b extended (12 models NC) =====
d = L(f"{OM}/q5b_extended_effectiveness.json")
ext = {}
for r in rows_with(d, "model", "acc_mean"):
    ext[r["model"]] = (round(r["acc_mean"]*100,2), round(r.get("acc_std",0)*100,2))
S["q5b_extended"] = ext

# ===== Q5c significance (10-run) =====
d = L(f"{OM}/q5c_significance.json")
sig = {}
pm = d.get("per_model", d) if isinstance(d, dict) else d
if isinstance(pm, list):
    for r in pm:
        if isinstance(r, dict) and "model" in r:
            sig[r["model"]] = {"acc": round(r.get("acc_mean",0)*100,2), "std": round(r.get("acc_std",0)*100,2), "n": len(r.get("accs",[]))}
elif isinstance(pm, dict):
    sig = pm
S["q5c_significance"] = sig

# ===== Q2 neighbors =====
d = L(f"{OM}/q2_neighbor_ablation.json")
if d:
    res = d.get("results", {})
    S["q2_neighbors"] = {k: (round(v["mean"]*100,2), round(v["std"]*100,3)) for k,v in res.items()}

# ===== Q3 finetune =====
d = L(f"{OM}/q3_finetune.json")
if d:
    S["q3_finetune"] = {r["config"]: round(r["test_acc_mean"]*100,2) for r in d.get("results",[])}

# ===== Q4 encoding comparison =====
d = L(f"{OM}/q4_encoding_comparison.json")
if d:
    enc = {}
    for scheme, v in d.items():
        if isinstance(v, dict) and "t2i_mrr" in v:
            enc[scheme] = {"t2i_mrr": round(v["t2i_mrr"]*100,3), "i2t_mrr": round(v["i2t_mrr"]*100,3),
                           "encoder": v.get("encoder","")}
    S["q4_encoding"] = enc

# ===== Q6 PEFT =====
d = L(f"{OM}/q6_peft.json")
if d:
    peft = {}
    for r in d.get("results", []):
        peft[r.get("method", r.get("config","?"))] = round(r.get("test_acc_mean", r.get("acc_mean",0))*100,2)
    S["q6_peft"] = peft

# ===== Q7 robustness =====
d = L(f"{OM}/q7_robustness.json")
rob = {}
if d:
    for model, mv in d.items():
        if not isinstance(mv, dict): continue
        rob[model] = {}
        for ntype, nv in mv.items():
            if not isinstance(nv, dict) or "results" not in nv: continue
            rob[model][ntype] = {rk: round(rv.get("mean",0)*100,2) for rk, rv in nv["results"].items()}
S["q7_robustness"] = rob

# ===== Q7b modality missing =====
d = L(f"{OM}/q7b_modality_missing.json")
mm = {}
if d:
    for model, mv in d.items() if isinstance(d,dict) else []:
        if isinstance(mv, dict):
            mm[model] = {k: round(v["acc_mean"]*100,2) for k,v in mv.items() if isinstance(v,dict) and "acc_mean" in v}
S["q7b_modality_missing"] = mm

# ===== Q8 edge noise =====
d = L(f"{OM}/q8_edge_noise.json")
en = {}
if d:
    for model, mv in d.items() if isinstance(d,dict) else []:
        if isinstance(mv, dict):
            res = mv.get("results", mv)
            en[model] = {rk: round(rv.get("mean",0)*100,2) for rk,rv in res.items() if isinstance(rv,dict)}
S["q8_edge_noise"] = en

# ===== Q9/Q10 efficiency =====
d = L(f"{OM}/q9_q10_efficiency.json")
S["q9_q10_efficiency"] = d if d else {}

# ===== Q5d retrieval =====
d = L(f"{OM}/q5d_retrieval_task.json")
rr = {}
if d:
    rr_raw = d.get("retrieval_results", d)
    if isinstance(rr_raw, list):
        for r in rr_raw:
            if isinstance(r, dict) and "model" in r:
                rr[r["model"]] = {"n2img_mrr": round(r.get("node2img_mrr_mean",0)*1000,2),
                                  "n2txt_mrr": round(r.get("node2txt_mrr_mean",0)*1000,2)}
    elif isinstance(rr_raw, dict):
        for k,v in rr_raw.items():
            if isinstance(v, dict): rr[k] = {kk:(round(vv,4) if isinstance(vv,float) else vv) for kk,vv in v.items()}
S["q5d_retrieval"] = rr

# ===== Q5e selfsup =====
d = L(f"{OM}/q5e_selfsupervised.json")
ss = {}
if d:
    for k,v in d.items():
        if isinstance(v, dict) and "acc_mean" in v:
            ss[k] = round(v["acc_mean"]*100,2)
S["q5e_selfsupervised"] = ss

# ===== few-shot (GCN text as headline + per-modality k-curve) =====
d = L(f"{OM}/few_shot_full.json")
fs = {}
for k,v in d.items():
    if isinstance(v, dict) and "acc_mean" in v:
        fs[k] = (round(v["acc_mean"]*100,2), round(v.get("acc_std",0)*100,2))
S["few_shot"] = fs

# ===== cross-domain transfer =====
d = L(f"{OM}/cross_domain_transfer.json")
cd = {}
for k,v in d.items():
    if isinstance(v, dict):
        cd[k] = {"src": round(v.get("source_test_acc_mean",0)*100,2),
                 "scratch": round(v.get("target_scratch_acc_mean",0)*100,2),
                 "finetune": round(v.get("target_finetune_acc_mean",0)*100,2),
                 "gain": round((v.get("target_finetune_acc_mean",0)-v.get("target_scratch_acc_mean",0))*100,2)}
S["cross_domain"] = cd

# ===== cross-domain LP =====
d = L(f"{OM}/cross_domain_link_prediction.json")
S["cross_domain_lp"] = d if d else {}

# ===== Q1b domain preference =====
d = L(f"{OM}/q1b_domain_preference.json")
S["q1b_domain_preference"] = d if d else {}

json.dump(S, open(f"{F}/_extract_summary.json","w"), indent=2, ensure_ascii=False, default=str)
print("saved _extract_summary.json with keys:", list(S.keys()))
for k,v in S.items():
    print(f"  {k}: {len(v) if hasattr(v,'__len__') else type(v)}")
