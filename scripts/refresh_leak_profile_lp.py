#!/usr/bin/env python3
"""Refresh leak_profile_table.json LP section from new mean-fusion lp_{domain}_summary.json.
Structure metrics (homophily, lp_ceiling, edges_removed) are fusion-independent and unchanged."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
LP = ROOT / "benchmark" / "results" / "leak_profile_table.json"
NLP = ROOT / "benchmark" / "results" / "newdomain_lp"
DOM_MOD = {"arxiv": "text+image+formula", "steam": "text+image", "tmdb": "text+image",
           "wikiart": "text+image", "spotify": "text"}

d = json.load(open(LP))
for dom, info in d.items():
    mod = DOM_MOD.get(dom, "text+image")
    src = NLP / f"lp_{dom}_summary.json"
    if not src.exists():
        print(f"  {dom}: lp summary missing, skipping"); continue
    t = json.load(open(src))
    new_lp = {}
    for model in ["mlp", "gcn", "gat", "graphsage", "gin", "gcnii"]:
        r = t.get(f"{model}+{mod}")
        if not r:
            continue
        entry = {reg: round(r[reg]["auc_mean"], 3) for reg in ["random", "hard", "hard_degree"]}
        if "hitsk_random_filtered" in r:
            hk = r["hitsk_random_filtered"]
            entry["hitsk"] = {k: round(hk[k], 3) for k in ["hits@1", "hits@3", "hits@10", "mrr"] if k in hk}
        entry["delta_desat"] = round(r["random"]["auc_mean"] - r["hard_degree"]["auc_mean"], 3)
        entry["healthy_seeds"] = r.get("healthy_seeds")
        new_lp[model] = entry
    info["lp"] = new_lp
    info["_proj_note"] = "2026-07-15: mean-fusion (OpenMAG) + grad-clip + per-model-LR + 10 seeds; LP re-run."
    print(f"  {dom}: LP refreshed ({len(new_lp)} models)")

json.dump(d, open(LP, "w"), indent=2)
print(f"\nrefreshed {LP.name}")
