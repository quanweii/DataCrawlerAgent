#!/usr/bin/env python3
"""Re-aggregate NC results after the mean-fusion + grad-clip + per-model-LR + 10-seed fix.

Inputs : benchmark/results/newdomain_nc/nc_{domain}_summary.json (10 init-seeds, paired)
         + nc_arxiv_summary_part2.json merged into nc_arxiv_summary.json
Outputs: benchmark/results/newdomain_nc/main_table_significance_2026-07-15.json
         benchmark/results/newdomain_nc/modality_ablation_2026-07-15.json

Significance: paired Wilcoxon over init-seeds (n=10 → can clear the old n=5 floor p=0.0625).
"""
import json
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
NC = ROOT / "benchmark" / "results" / "newdomain_nc"
DOMAINS = ["steam", "tmdb", "wikiart", "arxiv", "spotify"]
GNN = ["gcn", "gat", "graphsage", "gin", "gcnii"]


def merge_arxiv():
    """Merge nc_arxiv_summary_part2.json (extra modalities) into nc_arxiv_summary.json."""
    main = NC / "nc_arxiv_summary.json"
    part2 = NC / "nc_arxiv_summary_part2.json"
    if not part2.exists():
        return
    a = json.load(open(main))
    b = json.load(open(part2))
    a["table"].update(b["table"])
    a["init_seeds"] = b.get("init_seeds", a.get("init_seeds"))
    a["merged_part2"] = True
    json.dump(a, open(main, "w"), indent=2)
    print(f"merged arxiv part2: +{len(b['table'])} cells -> {len(a['table'])} total")
    part2.rename(part2.with_suffix(".json.merged"))


def load(dom):
    p = NC / f"nc_{dom}_summary.json"
    if not p.exists():
        return None
    return json.load(open(p))


def paired(gnn_accs, mlp_accs):
    a, b = np.array(gnn_accs), np.array(mlp_accs)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    try:
        w, p = stats.wilcoxon(a, b)
    except ValueError:
        w, p = np.nan, np.nan
    return {"delta": float(np.mean(a - b)), "p": float(p) if not np.isnan(p) else None,
            "n": int(n)}


def main():
    merge_arxiv()
    sig = {"task": "main_table_significance_2026-07-15_MEANFIX",
           "test": "paired Wilcoxon signed-rank over init-seeds",
           "fix": "mean-fusion + grad-clip + per-model-LR(GIN 0.005) + 10 seeds",
           "domains": {}}
    mod = {"task": "modality_ablation_2026-07-15_MEANFIX", "domains": {}}
    for dom in DOMAINS:
        s = load(dom)
        if s is None:
            print(f"  {dom}: MISSING"); continue
        t = s["table"]
        sig["domains"][dom] = {"modality": "text+image", "init_seeds": s.get("init_seeds"),
                               "mean_acc": {}, "gnn_vs_mlp": {}}
        mlp = t.get("mlp+text+image") or t.get("mlp+text")
        for m in GNN:
            cell = t.get(f"{m}+text+image") or t.get(f"{m}+text")
            if cell:
                sig["domains"][dom]["mean_acc"][m] = cell["acc"]
                if mlp:
                    sig["domains"][dom]["gnn_vs_mlp"][m] = paired(cell["accs"], mlp["accs"])
        if mlp:
            sig["domains"][dom]["mean_acc"]["mlp"] = mlp["acc"]
        # modality ablation (text / image / text+image[, formula])
        mods = ["text", "image", "text+image"]
        if dom == "arxiv":
            mods += ["formula", "text+image+formula"]
        mod["domains"][dom] = {}
        for m in GNN + ["mlp"]:
            mod["domains"][dom][m] = {}
            for mm in mods:
                c = t.get(f"{m}+{mm}")
                if c:
                    mod["domains"][dom][m][mm] = {"acc": c["acc"], "std": c["acc_std"]}
        # multi - text summary for stable models
        d = mod["domains"][dom]
        for m in ["gcn", "gcnii", "graphsage", "gin", "mlp"]:
            if "text" in d.get(m, {}) and "text+image" in d.get(m, {}):
                d[m]["(multi-text)"] = round(d[m]["text+image"]["acc"] - d[m]["text"]["acc"], 4)
        print(f"  {dom}: seeds={s.get('init_seeds')} | " + " ".join(
            f"{m}={sig['domains'][dom]['mean_acc'].get(m,'?'):.3f}" if isinstance(sig['domains'][dom]['mean_acc'].get(m),float) else f"{m}=NA"
            for m in GNN + ["mlp"]))

    json.dump(sig, open(NC / "main_table_significance_2026-07-15.json", "w"), indent=2)
    json.dump(mod, open(NC / "modality_ablation_2026-07-15.json", "w"), indent=2)
    print(f"\nsaved main_table_significance_2026-07-15.json + modality_ablation_2026-07-15.json")


if __name__ == "__main__":
    main()
