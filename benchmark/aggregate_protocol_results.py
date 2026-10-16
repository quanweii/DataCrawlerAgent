#!/usr/bin/env python3
"""Aggregate four-view node-classification results into JSON and Markdown."""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark" / "results" / "newdomain_nc" / "protocol_matrix"
DOMAINS = ("arxiv", "steam", "tmdb", "wikiart")
PROTOCOLS = ("transductive", "split_disjoint", "temporal", "group_inductive")


def validation_mean(metrics):
    values = metrics.get("val_accs")
    if not values:
        raise ValueError("Validation scores are missing. Replay with run_validation_selected_nc.py; do not infer them from test accuracy.")
    return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(RESULTS))
    args = parser.parse_args()
    results_dir = Path(args.input_dir)
    aggregate = {}
    rows = []
    sensitivity = []
    for domain in DOMAINS:
        path = results_dir / f"nc_{domain}_protocol_matrix.json"
        if not path.exists():
            aggregate[domain] = {"status": "pending", "path": str(path.relative_to(ROOT))}
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        aggregate[domain] = obj
        # Prefer the richest modality requested in the run.
        modality = obj["modalities"][-1]
        for protocol in PROTOCOLS:
            cell = obj["results"].get(f"{protocol}/{modality}")
            if not cell:
                continue
            models = cell["models"]
            best_name, best = max(models.items(), key=lambda kv: validation_mean(kv[1]))
            mlp = models.get("mlp", {}).get("acc")
            gnn_name = max((m for m in models if m != "mlp"), key=lambda m: validation_mean(models[m]))
            gap = models[gnn_name]["acc"] - mlp if mlp is not None else None
            rows.append({
                "domain": domain, "protocol": protocol, "modality": modality,
                "best_model": best_name, "selection_val_acc": validation_mean(best),
                "best_acc": best["acc"],
                "best_f1": best["f1"], "mlp_acc": mlp, "best_minus_mlp": gap,
                "edges": cell["directed_edges"], "split_sizes": cell["split_sizes"],
            })
        trans = obj["results"].get(f"transductive/{modality}")
        if trans:
            reference_model = max(trans["models"], key=lambda m: validation_mean(trans["models"][m]))
            reference = trans["models"][reference_model]
            for protocol in PROTOCOLS[1:]:
                other = obj["results"].get(f"{protocol}/{modality}")
                if not other or reference_model not in other["models"]:
                    continue
                compared = other["models"][reference_model]
                diffs = np.asarray(reference["accs"]) - np.asarray(compared["accs"])
                rng = np.random.RandomState(20260910)
                boot = np.asarray([
                    rng.choice(diffs, size=len(diffs), replace=True).mean()
                    for _ in range(10000)
                ])
                names = sorted(set(trans["models"]) & set(other["models"]))
                a = np.asarray([trans["models"][m]["acc"] for m in names])
                b = np.asarray([other["models"][m]["acc"] for m in names])
                ra = np.argsort(np.argsort(a)).astype(float)
                rb = np.argsort(np.argsort(b)).astype(float)
                rank_corr = float(np.corrcoef(ra, rb)[0, 1]) if len(names) > 1 else None
                view_stats = obj.get("protocol_manifest", {}).get("views", {}).get(protocol, {})
                sensitivity.append({
                    "domain": domain, "protocol": protocol,
                    "reference_model": reference_model,
                    "transductive_minus_protocol_acc": float(diffs.mean()),
                    "paired_seed_bootstrap_95ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                    "model_rank_correlation": rank_corr,
                    "train_missing_classes": view_stats.get("train_missing_classes"),
                    "unassigned_nodes": view_stats.get("unassigned_nodes"),
                })
    out = {"selection": "mean validation accuracy; test reporting only", "rows": rows, "sensitivity": sensitivity, "domains": aggregate}
    (results_dir / "protocol_matrix_all.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [
        "# Four-Protocol Benchmark Matrix",
        "",
        "> Full topology is a standard transductive condition, not automatically leakage. "
        "The other three views measure robustness to structural exposure, time shift, and unseen groups.",
        "",
        "| Dataset | Protocol | Modality | Validation-selected model | Test accuracy | Macro-F1 | MLP | Validation-selected GNN−MLP | Directed edges | Train/Val/Test |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        mlp = "-" if r["mlp_acc"] is None else f"{r['mlp_acc']:.4f}"
        md.append(
            f"| {r['domain']} | {r['protocol']} | {r['modality']} | {r['best_model']} | "
            f"{r['best_acc']:.4f} | {r['best_f1']:.4f} | {mlp} | {r['best_minus_mlp']:+.4f} | "
            f"{r['edges']} | {'/'.join(map(str, r['split_sizes']))} |")
    md += [
        "", "## Protocol sensitivity (fixed validation-selected transductive model)", "",
        "| Dataset | Compared view | Reference model | Transductive−view accuracy | Paired-seed bootstrap 95% CI | Model-rank corr. | Missing train classes | Unassigned nodes |",
        "|---|---|---|---:|---|---:|---:|---:|",
    ]
    for r in sensitivity:
        lo, hi = r["paired_seed_bootstrap_95ci"]
        md.append(
            f"| {r['domain']} | {r['protocol']} | {r['reference_model']} | "
            f"{r['transductive_minus_protocol_acc']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
            f"{r['model_rank_correlation']:.3f} | {r['train_missing_classes']} | {r['unassigned_nodes']} |")
    pending = [d for d, x in aggregate.items() if x.get("status") == "pending"]
    if pending:
        md += ["", "## Pending", "", ", ".join(pending)]
    (results_dir / "protocol_matrix_all.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {results_dir / 'protocol_matrix_all.json'}")
    print(f"wrote {results_dir / 'protocol_matrix_all.md'}")


if __name__ == "__main__":
    main()
