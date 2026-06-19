#!/usr/bin/env python3
"""
Rerun problematic LP configs (RevGAT/GIN/GT multi-modal) with fix:
1. Keep lr=0.01 (original, works well for simple models GCN/GAT/GraphSAGE)
2. Gradient clipping (max_norm=1.0) — stabilizes training for RevGAT/GIN/GT

Root cause: Without gradient clipping, some runs of RevGAT/GIN/GT on
high-dim multi-modal input (1536d/2304d) suffer gradient explosion
(training crashes at 44-80s with MRR ~0.40), dragging down the mean.
Gradient clipping prevents these crash runs without hurting models
that already converge well.

Note: lr=0.001 was tested (lp_ablation_fixed_v3) — it fixed the
problematic models but hurt simple models (GCN -0.100, GAT -0.059)
because they early-stopped before converging.

Only reruns the 6 problematic configs:
- RevGAT: text+formula, text+image+formula
- GIN: text+formula, text+image+formula
- GT: text+formula, text+image+formula

Plus 3 well-performing configs as control:
- GCN: text+image+formula (should stay ~0.730)
- GAT: text+image+formula (should stay ~0.651)
- GraphSAGE: text+image+formula (should stay ~0.675)
"""
import os
import sys
import json
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from benchmark.data.dataset import AMMAGDataset
from benchmark.tasks.link_prediction import run_link_prediction

# Problematic configs that need rerun
PROBLEMATIC_CONFIGS = [
    # (model, modality, expected_old_mrr)
    ("revgat", "text+formula", 0.470),
    ("revgat", "text+image+formula", 0.532),
    ("gin", "text+formula", 0.611),
    ("gin", "text+image+formula", 0.614),
    ("gt", "text+formula", 0.644),
    ("gt", "text+image+formula", 0.607),
]

# Control configs (should not change much)
CONTROL_CONFIGS = [
    ("gcn", "text+image+formula", 0.730),
    ("gat", "text+image+formula", 0.651),
    ("graphsage", "text+image+formula", 0.675),
]

OUTPUT_DIR = "./benchmark/results/lp_ablation_fixed_v3a"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save old results for comparison
    old_results = {}
    old_summary_path = "./benchmark/results/lp_ablation_noproj/lp_7modality_summary.json"
    if os.path.exists(old_summary_path):
        with open(old_summary_path) as f:
            old_data = json.load(f)
        for key, val in old_data.items():
            old_results[key] = val

    all_configs = PROBLEMATIC_CONFIGS + CONTROL_CONFIGS

    # Group by modality to avoid reloading dataset
    from collections import defaultdict
    modality_groups = defaultdict(list)
    for model, modality, old_mrr in all_configs:
        modality_groups[modality].append((model, old_mrr))

    results_table = {}
    domains = ["cs_si", "cs_lo", "math_lo", "physics_socph"]

    for modality, configs in modality_groups.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Loading modality: {modality}")
        logger.info(f"{'='*60}")

        dataset = AMMAGDataset(
            root="./", domains=domains, modality=modality, label_type="domain"
        )
        raw_dim = dataset.data.x.size(1)
        logger.info(f"  Feature dim: {raw_dim}")

        for model_name, old_mrr in configs:
            logger.info(f"\n--- {model_name} + {modality} (raw_dim={raw_dim}) ---")
            logger.info(f"  Old MRR: {old_mrr:.4f}")

            summary = run_link_prediction(
                dataset=dataset,
                model_name=model_name,
                modality=modality,
                num_runs=3,
                num_epochs=500,
                hidden_dim=256,
                num_layers=2,
                dropout=0.5,
                lr=0.01,
                device="cuda",
                output_dir=OUTPUT_DIR,
                proj_dim=None,
                neg_ratio=10,
            )

            new_mrr = summary["mrr_mean"]
            delta = new_mrr - old_mrr
            arrow = "↑" if delta > 0 else "↓"
            logger.info(f"  New MRR: {new_mrr:.4f} ({arrow}{abs(delta):.4f})")
            logger.info(f"  Std: {summary['mrr_std']:.4f}")

            key = f"{model_name}_{modality}"
            results_table[key] = {
                "model": model_name,
                "modality": modality,
                "raw_dim": raw_dim,
                "mrr_mean": new_mrr,
                "mrr_std": summary["mrr_std"],
                "hits@1_mean": summary["hits@1_mean"],
                "hits@3_mean": summary["hits@3_mean"],
                "hits@10_mean": summary["hits@10_mean"],
                "old_mrr": old_mrr,
                "delta": delta,
            }

    # Save results
    summary_path = os.path.join(OUTPUT_DIR, "lp_fix_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results_table, f, indent=2, default=str)

    # Print comparison table
    logger.info(f"\n{'='*80}")
    logger.info("LP Fix Results: Before vs After")
    logger.info(f"{'='*80}")
    logger.info(f"{'Config':<35} {'Old MRR':>10} {'New MRR':>10} {'Delta':>10} {'New Std':>10}")
    logger.info("-" * 80)

    for key in sorted(results_table.keys()):
        r = results_table[key]
        delta_str = f"{r['delta']:+.4f}"
        logger.info(f"{key:<35} {r['old_mrr']:>10.4f} {r['mrr_mean']:>10.4f} {delta_str:>10} {r['mrr_std']:>10.4f}")

    logger.info(f"\nResults saved to {summary_path}")
    return results_table


if __name__ == "__main__":
    main()
