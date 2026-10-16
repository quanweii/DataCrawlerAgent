"""
Q7 ROBUSTNESS ablation for the 3 new domains (steam / tmdb / wikiart).

benchmark/tasks/noise_experiments.py only supports arxiv; this script extends
the feature / edge noise study to the new-domain benchmark. We keep the model
fixed at GCN + text+image (the apples-to-apples structural baseline) and sweep
two noise types:

  (a) FEATURE noise: replace `rho` fraction of node feature ROWS with Gaussian
      random vectors (mean/std estimated from the clean feature matrix). This
      destroys the semantic content of those nodes while keeping their graph
      neighbourhood intact.
  (b) EDGE noise:    add `rho * |E|` random (undirected) edges, diluting the
      graph's structural signal with spurious connections.

rho in {0, 0.2, 0.4, 0.6, 0.8}, runs=3, seeds = run*42+1 (identical to
run_newdomain_nc.py so rho=0 reproduces the published GCN+text+image NC acc).

Findings we want:
  - which noise type hurts more (per domain)?
  - does sensitivity track homophily? (wikiart 0.894 homophilic vs tmdb 0.237
    heterophilic vs steam 0.526 middle.)

用法:
  python benchmark/run_newdomain_robustness.py --domain steam --gpu 5
  python benchmark/run_newdomain_robustness.py --domain tmdb  --gpu 5
  python benchmark/run_newdomain_robustness.py --domain wikiart --gpu 5

Output: benchmark/results/newdomain_robustness/
        robust_{domain}_feature.json
        robust_{domain}_edge.json
        robust_{domain}_summary.json   (compact: per-rho acc/f1, deltas vs rho=0)
"""
# ---- CLI parsed BEFORE import torch (control CUDA visibility) ----
import os
import argparse

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--domain", required=True, choices=["steam", "tmdb", "wikiart"])
_pre.add_argument("--gpu", type=int, default=None)
_pre.add_argument("--noise_types", default="feature,edge",
                  help="comma list: feature,edge")
_pre.add_argument("--rhos", default="0,0.2,0.4,0.6,0.8",
                  help="comma list of noise ratios")
_pre.add_argument("--runs", type=int, default=10)
_pre.add_argument("--epochs", type=int, default=300)
_pre.add_argument("--hidden", type=int, default=256)
_pre.add_argument("--layers", type=int, default=2)
_pre.add_argument("--dropout", type=float, default=0.5)
_pre.add_argument("--lr", type=float, default=0.01)
_pre.add_argument("--weight_decay", type=float, default=5e-4)
_pre.add_argument("--patience", type=int, default=50)
_pre.add_argument("--model", default="gcn")
_pre.add_argument("--modality", default="text+image")
_pre.add_argument("--proj-dim", type=int, default=768, help="learnable Linear 投影;0=不投影")
_args, _ = _pre.parse_known_args()
if _args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_args.gpu)

import sys
import json
import copy
from pathlib import Path

import numpy as np
import torch
from loguru import logger

logger.add(sys.stderr, level="INFO")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.data.domain_loader import SingleDomainDataset
from benchmark.models.gnn import get_model
from benchmark.tasks.node_classification import train_and_eval


# ============================ noise injectors ============================
# Modeled on benchmark/tasks/noise_experiments.py but adapted for the
# single-domain pipeline:
#   - feature noise: replace rho*N whole rows with Gaussian vectors drawn from
#     the column-wise mean/std of the clean matrix (matches "gaussian" mode).
#   - edge noise: add rho*|E| random UNDIRECTED edges (both directions added,
#     matching SingleDomainDataset's symmetric edge_index convention).

def inject_feature_noise_rows(features: torch.Tensor, rho: float,
                              rng: np.random.RandomState) -> torch.Tensor:
    """Replace rho fraction of node feature ROWS with Gaussian random vectors.

    Gaussian params (mean, std) are estimated column-wise from the CLEAN matrix
    so the perturbed rows stay on the same marginal scale — this isolates the
    effect of losing per-node semantic alignment rather than rescaling features.
    """
    if rho <= 0.0:
        return features
    n = features.size(0)
    n_noisy = int(round(n * rho))
    if n_noisy <= 0:
        return features
    idx = rng.choice(n, n_noisy, replace=False)
    mean = features.mean(dim=0)
    std = features.std(dim=0) + 1e-8
    out = features.clone()
    noise = torch.randn(n_noisy, features.size(1)) * std + mean
    out[idx] = noise
    return out


def inject_edge_noise(edge_index: torch.Tensor, num_nodes: int, rho: float,
                      rng: np.random.RandomState) -> torch.Tensor:
    """Add rho*|E| random undirected edges (symmetrized).

    Avoids self-loops and exact-duplicate edges (cheap dedup via a set over the
    existing edges; collisions just reduce the added count slightly, which is
    negligible relative to |E|)."""
    if rho <= 0.0:
        return edge_index
    n_edges = edge_index.size(1)
    n_new_pairs = int(round(n_edges * rho))  # each pair counts once
    if n_new_pairs <= 0:
        return edge_index

    existing = set()
    for i in range(n_edges):
        s = edge_index[0, i].item()
        d = edge_index[1, i].item()
        # store canonical (min,max) so reverse dedups too
        existing.add((min(s, d), max(s, d)))

    new_pairs = []
    attempts = 0
    max_attempts = n_new_pairs * 30 + 1000
    while len(new_pairs) < n_new_pairs and attempts < max_attempts:
        attempts += 1
        s = int(rng.randint(0, num_nodes))
        d = int(rng.randint(0, num_nodes))
        if s == d:
            continue
        key = (min(s, d), max(s, d))
        if key in existing:
            continue
        existing.add(key)
        new_pairs.append((s, d))

    if not new_pairs:
        return edge_index

    src = [s for s, d in new_pairs] + [d for s, d in new_pairs]
    dst = [d for s, d in new_pairs] + [s for s, d in new_pairs]
    add = torch.tensor([src, dst], dtype=edge_index.dtype)
    return torch.cat([edge_index, add], dim=1)


# ============================ one noise-sweep cell ============================

def run_cell(dataset, noise_type: str, rho: float, num_runs: int,
             num_epochs: int, hidden_dim: int, num_layers: int,
             dropout: float, lr: float, weight_decay: float, patience: int,
             device: str):
    """Train GCN+text+image under one (noise_type, rho) setting, `num_runs`
    seeds. Seeds match run_newdomain_nc.py (run*42+1) so rho=0 reproduces the
    published anchor.

    Noise is injected FRESH per run from the same seed that controls masks and
    model init — this keeps each run comparable and yields a fair std.
    """
    clean_data = dataset.data
    num_nodes = clean_data.num_nodes
    in_dim = clean_data.x.size(1)
    out_dim = int(clean_data.y.max().item()) + 1

    # base RNG for noise (decoupled from torch model init RNG, but seeded
    # identically per run for reproducibility)
    accs, f1s = [], []
    for run in range(num_runs):
        seed = run * 42 + 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng_noise = np.random.RandomState(seed * 7 + 13)

        # build a noisy COPY of the data object (do NOT mutate dataset.data)
        data = copy.copy(clean_data)  # shallow; we replace x/edge_index below
        if noise_type == "feature":
            x_noisy = inject_feature_noise_rows(clean_data.x, rho, rng_noise)
            data.x = x_noisy
            data.edge_index = clean_data.edge_index
        elif noise_type == "edge":
            ei_noisy = inject_edge_noise(
                clean_data.edge_index, num_nodes, rho, rng_noise)
            data.x = clean_data.x
            data.edge_index = ei_noisy
        else:
            raise ValueError(f"unknown noise_type {noise_type}")

        # keep y / num_nodes / masks intact
        data.y = clean_data.y
        data.num_nodes = num_nodes

        train_mask, val_mask, test_mask = dataset.get_split_masks(
            split="supervised", seed=seed)

        model = get_model(_args.model, in_dim, hidden_dim, out_dim,
                          num_layers=num_layers, dropout=dropout,
                          proj_dim=_args.proj_dim if _args.proj_dim else None)

        # train_and_eval moves model+data to device; re-seed torch right before
        # model weight init already happened above, so deterministic per run.
        res = train_and_eval(
            model, data, train_mask, val_mask, test_mask,
            num_epochs=num_epochs, lr=lr, weight_decay=weight_decay,
            patience=patience, device=device,
        )
        accs.append(res["test_acc"])
        f1s.append(res["test_f1"])
        logger.info(f"  [{noise_type} rho={rho:.1f} run {run+1}/{num_runs}] "
                    f"acc={res['test_acc']:.4f} f1={res['test_f1']:.4f}")

    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "run_accs": [float(a) for a in accs],
        "run_f1s": [float(f) for f in f1s],
    }


# ============================ main ============================

def main():
    domain = _args.domain
    noise_types = [t.strip() for t in _args.noise_types.split(",") if t.strip()]
    rhos = [float(r.strip()) for r in _args.rhos.split(",") if r.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = PROJECT_ROOT / "benchmark" / "results" / "newdomain_robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'=' * 70}\n[{domain}] ROBUSTNESS ablation\n"
                f"  model={_args.model} modality={_args.modality}\n"
                f"  noise_types={noise_types} rhos={rhos} runs={_args.runs}\n"
                f"  device={device}\n{'=' * 70}")

    ds = SingleDomainDataset(domain, modality=_args.modality)
    homophily = ds.graph["homophily"]
    num_classes = int(ds.data.y.max().item()) + 1
    logger.info(f"[{domain}] homophily={homophily:.3f} classes={num_classes} "
                f"nodes={ds.data.num_nodes} edges={ds.data.edge_index.size(1)}")

    summary = {
        "domain": domain,
        "model": _args.model,
        "modality": _args.modality,
        "homophily": float(homophily),
        "num_classes": int(num_classes),
        "num_nodes": int(ds.data.num_nodes),
        "num_edges": int(ds.data.edge_index.size(1)),
        "rhos": rhos,
        "noise_types": {},
    }

    for nt in noise_types:
        logger.info(f"\n----- {domain} noise_type={nt} -----")
        per_rho = {}
        for rho in rhos:
            logger.info(f"--- {nt} rho={rho:.1f} ---")
            cell = run_cell(
                ds, nt, rho, num_runs=_args.runs, num_epochs=_args.epochs,
                hidden_dim=_args.hidden, num_layers=_args.layers,
                dropout=_args.dropout, lr=_args.lr,
                weight_decay=_args.weight_decay, patience=_args.patience,
                device=device,
            )
            per_rho[f"rho_{rho:.1f}"] = cell

        # save per noise-type raw file
        raw = {
            "domain": domain, "model": _args.model, "modality": _args.modality,
            "noise_type": nt, "homophily": float(homophily),
            "rhos": rhos, "results": per_rho,
        }
        raw_path = out_dir / f"robust_{domain}_{nt}.json"
        with open(raw_path, "w") as f:
            json.dump(raw, f, indent=2)
        logger.info(f"saved {raw_path}")
        summary["noise_types"][nt] = _compact(per_rho, rhos)

    # compact summary across noise types
    sp = out_dir / f"robust_{domain}_summary.json"
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"saved summary {sp}")
    _print_table(domain, summary, rhos, noise_types, homophily)


def _compact(per_rho, rhos):
    """acc/f1 at each rho + delta vs rho=0."""
    base = per_rho.get(f"rho_{rhos[0]:.1f}", {}).get("acc_mean", float("nan"))
    rows = []
    for rho in rhos:
        c = per_rho[f"rho_{rho:.1f}"]
        rows.append({
            "rho": rho,
            "acc": c["acc_mean"], "acc_std": c["acc_std"],
            "f1": c["f1_mean"], "f1_std": c["f1_std"],
            "delta_acc_vs_rho0": c["acc_mean"] - base,
        })
    return {"base_acc_rho0": base, "curve": rows}


def _print_table(domain, summary, rhos, noise_types, homophily):
    logger.info(f"\n{'=' * 70}\n[{domain}] ROBUSTNESS  homophily={homophily:.3f}\n"
                f"{'=' * 70}")
    header = f"{'noise':<8}{'rho':<7}{'acc':<18}{'f1':<18}{'Δacc vs rho0':<14}"
    logger.info(header)
    for nt in noise_types:
        comp = summary["noise_types"][nt]
        base = comp["base_acc_rho0"]
        for row in comp["curve"]:
            logger.info(
                f"{nt:<8}{row['rho']:<7.1f}"
                f"{row['acc']:.4f}±{row['acc_std']:.4f}    "
                f"{row['f1']:.4f}±{row['f1_std']:.4f}    "
                f"{row['delta_acc_vs_rho0']:+.4f}")


if __name__ == "__main__":
    main()
