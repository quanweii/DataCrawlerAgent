#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Partition-robustness NC suite (split_seed=2) + validation-selection logging.

Purpose (2026-09-19, review items #4/#5):
  1. Rerun the production protocol-matrix grid under a SECOND partition seed,
     so the fixed-GraphSAGE trans/split drop can be reported with
     initialization AND partition variation instead of init-only.
  2. Record per-model best-val accuracy alongside test accuracy, enabling the
     validation-selection vs test-argmax agreement statistic that the retained
     artifacts (test-only) cannot provide.

Safety:
  - Does NOT call save_protocol_suite: production protocol_views/ manifests
    under benchmark/results/protocol_views/ are never touched.
  - Writes only to benchmark/results/newdomain_nc_split2/.
  - Training configuration mirrors run_newdomain_nc.run_protocol_matrix
    defaults exactly (epochs 300, hidden 256, layers 2, dropout 0.5,
    model_lr per model, proj_dim 768, init seeds 0..4).
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", type=int, default=None)
_pre.add_argument("--domains", default="steam,tmdb,wikiart,arxiv")
_pre.add_argument("--split-seed", type=int, default=2)
_pre.add_argument("--init-seeds", type=int, default=5)
_args, _ = _pre.parse_known_args()

if _args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_args.gpu)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.data.domain_loader import SingleDomainDataset  # noqa: E402
from benchmark.data.splits import label_propagation_ceiling  # noqa: E402
from benchmark.data.protocols import (PROTOCOLS, generate_protocol_views,  # noqa: E402
                                      protocol_excluded_relations)
from benchmark.models.gnn import get_model, model_lr  # noqa: E402
from benchmark.tasks.node_classification import train_and_eval  # noqa: E402

OUT_DIR = PROJECT_ROOT / "benchmark" / "results" / "newdomain_nc_split2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FULL_FUSION = {
    "arxiv": "text+image+formula",
    "steam": "text+image",
    "tmdb": "text+image",
    "wikiart": "text+image",
}
MODELS = ["mlp", "gcn", "gat", "graphsage", "gin", "gcnii"]
HP = dict(epochs=300, hidden=256, layers=2, dropout=0.5, proj_dim=768)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_domain(domain, device):
    modality = FULL_FUSION[domain]
    excluded = protocol_excluded_relations(domain)
    ds = SingleDomainDataset(domain, modality=modality, exclude_relations=excluded)
    suite = generate_protocol_views(
        domain, ds.graph["node_ids"], ds.data.y, ds.data.edge_index,
        seed=_args.split_seed, excluded_relations=excluded)

    combined = {
        "domain": domain,
        "protocols": list(PROTOCOLS),
        "modalities": [modality],
        "models": MODELS,
        "split_seed": _args.split_seed,
        "init_seeds": _args.init_seeds,
        "representation_policy": (
            "frozen per-node encoder outputs reused across protocol views; "
            "label-derived fields removed before encoding; mean fusion has no fitted state"),
        "purpose": "partition robustness (second split seed) + validation-selection logging",
        "results": {},
        "protocol_manifest": copy.deepcopy(suite["manifest"]),
    }

    for protocol in PROTOCOLS:
        view = suite["views"][protocol]
        tr, va, te = view["masks"]
        edge_index = view["edge_index"]
        data = Data(x=ds.data.x, edge_index=edge_index, y=ds.data.y)
        data.num_nodes = ds.data.num_nodes
        lp = label_propagation_ceiling(ds.data.y, edge_index, tr, te)
        log(f"[{domain}/{protocol}] sizes={int(tr.sum())}/{int(va.sum())}/{int(te.sum())} "
            f"edges={edge_index.size(1)}")
        cells = {}
        for model in MODELS:
            accs, f1s, vals, times = [], [], [], []
            t_cell = time.time()
            for seed in range(_args.init_seeds):
                torch.manual_seed(seed)
                m = get_model(model, data.x.size(1), HP["hidden"],
                              int(data.y.max().item()) + 1,
                              num_layers=HP["layers"], dropout=HP["dropout"],
                              proj_dim=HP["proj_dim"])
                r = train_and_eval(m, data, tr, va, te,
                                   num_epochs=HP["epochs"], lr=model_lr(model),
                                   device=device)
                accs.append(float(r["test_acc"]))
                f1s.append(float(r["test_f1"]))
                vals.append(float(r.get("val_acc", r.get("best_val_acc", float("nan")))))
                times.append(float(r.get("training_time", float("nan"))))
            cells[model] = {
                "acc": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                "f1": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
                "val_acc": float(np.mean(vals)), "val_acc_std": float(np.std(vals)),
                "train_time_s_mean": float(np.nanmean(times)),
                "accs": accs, "f1s": f1s, "val_accs": vals,
            }
            log(f"  {model}: test={np.mean(accs):.4f}±{np.std(accs):.4f} "
                f"val={np.mean(vals):.4f} ({time.time()-t_cell:.0f}s)")
        combined["results"][f"{protocol}/{modality}"] = {
            "protocol": protocol, "modality": modality,
            "split_sizes": [int(tr.sum()), int(va.sum()), int(te.sum())],
            "directed_edges": int(edge_index.size(1)),
            "topology_only_predictability": lp,
            "interpretation": "split_seed_2_partition_robustness",
            "models": cells,
        }

    out_path = OUT_DIR / f"nc_{domain}_protocol_matrix_splitseed{_args.split_seed}.json"
    out_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    log(f"saved {out_path}")
    return combined


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device} split_seed={_args.split_seed} domains={_args.domains}")
    for domain in [d.strip() for d in _args.domains.split(",") if d.strip()]:
        t0 = time.time()
        run_domain(domain, device)
        log(f"[{domain}] done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
