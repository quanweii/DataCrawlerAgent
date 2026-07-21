#!/usr/bin/env python3
"""Leak-free node clustering (NMI/ARI) for the new-domain MAGs — OpenMAG Q5 cluster arm.

Mirrors run_newdomain_nc.py's leak-free protocol (fixed stratified split +
cross-split edge pruning + per-init-seed) but, after supervised training,
extracts the GNN output logits and runs KMeans(k=num_classes) → NMI/ARI vs the
true labels. Also reports a raw-features (no-GNN) KMeans baseline.

Usage:
  python benchmark/run_newdomain_clustering.py --domain wikiart --gpu 6 \\
      --modalities text+image --models mlp,gcn,gat,graphsage,gin,gcnii --init-seeds 5
"""
import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.data.domain_loader import SingleDomainDataset
from benchmark.data.splits import stratified_split, prune_cross_split_edges
from benchmark.models.gnn import get_model, model_lr

_args = None
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _train(model, data, tr, va, te, num_epochs, lr, device):
    """Mirror tasks/node_classification.train_and_eval; return best-state model."""
    model = model.to(device); data = data.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    crit = nn.CrossEntropyLoss()
    best_val, best_state, patience, pat_ctr = 0.0, None, 50, 0
    for epoch in range(1, num_epochs + 1):
        model.train(); opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = crit(out[tr], data.y[tr]); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if epoch % 5 == 0 or epoch == num_epochs:
            model.eval()
            with torch.no_grad():
                pred = model(data.x, data.edge_index).argmax(dim=1)
                va_acc = float((pred[va] == data.y[va]).float().mean())
            if va_acc > best_val:
                best_val, best_state, pat_ctr = va_acc, {k: v.detach().cpu().clone()
                                                          for k, v in model.state_dict().items()}, 0
            else:
                pat_ctr += 1
                if pat_ctr >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def _kmeans_metrics(model, data, num_classes, device, raw=False):
    model.eval(); model = model.to(device)
    if raw:
        feats = data.x.detach().cpu().numpy()
    else:
        feats = model(data.x.to(device), data.edge_index.to(device)).cpu().numpy()
    y = data.y.detach().cpu().numpy()
    km = KMeans(n_clusters=num_classes, random_state=42, n_init=10)
    pred = km.fit_predict(feats)
    return (normalized_mutual_info_score(y, pred), adjusted_rand_score(y, pred))


def run_clustering(domain, modality, models, device, out_dir):
    ds = SingleDomainDataset(domain, modality=modality)
    y = ds.data.y
    ei_full = ds.data.edge_index
    n = ds.data.num_nodes
    masks = stratified_split(y, ratios=(0.6, 0.2, 0.2), seed=_args.split_seed)
    tr, va, te = masks
    ei_pruned = prune_cross_split_edges(ei_full, masks)
    orig, kept = ei_full.size(1), ei_pruned.size(1)
    data = Data(x=ds.data.x, edge_index=ei_pruned, y=y); data.num_nodes = n
    in_dim, out_dim = data.x.size(1), int(y.max().item()) + 1

    table = {}
    # raw-features baseline (no GNN) — shared across models, compute once
    dummy = get_model("mlp", in_dim, _args.hidden, out_dim, num_layers=_args.layers,
                      dropout=_args.dropout,
                      proj_dim=_args.proj_dim if _args.proj_dim else None).to(device)
    raw_nmis, raw_aris = [], []
    for seed in range(_args.init_seeds):
        nmi, ari = _kmeans_metrics(dummy, Data(x=ds.data.x, edge_index=ei_pruned, y=y),
                                   out_dim, device, raw=True)
        raw_nmis.append(nmi); raw_aris.append(ari)
    table["raw_features"] = {"nmi": float(np.mean(raw_nmis)), "nmi_std": float(np.std(raw_nmis)),
                             "ari": float(np.mean(raw_aris)), "ari_std": float(np.std(raw_aris))}
    logger.info(f"[{domain}/{modality}] raw-features KMeans: NMI={np.mean(raw_nmis):.3f} "
                f"ARI={np.mean(raw_aris):.3f}")

    for model in models:
        nmis, aris = [], []
        for seed in range(_args.init_seeds):
            torch.manual_seed(seed)
            m = get_model(model, in_dim, _args.hidden, out_dim,
                          num_layers=_args.layers, dropout=_args.dropout,
                          proj_dim=_args.proj_dim if _args.proj_dim else None)
            m = _train(m, data, tr, va, te, _args.epochs, model_lr(model), device)
            nmi, ari = _kmeans_metrics(m, data, out_dim, device, raw=False)
            nmis.append(nmi); aris.append(ari)
        key = f"{model}+{modality}"
        table[key] = {"nmi": float(np.mean(nmis)), "nmi_std": float(np.std(nmis)),
                      "ari": float(np.mean(aris)), "ari_std": float(np.std(aris))}
        logger.info(f"  {model}: NMI={np.mean(nmis):.3f}±{np.std(nmis):.3f} "
                    f"ARI={np.mean(aris):.3f}±{np.std(aris):.3f}")

    summary = {"domain": domain, "modality": modality, "protocol": "leak_free_pruned_clustering",
               "split_seed": _args.split_seed, "init_seeds": _args.init_seeds,
               "edges_original": orig, "edges_pruned": kept,
               "num_classes": out_dim, "num_nodes": n, "table": table}
    sp = out_dir / f"cluster_{domain}_summary.json"
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"saved {sp}")
    return summary


def main():
    global _args
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--domain", required=True, choices=["steam", "tmdb", "wikiart", "arxiv", "spotify"])
    _pre.add_argument("--gpu", type=int, default=None)
    _pre.add_argument("--modalities", default="text+image")
    _pre.add_argument("--models", default="mlp,gcn,gat,graphsage,gin,gcnii")
    _pre.add_argument("--epochs", type=int, default=300)
    _pre.add_argument("--hidden", type=int, default=256)
    _pre.add_argument("--layers", type=int, default=2)
    _pre.add_argument("--dropout", type=float, default=0.5)
    _pre.add_argument("--lr", type=float, default=0.01)
    _pre.add_argument("--init-seeds", type=int, default=10)
    _pre.add_argument("--split-seed", type=int, default=1)
    _pre.add_argument("--proj-dim", type=int, default=768, help="learnable Linear 投影;0=不投影")
    _args = _pre.parse_args()
    if _args.gpu is not None and torch.cuda.is_available():
        device = f"cuda:{_args.gpu}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = PROJECT_ROOT / "benchmark" / "results" / "newdomain_clustering"
    out_dir.mkdir(parents=True, exist_ok=True)
    domain = _args.domain
    modalities = [m.strip() for m in _args.modalities.split(",") if m.strip()]
    models = [m.strip() for m in _args.models.split(",") if m.strip()]
    merged_table = {}
    meta = {}
    for modality in modalities:
        last = run_clustering(domain, modality, models, device, out_dir)
        merged_table.update(last["table"])
        meta = {k: last[k] for k in ("protocol", "split_seed", "init_seeds",
                "edges_original", "edges_pruned", "num_classes", "num_nodes") if k in last}
        meta["modalities"] = modalities
    summary = {"domain": domain, **meta, "table": merged_table}
    sp = out_dir / f"cluster_{domain}_summary.json"
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"final merged → {sp}")


if __name__ == "__main__":
    main()
