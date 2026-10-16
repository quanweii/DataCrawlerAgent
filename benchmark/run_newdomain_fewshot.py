#!/usr/bin/env python3
"""Leak-free few-shot node classification (k=3/10 per class) for new-domain MAGs.

Protocol (consistent with run_newdomain_nc leak-free headline):
  - Fixed stratified 60/20/20 split (seed=1); leak-free cross-split edge pruning
    (same pruned graph as the headline NC table).
  - Few-shot: sample k nodes per class FROM the 60% train portion as labels;
    val/test masks are the fixed stratified ones.
  - Transductive: full (pruned) graph visible; only labels are few-shot.
  - R label-samples (different k-shot draws) → mean±std test acc/f1.

Usage:
  python benchmark/run_newdomain_fewshot.py --domain steam --gpu 1 \\
      --modalities text+image --models mlp,gcn,gat,graphsage,gin,gcnii --ks 3,10 --runs 10
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
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.data.domain_loader import SingleDomainDataset
from benchmark.data.splits import stratified_split, prune_cross_split_edges
from benchmark.models.gnn import get_model, model_lr

_args = None


def _fewshot_masks(y, train_idx, num_classes, k, seed):
    """Sample k nodes per class from train_idx → few-shot train mask."""
    rng = np.random.default_rng(seed)
    y_np = y.numpy() if torch.is_tensor(y) else np.asarray(y)
    fs_idx = []
    for c in range(num_classes):
        cand = train_idx[y_np[train_idx] == c]
        if len(cand) == 0:
            continue
        take = min(k, len(cand))
        fs_idx.extend(rng.choice(cand, take, replace=False))
    fs_idx = np.array(fs_idx)
    n = len(y_np)
    tr = torch.zeros(n, dtype=torch.bool); tr[fs_idx] = True
    return tr


def _train_eval(model, data, tr, va, te, num_epochs, lr, device):
    model = model.to(device); data = data.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    crit = nn.CrossEntropyLoss()
    best_val, best_state, pat, ctr = 0.0, None, 80, 0
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
                best_val = va_acc
                best_state = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
                ctr = 0
            else:
                ctr += 1
                if ctr >= pat:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)
        te_acc = float((pred[te] == data.y[te]).float().mean())
        te_f1 = float(f1_score(data.y[te].cpu(), pred[te].cpu(), average="macro"))
    return te_acc, te_f1


def run_fewshot(domain, modality, models, device, out_dir):
    ds = SingleDomainDataset(domain, modality=modality)
    y = ds.data.y
    ei_full = ds.data.edge_index
    n = ds.data.num_nodes
    num_classes = int(y.max().item()) + 1
    masks = stratified_split(y, ratios=(0.6, 0.2, 0.2), seed=_args.split_seed)
    tr_full, va, te = masks
    ei_pruned = prune_cross_split_edges(ei_full, masks)
    data = Data(x=ds.data.x, edge_index=ei_pruned, y=y); data.num_nodes = n
    in_dim = data.x.size(1)
    train_idx = np.where(tr_full.numpy())[0]

    table = {}
    for model in models:
        for k in _ks:
            accs, f1s = [], []
            for run in range(_args.runs):
                fs_tr = _fewshot_masks(y, train_idx, num_classes, k, seed=1000 * k + run)
                torch.manual_seed(run)
                m = get_model(model, in_dim, _args.hidden, num_classes,
                              num_layers=_args.layers, dropout=_args.dropout,
                              proj_dim=_args.proj_dim if _args.proj_dim else None)
                acc, f1 = _train_eval(m, data, fs_tr, va, te, _args.epochs, model_lr(model), device)
                accs.append(acc); f1s.append(f1)
            key = f"{model}+{modality}+k{k}"
            table[key] = {"k": k, "model": model, "modality": modality,
                          "acc": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                          "f1": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
                          "n_runs": len(accs)}
            logger.info(f"  {model}+k{k}: acc={np.mean(accs):.3f}±{np.std(accs):.3f} f1={np.mean(f1s):.3f}")
    summary = {"domain": domain, "modality": modality, "protocol": "leak_free_fewshot",
               "split_seed": _args.split_seed, "runs": _args.runs, "ks": _ks,
               "num_classes": num_classes, "num_nodes": n, "table": table}
    sp = out_dir / f"fewshot_{domain}_summary.json"
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"saved {sp}")
    return summary


def main():
    global _args, _ks
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--domain", required=True, choices=["steam", "tmdb", "wikiart", "arxiv", "spotify"])
    _pre.add_argument("--gpu", type=int, default=None)
    _pre.add_argument("--modalities", default="text+image")
    _pre.add_argument("--models", default="mlp,gcn,gat,graphsage,gin,gcnii")
    _pre.add_argument("--ks", default="3,10")
    _pre.add_argument("--epochs", type=int, default=300)
    _pre.add_argument("--hidden", type=int, default=256)
    _pre.add_argument("--layers", type=int, default=2)
    _pre.add_argument("--dropout", type=float, default=0.5)
    _pre.add_argument("--lr", type=float, default=0.01)
    _pre.add_argument("--runs", type=int, default=10)
    _pre.add_argument("--split-seed", type=int, default=1)
    _pre.add_argument("--proj-dim", type=int, default=768, help="learnable Linear 投影;0=不投影")
    _args = _pre.parse_args()
    _ks = [int(x) for x in _args.ks.split(",") if x.strip()]
    if _args.gpu is not None and torch.cuda.is_available():
        device = f"cuda:{_args.gpu}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = PROJECT_ROOT / "benchmark" / "results" / "newdomain_fewshot"
    out_dir.mkdir(parents=True, exist_ok=True)
    domain = _args.domain
    modalities = [m.strip() for m in _args.modalities.split(",") if m.strip()]
    models = [m.strip() for m in _args.models.split(",") if m.strip()]
    merged_table = {}
    for modality in modalities:
        last = run_fewshot(domain, modality, models, device, out_dir)
        merged_table.update(last["table"])
    summary = {"domain": domain, "protocol": "leak_free_fewshot", "table": merged_table}
    sp = out_dir / f"fewshot_{domain}_summary.json"
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"final merged → {sp}")


if __name__ == "__main__":
    main()
