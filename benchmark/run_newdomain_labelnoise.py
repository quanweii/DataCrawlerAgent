#!/usr/bin/env python3
"""Label-noise robustness for new-domain MAGs (P2-2, OpenMAG Q7 label-noise arm).

Flip a rho fraction of TRAIN labels (val/test labels stay clean), train
GCN/GAT/GraphSAGE on the leak-free pruned graph, report test-acc degradation vs
the clean (rho=0) anchor. Tests label-quality robustness.

Usage:
  python benchmark/run_newdomain_labelnoise.py --domain tmdb --gpu 0
"""
import os
import sys
import argparse
import json
import copy
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


def _train_eval(model, data, tr, va, te, num_epochs, lr, device):
    model = model.to(device); data = data.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    crit = nn.CrossEntropyLoss()
    best_val, best_state, pat, ctr = 0.0, None, 50, 0
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


def run_domain(domain, device, out_dir):
    ds = SingleDomainDataset(domain, modality="text+image")
    y = ds.data.y; ei = ds.data.edge_index; n = ds.data.num_nodes
    C = int(y.max().item()) + 1
    masks = stratified_split(y, ratios=(0.6, 0.2, 0.2), seed=1)
    tr, va, te = masks
    ei_p = prune_cross_split_edges(ei, masks)
    in_dim = ds.data.x.size(1)
    table = {}
    for model in ["gcn", "gat", "graphsage"]:
        for rho in [0.0, 0.2, 0.4, 0.6]:
            accs = []
            for run in range(10):
                seed = run * 42 + 1
                rng = np.random.RandomState(seed * 7 + 13)
                y_noisy = y.clone()
                if rho > 0:
                    tr_idx = np.where(tr.numpy())[0]
                    flip_n = int(len(tr_idx) * rho)
                    flip_sel = rng.choice(tr_idx, flip_n, replace=False)
                    new_labels = rng.randint(0, C, size=flip_n)
                    # avoid "flipping" to the same label
                    for j, idx in enumerate(flip_sel):
                        if y_noisy[idx] == new_labels[j]:
                            new_labels[j] = (new_labels[j] + 1) % C
                        y_noisy[idx] = new_labels[j]
                data = Data(x=ds.data.x, edge_index=ei_p, y=y_noisy); data.num_nodes = n
                torch.manual_seed(seed)
                m = get_model(model, in_dim, 256, C, num_layers=2, dropout=0.5,
                              proj_dim=_args.proj_dim if _args.proj_dim else None)
                acc, _ = _train_eval(m, data, tr, va, te, _args.epochs, model_lr(model), device)
                accs.append(acc)
            table[f"{model}+rho{rho}"] = {"rho": rho, "model": model,
                                          "acc": float(np.mean(accs)), "acc_std": float(np.std(accs))}
            logger.info(f"[{domain}] {model} rho={rho}: acc={np.mean(accs):.3f}±{np.std(accs):.3f}")
    res = {"domain": domain, "protocol": "leak_free_labelnoise", "num_classes": C,
           "num_nodes": n, "table": table}
    sp = out_dir / f"labelnoise_{domain}_summary.json"
    with open(sp, "w") as f:
        json.dump(res, f, indent=2)
    logger.info(f"saved {sp}")
    return res


def main():
    global _args
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--domain", required=True, choices=["steam", "tmdb", "wikiart"])
    _pre.add_argument("--gpu", type=int, default=None)
    _pre.add_argument("--epochs", type=int, default=300)
    _pre.add_argument("--proj-dim", type=int, default=768, help="learnable Linear 投影;0=不投影")
    _args = _pre.parse_args()
    if _args.gpu is not None and torch.cuda.is_available():
        device = f"cuda:{_args.gpu}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = PROJECT_ROOT / "benchmark" / "results" / "newdomain_labelnoise"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_domain(_args.domain, device, out_dir)


if __name__ == "__main__":
    main()
