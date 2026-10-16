#!/usr/bin/env python3
"""Modality-missing robustness (OpenMAG Q7b) for new-domain MAGs.

For a multimodal feature, zero / 50%-dropout one modality's embedding block and
measure NC accuracy degradation on the leak-free pruned graph. Reports a
per-domain degradation matrix across mask conditions × models.

Conditions (on the concatenated CLIP feature):
  full            — baseline (no masking)
  drop_text       — zero the entire text block
  drop_image      — zero the entire image block
  dropout_text_50 — zero a random 50% of the text block dims
  dropout_image_50— zero a random 50% of the image block dims

Usage:
  python benchmark/run_newdomain_modality_missing.py --domain steam --gpu 2 \\
      --models gcn,gat,graphsage --init-seeds 5
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

from benchmark.data.domain_loader import SingleDomainDataset, build_features, load_domain_graph
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


def run_modality_missing(domain, models, device, out_dir):
    graph = load_domain_graph(domain)
    # 各模态 L2 归一化嵌入（mean 融合在这些之上做；OpenMAG 逐元素融合，非拼接）
    x_text = build_features(domain, graph, "text")     # 768, 已归一化
    x_img = build_features(domain, graph, "image")     # 768, 已归一化
    mods = [x_text, x_img]
    dim = x_text.size(1)                               # 768

    ds = SingleDomainDataset(domain, modality="text+image")
    y = ds.data.y
    ei_full = ds.data.edge_index
    n = ds.data.num_nodes
    masks = stratified_split(y, ratios=(0.6, 0.2, 0.2), seed=_args.split_seed)
    tr, va, te = masks
    ei_pruned = prune_cross_split_edges(ei_full, masks)
    in_dim, out_dim = dim, int(y.max().item()) + 1     # mean 融合恒为 768

    conds = ["full", "drop_text", "drop_image", "dropout_text_50", "dropout_image_50"]
    table = {}
    for model in models:
        for cond in conds:
            accs, f1s = [], []
            for seed in range(_args.init_seeds):
                rng = np.random.default_rng(seed)
                # 在 mean 融合前对单个模态做破坏，再逐元素平均（drop_text→等价 image-only，
                # drop_image→等价 text-only；dropout_50 是真正的部分破坏鲁棒探针）
                blocks = []
                for mi, xb in enumerate(mods):
                    b = xb.clone()
                    if cond == "drop_text" and mi == 0:
                        b = torch.zeros_like(b)
                    elif cond == "drop_image" and mi == 1:
                        b = torch.zeros_like(b)
                    elif cond == "dropout_text_50" and mi == 0:
                        msk = torch.ones(dim); msk[rng.choice(dim, dim // 2, replace=False)] = 0.0
                        b = b * msk
                    elif cond == "dropout_image_50" and mi == 1:
                        msk = torch.ones(dim); msk[rng.choice(dim, dim // 2, replace=False)] = 0.0
                        b = b * msk
                    blocks.append(b)
                xx = torch.nn.functional.normalize(torch.stack(blocks, dim=0).mean(dim=0), dim=1)
                data = Data(x=xx, edge_index=ei_pruned, y=y); data.num_nodes = n
                torch.manual_seed(seed)
                m = get_model(model, in_dim, _args.hidden, out_dim,
                              num_layers=_args.layers, dropout=_args.dropout,
                              proj_dim=_args.proj_dim if _args.proj_dim else None)
                acc, f1 = _train_eval(m, data, tr, va, te, _args.epochs, model_lr(model), device)
                accs.append(acc); f1s.append(f1)
            key = f"{model}+{cond}"
            table[key] = {"cond": cond, "model": model,
                          "acc": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                          "f1": float(np.mean(f1s)), "f1_std": float(np.std(f1s))}
            logger.info(f"  {model}+{cond}: acc={np.mean(accs):.3f}±{np.std(accs):.3f}")
    summary = {"domain": domain, "protocol": "leak_free_modality_missing", "fusion": "mean",
               "modalities": ["text", "image"], "split_seed": _args.split_seed,
               "init_seeds": _args.init_seeds, "conditions": conds, "table": table,
               "note": "drop_text≡image-only, drop_image≡text-only under mean fusion"}
    sp = out_dir / f"modmiss_{domain}_summary.json"
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"saved {sp}")
    return summary


def main():
    global _args
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--domain", required=True, choices=["steam", "tmdb", "wikiart", "arxiv", "spotify"])
    _pre.add_argument("--gpu", type=int, default=None)
    _pre.add_argument("--models", default="gcn,gat,graphsage")
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
    out_dir = PROJECT_ROOT / "benchmark" / "results" / "newdomain_modality_missing"
    out_dir.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in _args.models.split(",") if m.strip()]
    run_modality_missing(_args.domain, models, device, out_dir)


if __name__ == "__main__":
    main()
