#!/usr/bin/env python3
"""Training efficiency profiling for new-domain MAGs (P2-1, OpenMAG Q9/Q10 arm).

Profiles time/epoch + peak GPU memory + #params for GCN/GAT/GraphSAGE on the
leak-free pruned graph of each domain. No accuracy goal here — pure efficiency.

Usage:
  python benchmark/run_newdomain_efficiency.py --domains steam,tmdb,wikiart,arxiv --gpu 0
"""
import os
import sys
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.data.domain_loader import SingleDomainDataset
from benchmark.data.splits import stratified_split, prune_cross_split_edges
from benchmark.models.gnn import get_model

_args = None


def profile(model, data, device, epochs=30, lr=0.01):
    model = model.to(device); data = data.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    crit = torch.nn.CrossEntropyLoss()
    if device != "cpu":
        torch.cuda.reset_peak_memory_stats(device)
    # warmup
    for _ in range(3):
        model.train(); opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = crit(out[data.train_mask], data.y[data.train_mask]); loss.backward(); opt.step()
    if device != "cpu":
        torch.cuda.synchronize()
    times = []
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        t0 = time.time()
        out = model(data.x, data.edge_index)
        loss = crit(out[data.train_mask], data.y[data.train_mask]); loss.backward(); opt.step()
        if device != "cpu":
            torch.cuda.synchronize()
        times.append(time.time() - t0)
    peak = 0.0
    if device != "cpu":
        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    params = sum(p.numel() for p in model.parameters())
    return {"time_per_epoch_ms": float(np.mean(times) * 1000),
            "time_std_ms": float(np.std(times) * 1000),
            "peak_memory_mb": float(peak), "params": int(params)}


def run_domain(domain, device, out_dir):
    mod = "text+image+formula" if domain == "arxiv" else "text+image"
    ds = SingleDomainDataset(domain, modality=mod)
    y = ds.data.y; ei = ds.data.edge_index; n = ds.data.num_nodes
    masks = stratified_split(y, ratios=(0.6, 0.2, 0.2), seed=1)
    ei_p = prune_cross_split_edges(ei, masks)
    data = Data(x=ds.data.x, edge_index=ei_p, y=y); data.num_nodes = n
    data.train_mask, data.val_mask, data.test_mask = masks
    in_dim = data.x.size(1); out_dim = int(y.max().item()) + 1
    table = {}
    for m in ["gcn", "gat", "graphsage"]:
        torch.manual_seed(0)
        model = get_model(m, in_dim, 256, out_dim, num_layers=2, dropout=0.5,
                          proj_dim=_args.proj_dim if _args.proj_dim else None)
        r = profile(model, data, device, epochs=30)
        table[m] = r
        logger.info(f"[{domain}] {m}: {r['time_per_epoch_ms']:.1f}ms/ep peak={r['peak_memory_mb']:.0f}MB params={r['params']}")
    res = {"domain": domain, "modality": mod, "num_nodes": n, "in_dim": in_dim,
           "edges_pruned": int(ei_p.size(1)), "device": torch.cuda.get_device_name(0) if device != "cpu" else "cpu",
           "table": table}
    sp = out_dir / f"efficiency_{domain}_summary.json"
    with open(sp, "w") as f:
        json.dump(res, f, indent=2)
    logger.info(f"saved {sp}")
    return res


def main():
    global _args
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--domains", default="steam,tmdb,wikiart,arxiv")
    _pre.add_argument("--gpu", type=int, default=None)
    _pre.add_argument("--proj-dim", type=int, default=768, help="learnable Linear 投影;0=不投影")
    _args = _pre.parse_args()
    if _args.gpu is not None and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(_args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = PROJECT_ROOT / "benchmark" / "results" / "newdomain_efficiency"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_res = {}
    for d in _args.domains.split(","):
        d = d.strip()
        if d:
            all_res[d] = run_domain(d, device, out_dir)
    sp = out_dir / "efficiency_all_summary.json"
    with open(sp, "w") as f:
        json.dump(all_res, f, indent=2)
    logger.info(f"merged → {sp}")


if __name__ == "__main__":
    main()
