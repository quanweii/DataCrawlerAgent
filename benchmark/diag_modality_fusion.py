#!/usr/bin/env python3
"""Diagnostic v2: isolate the modality-FUSION strategy as the only variable, using the
REAL train_and_eval loop (same as run_newdomain_nc.py headline), leak-free protocol.

Question: which fusion makes multimodal >= unimodal-text (the sane result), and which
reproduces the current "multimodal < text" artifact?

Strategies (all built from the SAME aligned CLIP text & image 768-d embeddings):
  text       : text emb (768)                      -> GNN(768)   [unimodal baseline]
  image      : image emb (768)                     -> GNN(768)   [unimodal baseline]
  mean       : (text+image)/2, L2-norm (768)       -> GNN(768)   [OpenMAG element-wise fusion]
  sum        : text+image, L2-norm (768)           -> GNN(768)   [OpenMAG element-wise variant]
  concat     : [text;image] (1536), L2-norm        -> GNN(1536)  [standard GNN, NO projection]
  bottleneck : [text;image] (1536) -> Linear(1536,768) -> GNN(768)   [CURRENT bug: only multimodal projected]

For text/image (768) the bottleneck path is identity (matches get_model in==proj rule).
We do NOT project text/image/mean/sum/concat — only 'bottleneck' adds the Linear, exactly as now.

Usage: python benchmark/diag_modality_fusion.py --gpu 1
"""
import os, argparse, sys
_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--gpu", type=int, default=1)
_p.add_argument("--domains", default="wikiart,steam,tmdb")
_p.add_argument("--models", default="gcnii,gcn")
_p.add_argument("--seeds", type=int, default=3)
_p.add_argument("--epochs", type=int, default=300)
_a, _ = _p.parse_known_args()
if _a.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_a.gpu)

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from benchmark.data.domain_loader import SingleDomainDataset, build_features, load_domain_graph
from benchmark.data.splits import stratified_split, prune_cross_split_edges
from benchmark.models.gnn import MODEL_REGISTRY
from benchmark.tasks.node_classification import train_and_eval

SPLIT_SEED = 1
HIDDEN = 256


class PerModality(nn.Module):
    def __init__(self, base_factory, n_mod, mod_dim=768, proj=768):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(mod_dim, proj) for _ in range(n_mod)])
        self.base = base_factory(n_mod * proj)
    def forward(self, x, ei):
        d = x.size(1) // len(self.proj)
        chunks = [self.proj[i](x[:, i*d:(i+1)*d]) for i in range(len(self.proj))]
        return self.base(torch.cat(chunks, dim=1), ei)


class Bottleneck(nn.Module):
    def __init__(self, base_factory, in_dim, proj=768):
        super().__init__()
        self.p = nn.Linear(in_dim, proj)
        self.base = base_factory(proj)
    def forward(self, x, ei): return self.base(self.p(x), ei)


def build_x(domain, graph, strategy):
    xt = build_features(domain, graph, "text")      # 768, L2-normed
    xi = build_features(domain, graph, "image")     # 768, L2-normed
    if strategy == "text":   return xt
    if strategy == "image":  return xi
    if strategy == "mean":   return F.normalize((xt + xi) / 2, dim=1)
    if strategy == "sum":    return F.normalize(xt + xi, dim=1)
    if strategy == "concat": return F.normalize(torch.cat([xt, xi], dim=1), dim=1)
    if strategy == "bottleneck": return F.normalize(torch.cat([xt, xi], dim=1), dim=1)
    raise ValueError(strategy)


def make_model(strategy, in_dim, model_name, out_dim):
    base_factory = lambda d: MODEL_REGISTRY[model_name](d, HIDDEN, out_dim, num_layers=2, dropout=0.5)
    if strategy == "bottleneck" and in_dim != 768:
        return Bottleneck(base_factory, in_dim, 768)
    if strategy == "permod":
        return PerModality(base_factory, 2, 768, 768)
    return base_factory(in_dim)   # text/image/mean/sum/concat -> raw GNN at in_dim


def run_cell(domain, model_name, strategy, ei_pruned, y, n, masks, x, device):
    tr, va, te = masks
    data = Data(x=x, edge_index=ei_pruned, y=y); data.num_nodes = n
    out_dim = int(y.max().item()) + 1
    in_dim = x.size(1)
    accs, f1s = [], []
    for s in range(_a.seeds):
        torch.manual_seed(s)
        m = make_model(strategy, in_dim, model_name, out_dim)
        r = train_and_eval(m, data, tr, va, te, num_epochs=_a.epochs, lr=0.01, device=device)
        accs.append(r["test_acc"]); f1s.append(r["test_f1"])
    return float(np.mean(accs)), float(np.std(accs))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} seeds={_a.seeds} epochs={_a.epochs}  (REAL train_and_eval loop)")
    strategies = ["text", "image", "mean", "sum", "concat", "bottleneck"]
    models = _a.models.split(",")
    results = {}
    for dom in _a.domains.split(","):
        dom = dom.strip()
        graph = load_domain_graph(dom)
        ds = SingleDomainDataset(dom, modality="text+image")
        y = ds.data.y; ei_full = ds.data.edge_index; n = ds.data.num_nodes
        masks = stratified_split(y, ratios=(0.6,0.2,0.2), seed=SPLIT_SEED)
        ei_pruned = prune_cross_split_edges(ei_full, masks)
        print(f"\n===== {dom} (nodes={n}, pruned_edges={ei_pruned.size(1)}) =====")
        for model_name in models:
            print(f"--- {model_name} ---")
            cell = {}
            for strat in strategies:
                x = build_x(dom, graph, strat)
                acc, std = run_cell(dom, model_name, strat, ei_pruned, y, n, masks, x, device)
                cell[strat] = (acc, std)
                print(f"  {strat:11s}: {acc:.4f}±{std:.4f}  (in_dim={x.size(1)})")
            tx = cell["text"][0]
            print(f"  >> (multi - text):  " + "  ".join(
                f"{s}={cell[s][0]-tx:+.4f}" for s in ["mean","sum","concat","bottleneck"]))
            results[f"{dom}/{model_name}"] = {s: {"acc":v[0],"std":v[1]} for s,v in cell.items()}
    out = PROJECT_ROOT/"benchmark"/"results"/"diag_modality_fusion.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"split_seed":SPLIT_SEED,"seeds":_a.seeds,"epochs":_a.epochs,"results":results},
              open(out,"w"), indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
