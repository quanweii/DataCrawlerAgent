#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
边关系类型预测（T3⑬）—— 任务多样性：证明基准支持 NC/LP 之外的边级分类。

任务
----
给定一条边 (u,v)，预测它的关系类型（relation）。
  steam:   shared_genre / same_developer / same_publisher / shared_category
  tmdb:    shared_genre / same_director / shared_actor
  wikiart: same_artist / shared_style
arxiv 单一关系(cites)排除。

模型
----
GCN 编码节点特征 → 边分类头 MLP([z_u; z_v; |z_u - z_v|]) → relation logits。
分层(按 relation)切边 60/20/20，5 初始化 seed，报 acc + macro-F1 + std。
这是 MAG 基准应有的「图结构语义」任务（边的关系类型本身是多模态图的关键语义）。

用法
  python benchmark/run_edge_relation_prediction.py --domain tmdb --gpu 6 --seeds 5
"""
import os
import sys
import argparse
import json
import random
from pathlib import Path

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--domain", required=True, choices=["steam", "tmdb", "wikiart"])
_pre.add_argument("--gpu", type=int, default=None)
_pre.add_argument("--modality", default="text+image")
_pre.add_argument("--seeds", type=int, default=5)
_pre.add_argument("--epochs", type=int, default=200)
_pre.add_argument("--max-edges-per-relation", type=int, default=50000,
                  help="每关系采样上限，平衡类别 + 控制规模")
_args, _ = _pre.parse_known_args()
if _args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_args.gpu)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, f1_score
from loguru import logger

logger.add(sys.stderr, level="INFO")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.data.domain_loader import SingleDomainDataset, DOMAIN_GRAPH
from benchmark.data.splits import stratified_split
import json as _json


def collect_edges_by_relation(domain):
    """读 v3.json，返回 [(src_idx, dst_idx, rel_id), ...] 与 rel->id 映射。"""
    gpath = PROJECT_ROOT / DOMAIN_GRAPH[domain]
    with open(gpath) as f:
        g = _json.load(f)
    ids = [n["id"] for n in g["nodes"]]
    id2idx = {nid: i for i, nid in enumerate(ids)}
    rel2id = {}
    triples = []
    for e in g["edges"]:
        rel = e["attributes"].get("relation", "?")
        if rel not in rel2id:
            rel2id[rel] = len(rel2id)
        s, t = e["source"], e["target"]
        if s in id2idx and t in id2idx:
            triples.append((id2idx[s], id2idx[t], rel2id[rel]))
    return triples, rel2id


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = PROJECT_ROOT / "benchmark" / "results" / "edge_relation_prediction"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = SingleDomainDataset(_args.domain, modality=_args.modality)
    triples, rel2id = collect_edges_by_relation(_args.domain)
    # 每关系采样上限（类别平衡 + 规模控制）
    rng = random.Random(0)
    by_rel = {}
    for s, d, r in triples:
        by_rel.setdefault(r, []).append((s, d, r))
    cap = _args.max_edges_per_relation
    balanced = []
    for r, lst in by_rel.items():
        rng.shuffle(lst)
        balanced.extend(lst[:cap])
    edges = np.array([(s, d) for s, d, _ in balanced], dtype=np.int64)
    labels = np.array([r for _, _, r in balanced], dtype=np.int64)
    n_rel = len(rel2id)
    logger.info(f"[{_args.domain}] relations={rel2id} total={len(triples)} "
                f"sampled={len(edges)} (cap={cap}/rel) n_rel={n_rel}")

    x = ds.data.x.to(device)
    ei_full = ds.data.edge_index.to(device)  # 消息传递用全图
    in_dim = x.size(1)
    hidden = 256

    class EdgeRelModel(nn.Module):
        def __init__(self, in_dim, hidden, n_rel):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hidden)
            self.conv2 = GCNConv(hidden, hidden)
            self.head = nn.Sequential(
                nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(hidden, n_rel))

        def encode(self, x, ei):
            h = F.relu(self.conv1(x, ei)); h = self.conv2(h, ei)
            return h

        def forward(self, x, ei_msg, edge_index):
            z = self.encode(x, ei_msg)
            zs, zd = z[edge_index[0]], z[edge_index[1]]
            feat = torch.cat([zs, zd, (zs - zd).abs()], dim=1)
            return self.head(feat)

    table = {}
    for seed in range(_args.seeds):
        torch.manual_seed(seed)
        # 分层切边（按 relation）
        order = np.arange(len(labels))
        np_rng = np.random.RandomState(seed)
        tr_m = np.zeros(len(labels), bool); va_m = np.zeros(len(labels), bool); te_m = np.zeros(len(labels), bool)
        for r in np.unique(labels):
            idx = order[labels == r]
            np_rng.shuffle(idx)
            n_tr = int(len(idx) * 0.6); n_va = int(len(idx) * 0.2)
            tr_m[idx[:n_tr]] = True; va_m[idx[n_tr:n_tr + n_va]] = True; te_m[idx[n_tr + n_va:]] = True
        tr = torch.tensor(edges[tr_m].T, device=device)
        va = torch.tensor(edges[va_m].T, device=device)
        te = torch.tensor(edges[te_m].T, device=device)
        ytr = torch.tensor(labels[tr_m], device=device)
        yva = torch.tensor(labels[va_m], device=device)
        yte = torch.tensor(labels[te_m], device=device)

        model = EdgeRelModel(in_dim, hidden, n_rel).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        crit = nn.CrossEntropyLoss()
        best, bestst, pat = -1.0, None, 0
        for ep in range(1, _args.epochs + 1):
            model.train(); opt.zero_grad()
            out = model(x, ei_full, tr)
            loss = crit(out, ytr); loss.backward(); opt.step()
            if ep % 5 == 0:
                model.eval()
                with torch.no_grad():
                    vacc = accuracy_score(yva.cpu(), model(x, ei_full, va).argmax(1).cpu())
                if vacc > best:
                    best = vacc; bestst = {k: v.cpu().clone() for k, v in model.state_dict().items()}; pat = 0
                else:
                    pat += 5
                    if pat >= 50:
                        break
        model.load_state_dict(bestst); model.to(device); model.eval()
        with torch.no_grad():
            pred = model(x, ei_full, te).argmax(1).cpu()
        acc = accuracy_score(yte.cpu(), pred)
        f1 = f1_score(yte.cpu(), pred, average="macro", zero_division=0)
        # majority 基线
        maj = (yte.cpu().numpy() == np.bincount(labels[te_m]).argmax()).mean()
        table[f"seed{seed}"] = {"acc": float(acc), "macro_f1": float(f1), "majority_acc": float(maj)}
        logger.info(f"  [seed{seed}] acc={acc:.3f} macro-F1={f1:.3f} (majority={maj:.3f})")

    accs = [v["acc"] for v in table.values()]; f1s = [v["macro_f1"] for v in table.values()]
    res = {"domain": _args.domain, "modality": _args.modality, "relations": rel2id,
           "n_relations": n_rel, "n_edges_sampled": len(edges), "seeds": _args.seeds,
           "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
           "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
           "majority_acc": float(np.mean([v["majority_acc"] for v in table.values()])),
           "per_seed": table}
    sp = out_dir / f"edgerel_{_args.domain}.json"
    with open(sp, "w") as f:
        json.dump(res, f, indent=2)
    logger.info(f"[{_args.domain}] >>> acc={np.mean(accs):.3f}±{np.std(accs):.3f} "
                f"macro-F1={np.mean(f1s):.3f}±{np.std(f1s):.3f} 保存到 {sp}")


if __name__ == "__main__":
    main()
