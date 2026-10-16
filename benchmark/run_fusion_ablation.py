#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模态融合消融（T1④）—— 回答「多模态是否只是拼接？」

动机
----
3 新域 NC/LP 只跑了 vanilla GNN+拼接特征，审稿人会问「这跟多模态有什么关系？」。
本 runner 对比 4 种【早期融合】策略（同一 GCN backbone、同一固定切分、同 5 初始化 seed）：
  concat     拼接（= 现有基线）
  attention  可学习模态权重（ModalityAttentionFusion，repo 现成死代码，首次启用）
  mean       各模态 L2 归一化后逐元素平均
  sum        各模态 L2 归一化后逐元素求和
并报告 attention 学到的模态权重（跨域看「谁是主导模态」）。

用法
  python benchmark/run_fusion_ablation.py --domain steam --gpu 6 \\
      --modalities text+image --backbone gcn --seeds 5
"""
import os
import sys
import argparse
import json
from pathlib import Path

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--domain", required=True, choices=["steam", "tmdb", "wikiart", "arxiv"])
_pre.add_argument("--gpu", type=int, default=None)
_pre.add_argument("--modalities", default="text+image")
_pre.add_argument("--backbone", default="gcn", choices=["gcn", "gat", "graphsage"])
_pre.add_argument("--fusions", default="concat,attention,mean,sum")
_pre.add_argument("--seeds", type=int, default=5)
_pre.add_argument("--epochs", type=int, default=300)
_args, _ = _pre.parse_known_args()
if _args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_args.gpu)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from loguru import logger

logger.add(sys.stderr, level="INFO")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.data.domain_loader import load_domain_graph, _load_emb
from benchmark.data.splits import stratified_split
from benchmark.models.gnn import get_model, MultimodalGNN
from benchmark.tasks.node_classification import train_and_eval


def load_per_modality(domain, modalities, node_ids):
    """返回 {mod: L2-normalized Tensor[N,768]}，对齐 node_ids 顺序。"""
    out = {}
    for m in modalities:
        x = _load_emb(domain, m, node_ids)
        x = x / x.norm(dim=1, keepdim=True).clamp(min=1e-8)  # 每模态 L2 归一化
        out[m] = x
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = PROJECT_ROOT / "benchmark" / "results" / "fusion_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    modalities = [m.strip() for m in _args.modalities.replace(",", "+").split("+") if m.strip()]
    fusions = [f.strip() for f in _args.fusions.split(",") if f.strip()]
    backbone = _args.backbone

    graph = load_domain_graph(_args.domain)
    node_ids = graph["node_ids"]
    ei = graph["edge_index"].to(device)
    y = graph["y"].to(device)
    n = graph["num_nodes"]
    C = int(y.max().item()) + 1
    per_mod = load_per_modality(_args.domain, modalities, node_ids)
    dims = {m: per_mod[m].size(1) for m in modalities}
    logger.info(f"[{_args.domain}] modalities={modalities} dims={dims}")

    masks = stratified_split(y.cpu(), ratios=(0.6, 0.2, 0.2), seed=1)
    tr, va, te = [m.to(device) for m in masks]

    table = {}
    learned_weights = {}
    for fusion in fusions:
        accs, f1s = [], []
        for seed in range(_args.seeds):
            torch.manual_seed(seed)
            if fusion == "attention":
                model = MultimodalGNN(dims, 256, C, num_layers=2, dropout=0.5,
                                      gnn_type=backbone).to(device)
                emb_d = {m: per_mod[m].to(device) for m in modalities}
                # 自定义训练循环（需传 modality_embeddings）
                opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
                crit = nn.CrossEntropyLoss()
                best, beststate, pat = -1, None, 0
                for ep in range(1, _args.epochs + 1):
                    model.train(); opt.zero_grad()
                    out = model(None, ei, modality_embeddings=emb_d)
                    loss = crit(out[tr], y[tr]); loss.backward(); opt.step()
                    if ep % 10 == 0:
                        model.eval()
                        with torch.no_grad():
                            out = model(None, ei, modality_embeddings=emb_d)
                            from sklearn.metrics import accuracy_score
                            va_acc = accuracy_score(y[va].cpu(), out[va].argmax(1).cpu())
                        if va_acc > best:
                            best = va_acc; beststate = {k: v.cpu().clone() for k, v in model.state_dict().items()}; pat = 0
                        else:
                            pat += 10
                        if pat >= 50:
                            break
                model.load_state_dict(beststate); model.eval()
                with torch.no_grad():
                    out = model(None, ei, modality_embeddings={k: v.to(device) for k, v in emb_d.items()})
                    from sklearn.metrics import accuracy_score, f1_score
                    pred = out.argmax(1)
                    accs.append(accuracy_score(y[te].cpu(), pred[te].cpu()))
                    f1s.append(f1_score(y[te].cpu(), pred[te].cpu(), average="macro", zero_division=0))
                if seed == 0:
                    learned_weights[fusion] = model.get_modality_weights()
            else:
                # concat / mean / sum → 预融合成单一 x，用 plain backbone
                xs = [per_mod[m] for m in modalities]
                if fusion == "concat":
                    xf = torch.cat(xs, dim=1)
                elif fusion == "mean":
                    xf = torch.stack(xs, dim=0).mean(dim=0)
                else:  # sum
                    xf = torch.stack(xs, dim=0).sum(dim=0)
                xf = xf / xf.norm(dim=1, keepdim=True).clamp(min=1e-8)
                data = Data(x=xf.to(device), edge_index=ei, y=y)
                data.num_nodes = n
                m = get_model(backbone, xf.size(1), 256, C, num_layers=2, dropout=0.5)
                r = train_and_eval(m, data, tr, va, te, num_epochs=_args.epochs, lr=0.01, device=device)
                accs.append(r["test_acc"]); f1s.append(r["test_f1"])
        table[fusion] = {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                         "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s))}
        logger.info(f"  [{fusion:<10}] acc={np.mean(accs):.3f}±{np.std(accs):.3f} "
                    f"macro-F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}"
                    + (f"  weights={learned_weights.get(fusion)}" if fusion == "attention" else ""))

    res = {"domain": _args.domain, "modalities": modalities, "backbone": backbone,
           "seeds": _args.seeds, "homophily": graph["homophily"],
           "fusion_results": table, "attention_weights": learned_weights}
    sp = out_dir / f"fusion_{_args.domain}_{'+'.join(modalities)}.json"
    with open(sp, "w") as f:
        json.dump(res, f, indent=2)
    logger.info(f"保存到 {sp}")


if __name__ == "__main__":
    main()
