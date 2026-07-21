#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无泄漏节点分类（T0②）—— 跨切分边剪枝 + LP-ceiling 双报。

为什么
------
wikiart NC 0.970 被 shared_style/same_artist 边结构泄露抬高（LP-ceiling 无特征=0.915）。
审稿人要的是"修好"不只是"披露"。本 runner：
  1. stratified 60/20/20 切分（固定 seed=1，所有模型共用同一切分 → 方差只来自初始化）。
  2. 删掉所有跨 {train,val,test} 的边 → 三者不相连 → label propagation 无法从 train 抵达 test。
  3. 在【剪枝图】上重算 LP-ceiling：应跌到≈多数类 = 泄露已消除的铁证。
  4. 在剪枝图上跑 6 GNN + MLP（固定切分、变初始化 5 seed），报 acc + macro-F1 + std。
输出与随机切分基线（newdomain_nc）并列 → 论文"保守双报"。

用法
  python benchmark/run_leakage_free_nc.py --domain wikiart --gpu 6 \\
      --modalities text+image --models mlp,gcn,gat,graphsage,gin,gcnii --seeds 5
"""
import os
import argparse

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--domain", required=True, choices=["steam", "tmdb", "wikiart", "arxiv"])
_pre.add_argument("--gpu", type=int, default=None)
_pre.add_argument("--modalities", default="text+image")
_pre.add_argument("--models", default="mlp,gcn,gat,graphsage,gin,gcnii")
_pre.add_argument("--seeds", type=int, default=5)
_pre.add_argument("--epochs", type=int, default=300)
_pre.add_argument("--split-seed", type=int, default=1)
_args, _ = _pre.parse_known_args()
if _args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_args.gpu)

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from loguru import logger

logger.add(sys.stderr, level="INFO")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.data.domain_loader import SingleDomainDataset
from benchmark.data.splits import (stratified_split, prune_cross_split_edges,
                                   label_propagation_ceiling)
from benchmark.models.gnn import get_model
from benchmark.tasks.node_classification import train_and_eval


def run_one(domain, modality, models, device, out_dir):
    ds = SingleDomainDataset(domain, modality=modality)
    y = ds.data.y
    ei_full = ds.data.edge_index
    n = ds.data.num_nodes

    # 固定切分（所有模型共用 → 公平 + 方差只来自初始化）
    masks = stratified_split(y, ratios=(0.6, 0.2, 0.2), seed=_args.split_seed)
    tr, va, te = masks
    ei_pruned = prune_cross_split_edges(ei_full, masks)
    kept = ei_pruned.size(1)
    orig = ei_full.size(1)
    logger.info(f"[{domain}/{modality}] 跨切分边剪枝: {orig} -> {kept} "
                f"(删 {orig-kept}, {100*(orig-kept)/max(orig,1):.1f}%)")

    # LP-ceiling：原始图 vs 剪枝图（证明泄露已消除）
    lp_orig = label_propagation_ceiling(y, ei_full, tr, te)
    lp_pruned = label_propagation_ceiling(y, ei_pruned, tr, te)
    logger.info(f"[{domain}] LP-ceiling(无特征): 原始图 acc={lp_orig['acc']:.3f} "
                f"-> 剪枝图 acc={lp_pruned['acc']:.3f} (多数类={lp_pruned['majority_acc']:.3f})")

    # 固定切分 + 剪枝图上跑各模型（变初始化 seed）
    data = Data(x=ds.data.x, edge_index=ei_pruned, y=y)
    data.num_nodes = n
    in_dim = data.x.size(1)
    out_dim = int(y.max().item()) + 1

    table = {}
    for model in models:
        accs, f1s = [], []
        for seed in range(_args.seeds):
            torch.manual_seed(seed)
            m = get_model(model, in_dim, 256, out_dim, num_layers=2, dropout=0.5)
            r = train_and_eval(m, data, tr, va, te, num_epochs=_args.epochs,
                               lr=0.01, device=device)
            accs.append(r["test_acc"]); f1s.append(r["test_f1"])
        table[model] = {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                        "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s))}
        logger.info(f"  [{model}] acc={np.mean(accs):.3f}±{np.std(accs):.3f} "
                    f"macro-F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}")

    result = {
        "domain": domain, "modality": modality, "split": "leakage_free_pruned",
        "split_seed": _args.split_seed, "seeds": _args.seeds,
        "edges_original": orig, "edges_pruned": kept, "edges_removed": orig - kept,
        "lp_ceiling_original": lp_orig, "lp_ceiling_pruned": lp_pruned,
        "nc": table,
    }
    sp = out_dir / f"leakfree_{domain}_{modality.replace('+','_')}.json"
    with open(sp, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"保存到 {sp}")
    return result


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = PROJECT_ROOT / "benchmark" / "results" / "leakage_free_nc"
    out_dir.mkdir(parents=True, exist_ok=True)
    modalities = [m.strip() for m in _args.modalities.split(",") if m.strip()]
    models = [m.strip() for m in _args.models.split(",") if m.strip()]
    for modality in modalities:
        run_one(_args.domain, modality, models, device, out_dir)


if __name__ == "__main__":
    main()
