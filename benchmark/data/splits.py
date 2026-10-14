#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Protocol-aware node split and graph-view utilities.

问题
----
Cross-split edges are normal in a transductive protocol and are not, by
themselves, evidence of leakage.  This module therefore exposes both the full
transductive graph and a split-disjoint robustness view.  Only target-derived
relations/features, duplicates, future information, or an explicitly violated
protocol are treated as integrity defects.

The label-propagation routine is a topology-only predictability diagnostic.  A
high value can be a legitimate property of graph structure and must not be
reported as proof of leakage without provenance evidence.

提供
----
- stratified_split(y, ratios, seed): 每类内 shuffle 的分层切分（保护小类）。
- prune_cross_split_edges(edge_index, masks...): 删跨切分边 → 剪枝 edge_index。
- label_propagation_ceiling(y, edge_index, train_mask, test_mask, iters):
  迭代式 label propagation（无特征仅结构），返回 test 上的 acc/macro-F1。
  在剪枝图上应跌到≈多数类基线 = 泄露已消除的证据。
"""
from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def stratified_split(y: torch.Tensor, ratios=(0.6, 0.2, 0.2), seed: int = 1,
                     small_class_floor: int = 5
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """每类内 shuffle 的分层 train/val/test 切分；类太小(<floor)时保证 ≥1 训练样本。"""
    rng = np.random.RandomState(seed)
    y_np = y.cpu().numpy()
    n = len(y_np)
    tr = torch.zeros(n, dtype=torch.bool)
    va = torch.zeros(n, dtype=torch.bool)
    te = torch.zeros(n, dtype=torch.bool)
    for c in np.unique(y_np):
        idx = np.where(y_np == c)[0]
        rng.shuffle(idx)
        nc = len(idx)
        if nc < small_class_floor:
            n_tr = max(1, int(nc * ratios[0]))
            tr[idx[:n_tr]] = True
            for j, ii in enumerate(idx[n_tr:]):
                (va if j % 2 == 0 else te)[ii] = True
            continue
        n_tr = int(nc * ratios[0])
        n_va = int(nc * ratios[1])
        tr[idx[:n_tr]] = True
        va[idx[n_tr:n_tr + n_va]] = True
        te[idx[n_tr + n_va:]] = True
    return tr, va, te


def split_id(masks):
    """每节点 → 它属于哪个切分(0 train/1 val/2 test/ -1 无)。masks=(tr,va,te)。"""
    tr, va, te = masks
    sid = torch.full((tr.size(0),), -1, dtype=torch.long)
    sid[tr] = 0
    sid[va] = 1
    sid[te] = 2
    return sid


def make_split_disjoint_view(edge_index: torch.Tensor, masks) -> torch.Tensor:
    """Return the within-split edge union for a split-disjoint view.

    保留：两端都在 train / 都在 val / 都在 test 的边。
    删除：任何跨切分的边（train-val / train-test / val-test）。
    """
    sid = split_id(masks)
    s, d = edge_index[0], edge_index[1]
    keep = (sid[s] == sid[d]) & (sid[s] >= 0)  # 同切分且已分配
    return edge_index[:, keep]


def prune_cross_split_edges(edge_index: torch.Tensor, masks) -> torch.Tensor:
    """Backward-compatible alias for :func:`make_split_disjoint_view`."""
    return make_split_disjoint_view(edge_index, masks)


def label_propagation_ceiling(y: torch.Tensor, edge_index: torch.Tensor,
                              train_mask: torch.Tensor, test_mask: torch.Tensor,
                              num_iters: int = 50,
                              seed_labels: torch.Tensor = None) -> dict:
    """迭代 label propagation（无特征、仅图结构）→ test 上的 acc/macro-F1。

    经典闭式/迭代 LP：用 train 的 one-hot 标签做种子，沿边扩散，test 取 argmax。
    在【原始图】上 = 结构泄露上限（wikiart ≈ 0.915）。
    在【剪枝图】上应跌到≈多数类 = 泄露被消除的证据。
    """
    y = y.cpu()
    seed_labels = y if seed_labels is None else seed_labels.cpu()
    ei = edge_index.cpu()
    C = int(y.max().item()) + 1
    N = y.size(0)
    # 种子 one-hot（仅 train 节点初始化为真实标签，其余均匀先验）
    H = torch.full((N, C), 1.0 / C)
    H[train_mask] = 0.0
    H[train_mask, seed_labels[train_mask]] = 1.0
    # 归一化邻接（对称）— 向量化度数（避免 Python 边循环）
    src, dst = ei[0], ei[1]
    deg = torch.zeros(N)
    ones = torch.ones(ei.size(1))
    deg.index_add_(0, src, ones)
    deg.index_add_(0, dst, ones)  # 无向
    deg = deg.clamp(min=1.0)
    clamp = H.clone()
    for _ in range(num_iters):
        # 聚合邻居均值
        agg = torch.zeros_like(H)
        agg.index_add_(0, src, H[dst])
        agg.index_add_(0, dst, H[src])
        agg = agg / deg.unsqueeze(1)
        H = agg
        H[train_mask] = clamp[train_mask]  # 钳制 train 标签
    pred = H[test_mask].argmax(dim=1).numpy()
    truth = y[test_mask].numpy()
    return {
        "acc": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
        "majority_acc": float((truth == np.bincount(truth).argmax()).mean()),
    }
