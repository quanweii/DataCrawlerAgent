#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IGC graph-subset helper —— 把"选哪些关系"变成"换 edge_index"。

设计
----
特征/节点/标签/domain base 一次加载（贵），每个策略只换 edge_index（便宜，全 tensor 向量化）。
不改动 load_domain_graph（避免破坏 ~30 个 importer），本地读 v3.json 按关系集合筛边。

base 里预按关系分组边为 (src_tensor, dst_tensor)；subset 时 concat + 去重 + 无向；
homophily 用 base["y"] 向量化比对（label_index 已按节点顺序）。
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import torch

from benchmark.data.domain_loader import DOMAIN_GRAPH, _abs, build_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_raw(domain: str):
    gpath = _abs(DOMAIN_GRAPH[domain])
    with open(gpath, encoding="utf-8") as f:
        g = json.load(f)
    return g["nodes"], g["edges"]


def load_domain_base(domain: str, modality: str = "text+image") -> dict:
    """加载域 base：节点/标签/特征 + 按关系分组的边(tensor)。只调一次。"""
    nodes, edges = _load_raw(domain)
    node_ids = [n["id"] for n in nodes]
    id2idx = {nid: i for i, nid in enumerate(node_ids)}

    str2idx: dict[str, int] = {}
    y_list = []
    for n in nodes:
        attr = n["attributes"]
        li = attr.get("label_index")
        if li is None:
            lab = attr["label"]
            if lab not in str2idx:
                str2idx[lab] = len(str2idx)
            li = str2idx[lab]
        y_list.append(int(li))
    y = torch.tensor(y_list, dtype=torch.long)

    # 按关系分组【去自环】的 (src_list, dst_list)
    rel_src: dict[str, list[int]] = defaultdict(list)
    rel_dst: dict[str, list[int]] = defaultdict(list)
    rel_same: dict[str, int] = defaultdict(int)
    rel_tot: dict[str, int] = defaultdict(int)
    for e in edges:
        rel = e.get("attributes", {}).get("relation", "?")
        s, t = e["source"], e["target"]
        if s == t or s not in id2idx or t not in id2idx:
            continue
        a, b = id2idx[s], id2idx[t]
        rel_src[rel].append(a); rel_dst[rel].append(b)
        rel_tot[rel] += 1
        if y_list[a] == y_list[b]:
            rel_same[rel] += 1

    rel_edges = {r: (torch.tensor(rel_src[r], dtype=torch.long),
                     torch.tensor(rel_dst[r], dtype=torch.long))
                 for r in rel_src}
    relations = {r: rel_tot[r] for r in rel_src}
    per_rel_homo = {r: (rel_same[r] / max(rel_tot[r], 1), rel_tot[r]) for r in rel_src}

    graph_for_feat = {"node_ids": node_ids}
    x = build_features(domain, graph_for_feat, modality)

    return {
        "domain": domain, "modality": modality,
        "node_ids": node_ids, "id2idx": id2idx,
        "y": y, "num_nodes": len(node_ids), "x": x,
        "rel_edges": rel_edges,            # rel -> (src_tensor, dst_tensor)
        "relations": relations,            # rel -> 原始边数
        "per_rel_homo": per_rel_homo,      # rel -> (homophily, count)
        "all_relations": sorted(rel_src.keys()),
    }


def subset_edge_index(base: dict, rel_set: set[str]) -> tuple[torch.Tensor, float, int]:
    """按关系集合构造无向、去自环、去重的 edge_index（全 tensor）。"""
    srcs, dsts = [], []
    for rel in rel_set:
        if rel not in base["rel_edges"]:
            continue
        s, d = base["rel_edges"][rel]
        srcs.append(s); dsts.append(d)
    if not srcs:
        return torch.empty((2, 0), dtype=torch.long), 0.0, 0
    src = torch.cat(srcs); dst = torch.cat(dsts)
    y = base["y"]
    homophily = float((y[src] == y[dst]).float().mean().item()) if src.numel() else 0.0

    # 无向去重：每条边规范成 (min,max)，unique
    lo = torch.minimum(src, dst); hi = torch.maximum(src, dst)
    keys = torch.stack([lo, hi], dim=1)
    uniq = torch.unique(keys, dim=0)
    ei = torch.empty((2, 2 * uniq.size(0)), dtype=torch.long)
    ei[0, 0::2] = uniq[:, 0]; ei[1, 0::2] = uniq[:, 1]   # 正向
    ei[0, 1::2] = uniq[:, 1]; ei[1, 1::2] = uniq[:, 0]   # 反向
    return ei, homophily, int(uniq.size(0))


def measure_subset(base: dict, rel_set: set[str], split_seed: int = 1,
                   prune: bool = False) -> dict:
    """对 G(S) 计算：LP-ceiling、homophily、边数、per-rel homo。全确定性。

    prune:
      - False (默认, IGC 头条用) → LP-ceiling 在【原始构造图】上算 = 结构泄露信号。
        含定义性边(如 same_artist)的图 LP-ceiling 高；构造时把它们排除则低。
        这是 IGC 的核心指标：构造者能否在【构建期】(关系选择)而非【评估期】(剪枝)消除泄露。
      - True → 跨切分边剪枝后再算（与 leak-free NC 主表口径一致，用于 NC 一致性对照）。
    """
    from benchmark.data.splits import (stratified_split, prune_cross_split_edges,
                                       label_propagation_ceiling)
    ei, homo, n_edges = subset_edge_index(base, rel_set)
    y = base["y"]
    masks = stratified_split(y, ratios=(0.6, 0.2, 0.2), seed=split_seed)
    tr, va, te = masks
    ei_eval = prune_cross_split_edges(ei, masks) if prune else ei
    lp = label_propagation_ceiling(y, ei_eval, tr, te)
    return {
        "edge_index": ei, "homophily": homo, "n_edges": n_edges,
        "lp_ceiling": lp["acc"], "majority_acc": lp["majority_acc"],
        "per_rel_homo": {r: round(base["per_rel_homo"][r][0], 3)
                         for r in rel_set if r in base["per_rel_homo"]},
        "relations_used": sorted(rel_set),
    }


def per_rel_lp_profile(base, split_seed=1):
    """每个关系【单独】成图的 LP-ceiling（泄露画像）。域级常量、与选中子集无关 → 一次算好复用。
    盲模式下这是 critic 定位泄露关系的唯一信号（测量能力，静态规则没有）。"""
    return {r: round(measure_subset(base, {r}, split_seed=split_seed, prune=False)["lp_ceiling"], 3)
            for r in base["all_relations"]}


def measure_leakage(base, rel_set, split_seed=1, prlp=None):
    """IGC 头条：原始构造图(未剪枝)上的 LP-ceiling = 结构泄露信号。

    prlp: 可选的预算 per_rel_lp_profile（避免每轮重算）。提供则直接挂到结果里。
    """
    m = measure_subset(base, rel_set, split_seed=split_seed, prune=False)
    m["per_rel_lp_ceiling"] = prlp if prlp is not None else \
        per_rel_lp_profile(base, split_seed=split_seed)
    return m
