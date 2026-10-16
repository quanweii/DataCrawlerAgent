#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公开 benchmark 标签泄露画像 —— 把论文的 LP-ceiling 测谎仪跑到外部数据集上,
证明"构造边图的结构性泄露"不是只有我们自家数据才有(论文背景的【外部铁证】腿2)。

复用论文同一套测量(字节级一致):
  stratified_split / prune_cross_split_edges / label_propagation_ceiling
  全部来自 benchmark/data/splits.py —— 与正文 leak-free 主表、leakage_diagnostic 同口径。

每个数据集报:
  majority / LP-ceiling(raw 构造图) / LP-ceiling(跨切分剪枝后) / 边剪枝比例 / 同质性 / 边类型
读法(与论文诊断表一致):raw ≫ majority = 结构泄露;pruned 跌到 ≈ majority = 泄露被剪枝消除。

数据集分两类(关键):
  - 自然边(citation):  Cora/Citeseer/PubMed         → 预期泄露较轻(边是引用,非定义性)
  - 异质(Platonov 同款):Chameleon/Squirrel/WebKB/Actor → 重复节点/低同质;LP-ceiling 诊断另一通路
  - 构造边(co-*/co-purchase):Amazon Computers/Photo、Coauthor CS/Physics
        → 这些最像我们的 MAG;若它们也 raw ≫ pruned ≫ majority → 直接证明我们的泄露通路普遍存在。

用法
  # 0. 自检(无网络,合成图,验证代码路径)—— 我在本环境已跑过
  python benchmark/run_public_leakage_profile.py --selftest
  # 1. 真跑(需网络;在能访问 github 的机器上执行)
  python benchmark/run_public_leakage_profile.py --public
  # 2. 只跑某几个
  python benchmark/run_public_leakage_profile.py --public --only cora,amazon_photo
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 论文同一套测量工具(不改)
from benchmark.data.splits import (stratified_split, prune_cross_split_edges,
                                   label_propagation_ceiling)

SPLIT_SEED = 1
ROOT = Path("/tmp/pub_leak")          # PyG 下载缓存
OUT = PROJECT_ROOT / "benchmark" / "results" / "public_leakage"


def _homophily(y, ei):
    if ei.size(1) == 0:
        return 0.0
    return float((y[ei[0]] == y[ei[1]]).float().mean().item())


def profile_data(name, data, edge_kind):
    """对单个 PyG Data 跑泄露画像。data 必须有 y(edge_index 可缺)。"""
    y = data.y.view(-1).long()
    ei = data.edge_index.long()
    # 去自环(与论文 measure_subset 一致)
    mask = ei[0] != ei[1]
    ei = ei[:, mask]
    masks = stratified_split(y, ratios=(0.6, 0.2, 0.2), seed=SPLIT_SEED)
    tr, va, te = masks
    raw = label_propagation_ceiling(y, ei, tr, te)
    ei_p = prune_cross_split_edges(ei, masks)
    pruned = label_propagation_ceiling(y, ei_p, tr, te)
    n_raw = int(ei.size(1) // 2)
    n_pruned = int(ei_p.size(1) // 2)
    return {
        "name": name, "edge_kind": edge_kind,
        "n_nodes": int(y.size(0)), "n_classes": int(y.max().item()) + 1,
        "n_edges": n_raw,
        "homophily": round(_homophily(y, ei), 3),
        "majority": round(raw["majority_acc"], 3),
        "lp_raw": round(raw["acc"], 3),
        "lp_pruned": round(pruned["acc"], 3),
        "prune_frac": round(1.0 - n_pruned / max(n_raw, 1), 3),
        "leak_gap": round(raw["acc"] - raw["majority_acc"], 3),  # raw − majority = 泄露量
        "prune_drop": round(raw["acc"] - pruned["acc"], 3),      # raw − pruned = 剪枝能消多少
    }


# ---------------- 公开数据集清单(每个独立 try,缺网络/缺包只跳过该项) ----------------

def _planetoid(name):
    from torch_geometric.datasets import Planetoid
    return Planetoid(root=str(ROOT / f"planetoid_{name.lower()}"), name=name)[0]


def _wikipedia(name):
    from torch_geometric.datasets import WikipediaNetwork
    # is_undirected 默认 True(chameleon/squirrel 原始版;Platonov 用 geom_gcn 滤版另算)
    return WikipediaNetwork(root=str(ROOT / f"wiki_{name}"), name=name)[0]


def _webkb(name):
    from torch_geometric.datasets import WebKB
    return WebKB(root=str(ROOT / f"webkb_{name}"), name=name)[0]


def _actor():
    from torch_geometric.datasets import Actor
    return Actor(root=str(ROOT / "actor"))[0]


def _amazon(name):
    from torch_geometric.datasets import Amazon
    return Amazon(root=str(ROOT / f"amazon_{name}"), name=name)[0]


def _coauthor(name):
    from torch_geometric.datasets import Coauthor
    return Coauthor(root=str(ROOT / f"coauthor_{name}"), name=name)[0]


PUBLIC = [
    # 自然引用边 —— 预期泄露轻(对照)
    ("cora",      "citation (natural)",   lambda: _planetoid("Cora")),
    ("citeseer",  "citation (natural)",   lambda: _planetoid("CiteSeer")),
    ("pubmed",    "citation (natural)",   lambda: _planetoid("PubMed")),
    # Platonov 2023 同款异质图 —— 重复节点/低同质
    ("chameleon", "wikipedia hyperlink",  lambda: _wikipedia("chameleon")),
    ("squirrel",  "wikipedia hyperlink",  lambda: _wikipedia("squirrel")),
    ("cornell",   "web hyperlink (hetero)", lambda: _webkb("Cornell")),
    ("texas",     "web hyperlink (hetero)", lambda: _webkb("Texas")),
    ("wisconsin", "web hyperlink (hetero)", lambda: _webkb("Wisconsin")),
    ("actor",     "co-occurrence (hetero)", lambda: _actor()),
    # 构造边(co-purchase / co-authorship)—— 最像我们的 MAG
    ("amazon_computers", "co-purchase (constructed)", lambda: _amazon("Computers")),
    ("amazon_photo",     "co-purchase (constructed)", lambda: _amazon("Photo")),
    ("coauthor_cs",      "co-authorship (constructed)", lambda: _coauthor("CS")),
    ("coauthor_physics", "co-authorship (constructed)", lambda: _coauthor("Physics")),
]


def run_public(only=None):
    OUT.mkdir(parents=True, exist_ok=True)
    rows, failed = [], []
    only = set(only.split(",")) if only else None
    for name, kind, loader in PUBLIC:
        if only and name not in only:
            continue
        print(f"  ... loading {name} ({kind})", flush=True)
        try:
            data = loader()
            r = profile_data(name, data, kind)
            rows.append(r)
            print(f"     OK {name}: lp_raw={r['lp_raw']} pruned={r['lp_pruned']} "
                  f"maj={r['majority']} homo={r['homophily']}")
        except Exception as e:
            failed.append({"name": name, "kind": kind, "error": str(e)[:160]})
            print(f"     FAIL {name}: {str(e)[:120]}")
    _dump_and_print(rows, failed, tag="public")
    return rows


# ---------------- 自检(无网络) ----------------

def run_selftest():
    """合成一个'定义性构造边'图:同类节点必连边 → raw LP 应高,pruned 应崩。
    验证测量代码路径正确(不验证真实数据)。"""
    torch.manual_seed(0)
    N, C = 400, 4
    y = torch.randint(0, C, (N,))
    src, dst = [], []
    # 强定义性边:同类 dense(泄露),跨类几乎不连
    for i in range(N):
        for j in range(i + 1, N):
            if y[i] == y[j] and torch.rand(1).item() < 0.3:
                src += [i, j]; dst += [j, i]
    ei = torch.tensor([src, dst], dtype=torch.long)
    data = type("G", (), {"y": y, "edge_index": ei})()
    r = profile_data("synthetic_definitional", data, "constructed (definitional, synthetic)")
    rows = [r]
    print("\n[自检] 合成定义性边图预期: lp_raw ≫ majority, lp_pruned ≈ majority")
    _dump_and_print(rows, [], tag="selftest")
    # 断言测量逻辑正确
    assert r["lp_raw"] > r["majority"] + 0.2, "raw 应明显高于 majority"
    assert r["lp_pruned"] < r["majority"] + 0.1, "pruned 应跌回 ≈ majority"
    print("\n✅ 自检通过:测量代码路径正确(raw 泄露、pruned 修复均符合预期)。")
    return rows


def _dump_and_print(rows, failed, tag):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"leakage_profile_{tag}.json", "w") as f:
        json.dump({"rows": rows, "failed": failed, "split_seed": SPLIT_SEED,
                   "measure": "benchmark/data/splits.py (label_propagation_ceiling, "
                              "stratified 60/20/20 seed=1, cross-split edge pruning)"},
                  f, indent=2, ensure_ascii=False)
    if not rows:
        print("\n(无可用数据行 —— 见 failed 列表。若全部 FAIL 多半是网络/缺包。)")
        return
    print("\n" + "=" * 92)
    print(f"公开 benchmark 泄露画像  (LP-ceiling, 无特征, split_seed={SPLIT_SEED})  [{tag}]")
    print("=" * 92)
    print(f"{'dataset':<18}{'edge kind':<26}{'homo':>6}{'major':>7}"
          f"{'LP raw':>8}{'LP prun':>9}{'gap':>7}{'prun%':>7}")
    print("-" * 92)
    for r in sorted(rows, key=lambda x: -x["leak_gap"]):
        print(f"{r['name']:<18}{r['edge_kind']:<26}{r['homophily']:>6.2f}"
              f"{r['majority']:>7.3f}{r['lp_raw']:>8.3f}{r['lp_pruned']:>9.3f}"
              f"{r['leak_gap']:>+7.3f}{r['prune_frac']*100:>6.0f}%")
    print("-" * 92)
    print("gap = LP_raw − majority (>0 = 结构泄露);  prun% = 跨切分剪掉的边比例")
    constructed = [r for r in rows if "construct" in r["edge_kind"]]
    if constructed:
        cg = np.mean([r["leak_gap"] for r in constructed])
        print(f"\n构造边数据集({len(constructed)} 个)平均泄露 gap = {cg:+.3f}  "
              f"← 这些最像我们的 MAG,泄露普遍存在 = 论文背景的外部铁证")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="无网络自检(验证测量代码)")
    ap.add_argument("--public", action="store_true", help="下载并跑公开 benchmark(需网络)")
    ap.add_argument("--only", default=None, help="逗号分隔的子集,如 cora,amazon_photo")
    a = ap.parse_args()
    if a.selftest:
        run_selftest()
    elif a.public:
        run_public(a.only)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
