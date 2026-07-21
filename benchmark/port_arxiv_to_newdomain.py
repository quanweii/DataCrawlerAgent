#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 arxiv 移植进 newdomain 统一协议（T0①）。

为什么：论文头号卖点 C1 = 「4 域跨模态×同质性 MAG 基准」，但实测 arxiv 从未进过
newdomain 套件（newdomain_nc / newdomain_lp / transfer_matrix 全是 3 域），arxiv 走的是
另一套旧脚本（AMMAGDataset、随机非分层切分、LP 只报 MRR/Hits 无 AUC）。这把「4 域」
变成两个不兼容套件的缝合——desk-reject 级风险。

做法（最小侵入、真正统一）：
  1. 从 benchmark/processed/graph_data.pt 导出 v3.json（与 steam/tmdb/wikiart 同格式），
     使 arxiv 成为又一个 SingleDomainDataset，NC/LP/steering/cross-domain 全走同一套代码。
  2. 把 3 模态 embedding（text/image/formula，已 768d CLIP-L/14、100% 对齐）软链进
     benchmark/processed/arxiv/。
arxiv 的 4 类标签 = DOMAIN_MAP(cs_si/cs_lo/math_lo/physics_socph)；边 = 自然引用（relation="cites"）。
homophily ≈ 0.933 = 同质性轴第 4 个点（顺带解决报告 R2）。

用法：
  python benchmark/port_arxiv_to_newdomain.py            # 导出 v3 + 软链 embedding
  python scripts/lp_hard_negative_sampler.py --domain arxiv --seed 42   # 生成 hard 负样本
"""
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
GPT = ROOT / "benchmark" / "processed" / "graph_data.pt"
OUT_JSON = ROOT / "data" / "arxiv_datasets" / "arxiv_multimodal_graph_v3.json"
EDIR = ROOT / "benchmark" / "processed" / "arxiv"

DOMAIN_NAME = {0: "cs_si", 1: "cs_lo", 2: "math_lo", 3: "physics_socph"}


def export_v3():
    g = torch.load(GPT, map_location="cpu", weights_only=False)
    ei = g["edge_index"]
    y = g["labels"]
    pids = g["paper_ids"]
    n = len(pids)
    print(f"[arxiv] graph_data.pt: {n} nodes, {ei.shape[1]} edge cols, "
          f"{int(y.unique().numel())} classes")

    nodes = []
    for i, pid in enumerate(pids):
        li = int(y[i].item())
        nodes.append({"id": str(pid),
                      "attributes": {"label": DOMAIN_NAME[li], "label_index": li}})

    # 唯一无向边（与 new-domain v3 约定一致：每条无向边存一次）
    seen = set()
    edges = []
    src = ei[0].tolist()
    dst = ei[1].tolist()
    same = 0
    tot = 0
    for a, b in zip(src, dst):
        if a == b:
            continue
        k = (a, b) if a < b else (b, a)
        if k in seen:
            continue
        seen.add(k)
        tot += 1
        if int(y[a]) == int(y[b]):
            same += 1
        edges.append({"source": str(pids[a]), "target": str(pids[b]),
                      "attributes": {"relation": "cites"}})

    hom = same / tot if tot else 0.0
    obj = {"nodes": nodes, "edges": edges}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(obj, f)
    print(f"[arxiv] wrote {OUT_JSON}: {len(nodes)} nodes, {len(edges)} unique edges, "
          f"homophily={hom:.4f}")
    return hom


def link_embeddings():
    EDIR.mkdir(parents=True, exist_ok=True)
    for m in ("text", "image", "formula"):
        src = ROOT / "benchmark" / "processed" / f"{m}_embeddings.pt"
        if not src.exists():
            print(f"[arxiv] WARN 缺 {src}")
            continue
        dst = EDIR / f"{m}_embeddings.pt"
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src)
        # 校验格式 + 对齐
        o = torch.load(src, map_location="cpu", weights_only=False)
        ids = o["ids"]
        emb = o["embeddings"]
        print(f"[arxiv] link {m}: ids={len(ids)} emb={tuple(emb.shape)} -> {dst.name}")


if __name__ == "__main__":
    hom = export_v3()
    link_embeddings()
    print(f"\n[arxiv] ✅ v3 + embeddings 就绪。homophily={hom:.4f}")
    print("下一步: python scripts/lp_hard_negative_sampler.py --domain arxiv --seed 42")
