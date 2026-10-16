#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出固定切分（T1⑤）—— D&B 可复现性硬性要求。

为 4 域导出 canonical、版本化的切分文件：
  NC: data/{domain}_splits/nc_{train,val,test}.csv  (node_id, label)
      stratified 60/20/20（seed=1，与 SingleDomainDataset.get_split_masks(seed=1) 一致）
  LP: 已由 scripts/lp_hard_negative_sampler.py 生成 data/{domain}_lp/*.csv（此处只登记）
  manifest.json: 汇总每域 节点/边/类/同质性/切分计数/许可/已知泄露

注意：论文头条 NC 报的是 5-seed mean±std（seed=1,43,85,127,169，切分+初始化都变）。
本固定切分(seed=1)是供他人复现/对比的 canonical 切分。
"""
import json
import csv
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys_path = PROJECT_ROOT
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.data.domain_loader import load_domain_graph, DOMAIN_GRAPH, DOMAIN_MODALITIES
from benchmark.data.splits import stratified_split

# 许可 / 来源（供 Dataset Card）
DOMAIN_META = {
    "steam":   {"source": "Steam Store (web-scraped, public storefront)", "license": "Steam Subscriber Agreement (research use)",
                "raw_dir": "data/steam_games", "note": "scraped; no PII"},
    "tmdb":    {"source": "TMDb (community movie DB)", "license": "TMDb CC-BY (attribution required)",
                "raw_dir": "data/tmdb_movies", "note": "TMDb attribution: 'This product uses the TMDB API but is not endorsed or certified by TMDB.'"},
    "wikiart": {"source": "WikiArt.org", "license": "WikiArt content各自版权（图像为展示用途，研究使用）",
                "raw_dir": "data/wikiart_artworks_v2", "note": "⚠ 已知标签泄露：same_artist/shared_style 边定义性同质，见 leakage_free_nc"},
    "arxiv":   {"source": "arXiv (CC0 public metadata) + papers", "license": "arXiv CC0 (metadata); 论文文本依各自 arXiv 许可",
                "raw_dir": "data/arxiv_datasets", "note": "4-class cross-domain (cs_si/cs_lo/math_lo/physics_socph); 自然引用边"},
}


def export_domain(domain):
    graph = load_domain_graph(domain)
    node_ids = graph["node_ids"]
    y = graph["y"]
    n = graph["num_nodes"]
    masks = stratified_split(y, ratios=(0.6, 0.2, 0.2), seed=1)
    names = ["train", "val", "test"]
    sdir = PROJECT_ROOT / "data" / f"{domain}_splits"
    sdir.mkdir(parents=True, exist_ok=True)
    counts = {}
    label_names = graph["label_names"]
    for name, mask in zip(names, masks):
        idx = mask.nonzero(as_tuple=False).squeeze(1).tolist()
        with open(sdir / f"nc_{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["node_id", "label_index", "label"])
            for i in idx:
                w.writerow([node_ids[i], int(y[i]), label_names.get(int(y[i]), str(int(y[i])))])
        counts[name] = len(idx)
    # 节点全表（id/label/label_index）
    with open(sdir / "nodes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "label_index", "label"])
        for i in range(n):
            w.writerow([node_ids[i], int(y[i]), label_names.get(int(y[i]), str(int(y[i])))])
    meta = DOMAIN_META[domain]
    entry = {
        "domain": domain,
        "graph_json": DOMAIN_GRAPH[domain],
        "modalities_available": DOMAIN_MODALITIES[domain],
        "num_nodes": n,
        "num_edges_unique": graph["edge_index"].size(1) // 2,
        "num_classes": len(label_names),
        "homophily": round(float(graph["homophily"]), 4),
        "relations": graph["relations"],
        "split": {"protocol": "stratified_per_class_60_20_20", "seed": 1,
                  "train": counts["train"], "val": counts["val"], "test": counts["test"]},
        "lp_splits_dir": f"data/{domain}_lp (由 lp_hard_negative_sampler.py 生成，含 random/hard/hard_degree 负样本)",
        "source": meta["source"], "license": meta["license"],
        "raw_dir": meta["raw_dir"], "note": meta["note"],
    }
    print(f"[{domain}] nodes={n} edges={entry['num_edges_unique']} classes={entry['num_classes']} "
          f"homophily={entry['homophily']} split={counts}")
    return entry


def main():
    out_dir = PROJECT_ROOT / "benchmark" / "results" / "dataset_card"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"benchmark": "AMMAG 4-domain MAG", "split_seed": 1,
                "frozen_split_protocol": "stratified_per_class_60_20_20 (seed=1); "
                "headline NC = 5-seed mean±std (seeds 1/43/85/127/169)",
                "lp_protocol": "OGB-style 80/10/10 edges; random/hard/hard_degree negatives",
                "domains": {}}
    for d in ["steam", "tmdb", "wikiart", "arxiv"]:
        manifest["domains"][d] = export_domain(d)
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n保存 manifest → {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
