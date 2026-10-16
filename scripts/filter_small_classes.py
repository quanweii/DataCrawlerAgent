#!/usr/bin/env python3
"""
统一小类过滤 — 删除 <20 样本的标签类，重建图
对 Steam / TMDb 适用 (WikiArt 在 enrich 脚本里已过滤)
读取 *_v2.json, 过滤节点+边, 输出 *_v3.json
"""
import json
import csv
import random
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations
from datetime import datetime
from loguru import logger

MIN_CLASS_SIZE = 20

CONFIGS = {
    "steam": {
        "in_graph": "data/steam_datasets/steam_multimodal_graph_v2.json",
        "out_graph": "data/steam_datasets/steam_multimodal_graph_v3.json",
        "raw_json": "data/steam_raw/steam_games_raw.json",
        "artworks_dir": "data/steam_games",
        "id_key": "appid",
        "out_edges": "data/steam_graphs/co_attribute_edges_v3.csv",
    },
    "tmdb": {
        "in_graph": "data/tmdb_datasets/tmdb_multimodal_graph_v2.json",
        "out_graph": "data/tmdb_datasets/tmdb_multimodal_graph_v3.json",
        "raw_json": "data/tmdb_raw/tmdb_movies_raw.json",
        "artworks_dir": "data/tmdb_movies",
        "id_key": "movie_id",
        "out_edges": "data/tmdb_graphs/co_attribute_edges_v3.csv",
    },
}


def filter_domain(domain):
    cfg = CONFIGS[domain]
    logger.info(f"\n{'='*60}\n【{domain}】小类过滤\n{'='*60}")

    with open(cfg["in_graph"]) as f:
        g = json.load(f)
    nodes = g["nodes"]
    edges = g["edges"]
    logger.info(f"原始: {len(nodes)} 节点, {len(edges):,} 边")

    # 统计标签
    label_count = Counter(n["attributes"]["label"] for n in nodes)
    small = {l for l, c in label_count.items() if c < MIN_CLASS_SIZE}
    logger.info(f"标签分布: {len(label_count)} 类")
    logger.info(f"过滤 <{MIN_CLASS_SIZE} 的小类: {sorted(small)}")

    # 过滤节点
    keep_ids = {n["id"] for n in nodes if n["attributes"]["label"] not in small}
    new_nodes = [n for n in nodes if n["id"] in keep_ids]
    logger.info(f"过滤后节点: {len(new_nodes)}")

    # 重新编号 label_index
    new_labels = sorted(set(n["attributes"]["label"] for n in new_nodes))
    label_map = {l: i for i, l in enumerate(new_labels)}
    for n in new_nodes:
        n["attributes"]["label_index"] = label_map[n["attributes"]["label"]]

    # 过滤边 (两端都在)
    new_edges = [e for e in edges if e["source"] in keep_ids and e["target"] in keep_ids]
    logger.info(f"过滤后边: {len(new_edges):,}")

    # 重新统计
    rel_c = Counter(e["attributes"].get("relation","?") for e in new_edges)
    logger.info(f"边类型: {dict(rel_c)}")
    avg_deg = 2 * len(new_edges) / max(len(new_nodes), 1)
    logger.info(f"平均度: {avg_deg:.1f}")

    # 同质性
    id2l = {n["id"]: n["attributes"]["label"] for n in new_nodes}
    same = sum(1 for e in new_edges if id2l.get(e["source"]) == id2l.get(e["target"]))
    homophily = same / max(len(new_edges), 1)
    logger.info(f"同质性: {homophily:.3f}")

    # 标签分布
    ld = Counter(n["attributes"]["label"] for n in new_nodes)
    logger.info(f"新标签分布 ({len(ld)}类): {ld.most_common()}")

    # 保存
    total_img = sum(n["attributes"]["image_count"] for n in new_nodes)
    total_md = sum(n["attributes"]["markdown_length"] for n in new_nodes)
    dataset = {
        "nodes": new_nodes, "edges": new_edges,
        "metadata": {
            "name": f"{domain}_multimodal_graph_v3",
            "created_at": datetime.now().strftime("%Y-%m-%d"),
            "domain": domain,
            "statistics": {
                "node_count": len(new_nodes), "edge_count": len(new_edges),
                "total_images": total_img,
                "nodes_with_images": sum(1 for n in new_nodes if n["attributes"]["image_count"]>0),
                "total_markdown_chars": total_md,
                "avg_degree": round(avg_deg, 1),
                "homophily": round(homophily, 3),
                "label_distribution": dict(ld), "num_labels": len(ld),
            },
            "schema": g["metadata"]["schema"],
        }
    }
    with open(cfg["out_graph"], "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    logger.info(f"已保存: {cfg['out_graph']}")

    # 边CSV
    with open(cfg["out_edges"], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_id","target_id","edge_type","relation"])
        for e in new_edges:
            w.writerow([e["source"], e["target"], e["type"], e["attributes"].get("relation","")])

    return len(new_nodes), len(new_edges), homophily


def main():
    for domain in ["steam", "tmdb"]:
        filter_domain(domain)
    logger.info("\n✅ 小类过滤完成")


if __name__ == "__main__":
    main()
