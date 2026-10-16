#!/usr/bin/env python3
"""
Steam 图重建 — 优化边密度策略 (不重新爬取，只重建图)
读取 steam_games_raw.json，用更合理的采样策略构建图

新策略:
- shared_genre:    采样上限 500→50; 只在稀有流派(<50游戏)全连接
- same_developer:  保持 (最有信息量), 采样上限 100
- same_publisher:  保持, 采样上限 100
- shared_category: 新增! 40+类别(Co-op/Singleplayer等), 采样上限 100

目标: 平均度数 30-50, 边数 5万-10万
"""
import json
import csv
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from datetime import datetime
from loguru import logger

BASE_DIR = Path("./data")
RAW_DIR = BASE_DIR / "steam_raw"
GAMES_DIR = BASE_DIR / "steam_games"
DATASETS_DIR = BASE_DIR / "steam_datasets"
GRAPHS_DIR = BASE_DIR / "steam_graphs"

GRAPH_JSON = DATASETS_DIR / "steam_multimodal_graph.json"
GRAPH_JSON_V2 = DATASETS_DIR / "steam_multimodal_graph_v2.json"
EDGES_CSV = GRAPHS_DIR / "co_attribute_edges_v2.csv"

# 边密度控制参数
MAX_NODES_PER_GENRE_FULL = 50   # 稀有流派(<此数)全连接
MAX_NODES_PER_GENRE_SAMPLE = 50 # 常见流派采样数
MAX_NODES_PER_DEV = 100
MAX_NODES_PER_PUB = 100
MAX_NODES_PER_CATEGORY = 100


def load_raw():
    with open(RAW_DIR / "steam_games_raw.json", encoding="utf-8") as f:
        games = json.load(f)
    logger.info(f"加载 {len(games)} 个游戏")
    return games


def get_node_attrs():
    """扫描已创建的 cs_lo 目录获取节点属性"""
    node_attrs = {}
    for game_dir in GAMES_DIR.iterdir():
        if not game_dir.is_dir():
            continue
        appid = game_dir.name
        md_path = game_dir / f"{appid}.md"
        if not md_path.exists():
            continue
        text = md_path.read_text(encoding="utf-8")
        images_dir = game_dir / "images"
        img_count = len(list(images_dir.glob("*"))) if images_dir.exists() else 0
        node_attrs[appid] = {
            "appid": appid,
            "markdown_length": len(text),
            "image_count": img_count,
            "markdown_size": md_path.stat().st_size,
        }
    logger.info(f"扫描到 {len(node_attrs)} 个有效节点")
    return node_attrs


def build_graph(games, node_attrs):
    """优化后的图构建"""
    logger.info("构建 Steam 同构图 (v2 优化密度)...")

    valid_ids = set(node_attrs.keys())
    id_to_game = {str(g["appid"]): g for g in games}

    # 标签映射: genre → label_index (用第一个 genre 作为标签)
    all_genres = set()
    for g in games:
        all_genres.update(g.get("genres", []))
    genres_sorted = sorted(all_genres)
    label_map = {g: i for i, g in enumerate(genres_sorted)}
    logger.info(f"标签: {len(label_map)} 个流派")

    # 构建节点
    nodes = []
    for appid, na in node_attrs.items():
        game = id_to_game.get(appid, {})
        game_genres = game.get("genres", [])
        label = game_genres[0] if game_genres else "Unknown"
        nodes.append({
            "id": appid,
            "type": "game",
            "attributes": {
                **na,
                "name": game.get("name", ""),
                "label": label,
                "label_index": label_map.get(label, -1),
            }
        })

    edges = []
    edge_set = set()
    edge_type_counts = defaultdict(int)

    def add_edge(id_a, id_b, relation):
        key = tuple(sorted([id_a, id_b]))
        if key not in edge_set and key[0] != key[1]:
            edge_set.add(key)
            edges.append({
                "source": key[0], "target": key[1],
                "type": "co_attribute",
                "attributes": {"relation": relation}
            })
            edge_type_counts[relation] += 1

    def build_attr_edges(attr_to_ids, attr_name, full_threshold, sample_cap):
        """通用属性边构建: 稀有(<full_threshold)全连接, 常见采样"""
        for attr_val, ids in attr_to_ids.items():
            ids = [i for i in ids if i in valid_ids]
            if len(ids) < 2:
                continue
            if len(ids) <= full_threshold:
                # 稀有: 全连接
                for a, b in combinations(ids, 2):
                    add_edge(a, b, attr_name)
            else:
                # 常见: 采样 sample_cap 个全连接, 其余每个连1个
                sampled = random.Random(42).sample(ids, min(sample_cap, len(ids)))
                for a, b in combinations(sampled, 2):
                    add_edge(a, b, attr_name)
                remaining = set(ids) - set(sampled)
                for rid in remaining:
                    partner = random.Random(hash(rid) % 2**32).choice(sampled)
                    add_edge(rid, partner, attr_name)

    # --- shared_genre ---
    logger.info("构建 shared_genre 边...")
    genre_to_ids = defaultdict(list)
    for appid in valid_ids:
        game = id_to_game.get(appid, {})
        for genre in game.get("genres", []):
            genre_to_ids[genre].append(appid)
    build_attr_edges(genre_to_ids, "shared_genre",
                     MAX_NODES_PER_GENRE_FULL, MAX_NODES_PER_GENRE_SAMPLE)
    logger.info(f"  shared_genre: {edge_type_counts['shared_genre']:,}")

    # --- same_developer ---
    logger.info("构建 same_developer 边...")
    dev_to_ids = defaultdict(list)
    for appid in valid_ids:
        game = id_to_game.get(appid, {})
        for dev in game.get("developers", []):
            if dev:
                dev_to_ids[dev].append(appid)
    build_attr_edges(dev_to_ids, "same_developer", 100, MAX_NODES_PER_DEV)
    logger.info(f"  same_developer: {edge_type_counts['same_developer']:,}")

    # --- same_publisher ---
    logger.info("构建 same_publisher 边...")
    pub_to_ids = defaultdict(list)
    for appid in valid_ids:
        game = id_to_game.get(appid, {})
        for pub in game.get("publishers", []):
            if pub:
                pub_to_ids[pub].append(appid)
    build_attr_edges(pub_to_ids, "same_publisher", 100, MAX_NODES_PER_PUB)
    logger.info(f"  same_publisher: {edge_type_counts['same_publisher']:,}")

    # --- shared_category (新增, 过滤高频无区分度类别) ---
    logger.info("构建 shared_category 边...")
    cat_to_ids = defaultdict(list)
    for appid in valid_ids:
        game = id_to_game.get(appid, {})
        for cat in game.get("categories", []):
            cat_to_ids[cat].append(appid)
    # 过滤: 出现频率 > 40% 的类别无区分度(如 Single-player/Family Sharing), 去掉
    n_total = len(valid_ids)
    RARE_THRESHOLD = 0.40 * n_total
    cat_filtered = {k: v for k, v in cat_to_ids.items()
                    if len(v) <= RARE_THRESHOLD and len(v) >= 5}
    logger.info(f"  类别过滤: {len(cat_to_ids)} → {len(cat_filtered)} (去掉 >40% 的高频类别)")
    build_attr_edges(cat_filtered, "shared_category", 30, 30)
    logger.info(f"  shared_category: {edge_type_counts['shared_category']:,}")

    logger.info(f"节点: {len(nodes)}, 边: {len(edges):,}")
    logger.info(f"边分布: {dict({k:f'{v:,}' for k,v in edge_type_counts.items()})}")
    logger.info(f"平均度数: {2*len(edges)/max(len(nodes),1):.1f}")

    # 同质性
    if edges:
        id_to_label = {n["id"]: n["attributes"]["label"] for n in nodes}
        same_label = sum(1 for e in edges
                        if id_to_label.get(e["source"]) == id_to_label.get(e["target"]))
        homophily = same_label / len(edges)
        logger.info(f"Edge homophily: {homophily:.4f}")

    return nodes, edges, label_map


def save(nodes, edges, label_map):
    total_images = sum(n["attributes"]["image_count"] for n in nodes)
    total_md = sum(n["attributes"]["markdown_length"] for n in nodes)
    nodes_with_images = sum(1 for n in nodes if n["attributes"]["image_count"] > 0)
    label_dist = defaultdict(int)
    for n in nodes:
        label_dist[n["attributes"]["label"]] += 1
    top_labels = sorted(label_dist.items(), key=lambda x: -x[1])

    dataset = {
        "nodes": nodes, "edges": edges,
        "metadata": {
            "name": "steam_multimodal_graph_v2",
            "created_at": datetime.now().strftime("%Y-%m-%d"),
            "domain": "steam",
            "statistics": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "total_images": total_images,
                "games_with_images": nodes_with_images,
                "total_markdown_chars": total_md,
                "avg_images_per_game": round(total_images / max(len(nodes), 1), 1),
                "avg_md_length": round(total_md / max(len(nodes), 1), 0),
                "avg_degree": round(2 * len(edges) / max(len(nodes), 1), 1),
                "label_distribution": dict(label_dist),
                "num_labels": len(label_map),
                "top_labels": [{"label": l, "count": c} for l, c in top_labels[:15]],
            },
            "schema": {
                "node_types": ["game"],
                "edge_types": ["co_attribute"],
                "node_attributes": ["appid", "name", "markdown_length", "image_count",
                                   "markdown_size", "label", "label_index"],
                "edge_attributes": ["relation"],
                "edge_relations": ["shared_genre", "same_developer", "same_publisher", "shared_category"],
                "modalities": ["text", "image"],
            }
        }
    }
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_JSON_V2, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    logger.info(f"图 JSON 已保存: {GRAPH_JSON_V2}")

    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EDGES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_id", "target_id", "edge_type", "relation"])
        for e in edges:
            writer.writerow([e["source"], e["target"], e["type"],
                           e["attributes"].get("relation", "")])
    logger.info(f"边 CSV 已保存: {EDGES_CSV}")


def main():
    logger.info("=" * 60)
    logger.info("Steam 图重建 (v2 优化边密度)")
    logger.info("=" * 60)
    games = load_raw()
    node_attrs = get_node_attrs()
    nodes, edges, label_map = build_graph(games, node_attrs)
    save(nodes, edges, label_map)
    logger.info("=" * 60)
    logger.info("✅ Steam 图重建完成")
    logger.info(f"   节点: {len(nodes)}, 边: {len(edges):,}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
