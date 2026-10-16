#!/usr/bin/env python3
"""
TMDb 图重建 — 优化边密度策略 (不重新爬取，只重建图)
读取 tmdb_movies_raw.json，用更合理的采样策略构建图

新策略:
- shared_genre:   采样上限 500→30; 过滤高频流派(Drama/Comedy >30%)避免标签泄漏
                  稀有流派(<30电影)全连接
- same_director:  保持 (最有信息量), 采样上限 30
- shared_actor:   保持 (前3演员), 采样上限 30

TMDb 演员导演分布健康(最高86部电影), 无极端团, 主要问题在 genre 采样上限过大
目标: 平均度数 10-30, 边数 5万-15万
"""
import json
import csv
import random
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from datetime import datetime
from loguru import logger

BASE_DIR = Path("./data")
RAW_DIR = BASE_DIR / "tmdb_raw"
MOVIES_DIR = BASE_DIR / "tmdb_movies"
DATASETS_DIR = BASE_DIR / "tmdb_datasets"
GRAPHS_DIR = BASE_DIR / "tmdb_graphs"

GRAPH_JSON_V2 = DATASETS_DIR / "tmdb_multimodal_graph_v2.json"
EDGES_CSV = GRAPHS_DIR / "co_attribute_edges_v2.csv"

# 边密度控制参数
GENRE_FREQ_FILTER = 0.30   # 出现频率 >30% 的流派作为边属性时过滤(避免标签泄漏)
MAX_GENRE_FULL = 30        # 稀有流派(<此数)全连接
MAX_GENRE_SAMPLE = 30      # 常见流派采样数
MAX_DIRECTOR = 30          # 同导演采样上限
MAX_ACTOR = 30             # 同演员采样上限


def load_raw():
    with open(RAW_DIR / "tmdb_movies_raw.json", encoding="utf-8") as f:
        movies = json.load(f)
    logger.info(f"加载 {len(movies)} 部电影")
    return movies


def get_node_attrs():
    """扫描已创建的 cs_lo 目录获取节点属性"""
    node_attrs = {}
    for movie_dir in MOVIES_DIR.iterdir():
        if not movie_dir.is_dir():
            continue
        mid = movie_dir.name
        md_path = movie_dir / f"{mid}.md"
        if not md_path.exists():
            continue
        text = md_path.read_text(encoding="utf-8")
        images_dir = movie_dir / "images"
        img_count = len(list(images_dir.glob("*"))) if images_dir.exists() else 0
        node_attrs[mid] = {
            "movie_id": mid,
            "markdown_length": len(text),
            "image_count": img_count,
            "markdown_size": md_path.stat().st_size,
        }
    logger.info(f"扫描到 {len(node_attrs)} 个有效节点")
    return node_attrs


def build_graph(movies, node_attrs):
    """优化后的图构建"""
    logger.info("构建 TMDb 同构图 (v2 优化密度)...")

    valid_ids = set(node_attrs.keys())
    mid_to_movie = {str(m["movie_id"]): m for m in movies}
    n_total = len(valid_ids)

    # 标签映射: genre → label_index (用第一个 genre 作为标签)
    all_genres = set()
    for m in movies:
        all_genres.update(m.get("genres", []))
    genres_sorted = sorted(all_genres)
    label_map = {g: i for i, g in enumerate(genres_sorted)}
    logger.info(f"标签: {len(label_map)} 个流派")

    # 构建节点
    nodes = []
    for mid, na in node_attrs.items():
        movie = mid_to_movie.get(mid, {})
        movie_genres = movie.get("genres", [])
        label = movie_genres[0] if movie_genres else "Unknown"
        nodes.append({
            "id": mid,
            "type": "movie",
            "attributes": {
                **na,
                "title": movie.get("title", ""),
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

    def build_attr_edges(attr_to_ids, attr_name, full_threshold, sample_cap,
                         freq_filter=None):
        """通用属性边构建: 可选过滤高频属性
        - freq_filter: 出现频率 > 此比例的属性过滤掉(不连边)
        - 稀有(<full_threshold)全连接, 常见采样 sample_cap 个全连接+其余各连1
        """
        filtered = 0
        for attr_val, ids in attr_to_ids.items():
            ids = [i for i in ids if i in valid_ids]
            if freq_filter is not None and len(ids) > freq_filter * n_total:
                filtered += 1
                continue
            if len(ids) < 2:
                continue
            if len(ids) <= full_threshold:
                for a, b in combinations(ids, 2):
                    add_edge(a, b, attr_name)
            else:
                sampled = random.Random(42).sample(ids, min(sample_cap, len(ids)))
                for a, b in combinations(sampled, 2):
                    add_edge(a, b, attr_name)
                remaining = set(ids) - set(sampled)
                for rid in remaining:
                    partner = random.Random(hash(rid) % 2**32).choice(sampled)
                    add_edge(rid, partner, attr_name)
        return filtered

    # --- shared_genre (过滤高频流派避免标签泄漏) ---
    logger.info("构建 shared_genre 边...")
    genre_to_ids = defaultdict(list)
    for mid in valid_ids:
        movie = mid_to_movie.get(mid, {})
        for genre in movie.get("genres", []):
            genre_to_ids[genre].append(mid)
    n_filtered = build_attr_edges(genre_to_ids, "shared_genre",
                                  MAX_GENRE_FULL, MAX_GENRE_SAMPLE,
                                  freq_filter=GENRE_FREQ_FILTER)
    logger.info(f"  shared_genre: {edge_type_counts['shared_genre']:,} (过滤 {n_filtered} 个高频流派)")

    # --- same_director ---
    logger.info("构建 same_director 边...")
    dir_to_ids = defaultdict(list)
    for mid in valid_ids:
        movie = mid_to_movie.get(mid, {})
        for d in movie.get("directors", []):
            if d:
                dir_to_ids[d].append(mid)
    build_attr_edges(dir_to_ids, "same_director", MAX_DIRECTOR, MAX_DIRECTOR)
    logger.info(f"  same_director: {edge_type_counts['same_director']:,}")

    # --- shared_actor (前3演员) ---
    logger.info("构建 shared_actor 边...")
    actor_to_ids = defaultdict(list)
    for mid in valid_ids:
        movie = mid_to_movie.get(mid, {})
        for a in movie.get("actors", [])[:3]:
            if a:
                actor_to_ids[a].append(mid)
    build_attr_edges(actor_to_ids, "shared_actor", MAX_ACTOR, MAX_ACTOR)
    logger.info(f"  shared_actor: {edge_type_counts['shared_actor']:,}")

    logger.info(f"节点: {len(nodes)}, 边: {len(edges):,}")
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
            "name": "tmdb_multimodal_graph_v2",
            "created_at": datetime.now().strftime("%Y-%m-%d"),
            "domain": "tmdb",
            "statistics": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "total_images": total_images,
                "movies_with_images": nodes_with_images,
                "total_markdown_chars": total_md,
                "avg_images_per_movie": round(total_images / max(len(nodes), 1), 1),
                "avg_md_length": round(total_md / max(len(nodes), 1), 0),
                "avg_degree": round(2 * len(edges) / max(len(nodes), 1), 1),
                "label_distribution": dict(label_dist),
                "num_labels": len(label_map),
                "top_labels": [{"label": l, "count": c} for l, c in top_labels[:15]],
            },
            "schema": {
                "node_types": ["movie"],
                "edge_types": ["co_attribute"],
                "node_attributes": ["movie_id", "title", "markdown_length", "image_count",
                                   "markdown_size", "label", "label_index"],
                "edge_attributes": ["relation"],
                "edge_relations": ["shared_genre", "same_director", "shared_actor"],
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
    logger.info("TMDb 图重建 (v2 优化边密度)")
    logger.info("=" * 60)
    movies = load_raw()
    node_attrs = get_node_attrs()
    nodes, edges, label_map = build_graph(movies, node_attrs)
    save(nodes, edges, label_map)
    logger.info("=" * 60)
    logger.info("✅ TMDb 图重建完成")
    logger.info(f"   节点: {len(nodes)}, 边: {len(edges):,}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
