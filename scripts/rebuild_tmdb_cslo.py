#!/usr/bin/env python3
"""
TMDb 电影多模态属性图数据集构建
- 从 TMDb API 主动爬取电影数据
- 每部电影一个目录: tmdb_movies/{movie_id}/{movie_id}.md, {movie_id}.npy, images/*.jpg
- 同构图: movie 节点 + co_attribute 边 (shared_actor / same_director / shared_genre)
- 多模态: text (剧情简介+演职人员) + image (海报)
- 标签: genre (动作/喜剧/剧情/...)
"""
import os
import sys
import json
import csv
import time
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from datetime import datetime
from io import BytesIO

from loguru import logger
import requests
import urllib3
urllib3.disable_warnings()
from PIL import Image

# ==================== 路径配置 ====================
BASE_DIR = Path("./data")
RAW_DIR = BASE_DIR / "tmdb_raw"
MOVIES_DIR = BASE_DIR / "tmdb_movies"
DATASETS_DIR = BASE_DIR / "tmdb_datasets"
GRAPHS_DIR = BASE_DIR / "tmdb_graphs"

GRAPH_JSON = DATASETS_DIR / "tmdb_multimodal_graph.json"
EDGES_CSV = GRAPHS_DIR / "co_attribute_edges.csv"

RAW_JSON = RAW_DIR / "tmdb_movies_raw.json"
CHECKPOINT_JSON = RAW_DIR / "tmdb_checkpoint.json"

PROXY = "http://127.0.0.1:7890"
TMDB_API_KEY = "b6df494b2e2c4763d5c43c73a6c457e5"
TMDB_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJiNmRmNDk0YjJlMmM0NzYzZDVjNDNjNzNhNmM0NTdlNSIsIm5iZiI6MTc4MTMyNjM5NC4zNTEsInN1YiI6IjYyMWNlMjNhYjgyYjcyZDVjNTlkM2U4NiIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.W9Xo29e63g-CU-dWCC7_nAOECa9tzoUh_YvZnsmyGnQ"
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"

MAX_MOVIES = 20000
MAX_NODES_PER_DIRECTOR = 200
MAX_NODES_PER_ACTOR = 200
MAX_NODES_PER_GENRE = 500
REQUEST_DELAY = 0.25
IMAGE_TIMEOUT = 20
MAX_RETRIES = 3


def get_session():
    s = requests.Session()
    s.proxies = {"http": PROXY, "https": PROXY}
    s.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Authorization": f"Bearer {TMDB_TOKEN}",
    })
    return s


# ==================== Step 1: 从 TMDb API 爬取 ====================

def discover_movie_ids(session):
    """通过 discover API 获取热门电影 ID 列表"""
    logger.info("发现 TMDb 电影 ID...")
    movie_ids = []

    for sort_by in ["popularity.desc", "vote_count.desc", "revenue.desc"]:
        for page in range(1, 501):
            try:
                url = f"{TMDB_BASE}/discover/movie?api_key={TMDB_API_KEY}&language=en-US&sort_by={sort_by}&page={page}&vote_count.gte=50"
                resp = session.get(url, timeout=15, verify=False)
                if resp.status_code != 200:
                    break
                data = resp.json()
                results = data.get("results", [])
                if not results:
                    break
                for m in results:
                    mid = m.get("id")
                    if mid and mid not in movie_ids:
                        movie_ids.append(mid)
            except Exception:
                break
            time.sleep(0.1)

        if len(movie_ids) >= MAX_MOVIES:
            break

    movie_ids = list(dict.fromkeys(movie_ids))
    logger.info(f"发现 {len(movie_ids)} 个电影 ID")
    return movie_ids[:MAX_MOVIES]


def get_movie_details(session, movie_id):
    """获取电影详情 + 演职人员"""
    try:
        resp = session.get(
            f"{TMDB_BASE}/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US",
            timeout=15, verify=False
        )
        if resp.status_code != 200:
            return None
        details = resp.json()

        resp2 = session.get(
            f"{TMDB_BASE}/movie/{movie_id}/credits?api_key={TMDB_API_KEY}",
            timeout=15, verify=False
        )
        credits = resp2.json() if resp2.status_code == 200 else {}

        return {**details, "_credits": credits}
    except Exception:
        return None


def load_or_crawl_raw(session):
    """加载或爬取原始电影数据"""
    if RAW_JSON.exists():
        with open(RAW_JSON, 'r', encoding='utf-8') as f:
            movies = json.load(f)
        logger.info(f"加载已有数据: {len(movies)} 部电影")
        return movies

    movie_ids = discover_movie_ids(session)
    logger.info(f"开始爬取 {len(movie_ids)} 部电影详情")

    done_ids = set()
    if CHECKPOINT_JSON.exists():
        with open(CHECKPOINT_JSON, 'r', encoding='utf-8') as f:
            done_ids = set(json.load(f).get("done", []))

    movies = []
    partial_path = RAW_DIR / "tmdb_movies_partial.json"
    if partial_path.exists():
        with open(partial_path, 'r', encoding='utf-8') as f:
            movies = json.load(f)
        logger.info(f"加载部分结果: {len(movies)} 部电影")

    for idx, mid in enumerate(movie_ids):
        if mid in done_ids:
            continue

        data = get_movie_details(session, mid)
        if not data:
            continue

        credits = data.get("_credits", {})
        cast = credits.get("cast", [])[:10]
        crew = credits.get("crew", [])
        directors = [c["name"] for c in crew if c.get("job") == "Director"]

        movie = {
            "movie_id": mid,
            "title": data.get("title", ""),
            "original_title": data.get("original_title", ""),
            "overview": data.get("overview", ""),
            "tagline": data.get("tagline", ""),
            "genres": [g["name"] for g in data.get("genres", [])],
            "release_date": data.get("release_date", ""),
            "vote_average": data.get("vote_average", 0),
            "vote_count": data.get("vote_count", 0),
            "popularity": data.get("popularity", 0),
            "runtime": data.get("runtime", 0),
            "poster_path": data.get("poster_path", ""),
            "backdrop_path": data.get("backdrop_path", ""),
            "production_companies": [c["name"] for c in data.get("production_companies", [])],
            "directors": directors,
            "actors": [a["name"] for a in cast],
        }

        if not movie["overview"] or not movie["genres"]:
            continue

        movies.append(movie)
        done_ids.add(mid)

        if len(movies) % 100 == 0:
            logger.info(f"[{len(movies)}/{MAX_MOVIES}] 最近: {movie['title'][:40]}")

        if len(movies) % 500 == 0:
            with open(CHECKPOINT_JSON, 'w') as f:
                json.dump({"done": list(done_ids)}, f)
            with open(partial_path, 'w', encoding='utf-8') as f:
                json.dump(movies, f, ensure_ascii=False)

        time.sleep(REQUEST_DELAY)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAW_JSON, 'w', encoding='utf-8') as f:
        json.dump(movies, f, ensure_ascii=False)
    logger.info(f"原始数据已保存: {len(movies)} 部电影")
    return movies


# ==================== Step 2: 文本内容构建 ====================

def build_text_content(movie):
    parts = []
    title = movie.get("title", "").strip()
    if title:
        parts.append(f"# {title}")

    meta_lines = []
    if movie.get("directors"):
        meta_lines.append(f"Director: {', '.join(movie['directors'])}")
    if movie.get("actors"):
        meta_lines.append(f"Cast: {', '.join(movie['actors'][:5])}")
    if movie.get("genres"):
        meta_lines.append(f"Genres: {', '.join(movie['genres'])}")
    if movie.get("release_date"):
        meta_lines.append(f"Release: {movie['release_date']}")
    if movie.get("runtime"):
        meta_lines.append(f"Runtime: {movie['runtime']} min")
    if movie.get("tagline"):
        meta_lines.append(f"Tagline: {movie['tagline']}")

    if meta_lines:
        parts.append("\n".join(meta_lines))

    overview = movie.get("overview", "").strip()
    if overview:
        parts.append(overview)

    return "\n\n".join(parts) if parts else title or "Untitled Movie"


# ==================== Step 3: 创建 cs_lo 目录 + 下载图片 ====================

def download_image(session, url, save_path):
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, timeout=IMAGE_TIMEOUT, verify=False)
            if resp.status_code == 200 and len(resp.content) > 500:
                img = Image.open(BytesIO(resp.content))
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                img.save(str(save_path), format='JPEG', quality=85)
                return True
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
    return False


def create_movie_directories(movies):
    MOVIES_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.proxies = {"http": PROXY, "https": PROXY}
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    node_attrs = []
    total_images = 0
    skipped = 0

    for idx, movie in enumerate(movies):
        mid = str(movie["movie_id"])
        movie_dir = MOVIES_DIR / mid

        md_path = movie_dir / f"{mid}.md"
        if md_path.exists():
            text = md_path.read_text(encoding='utf-8')
            images_dir = movie_dir / "images"
            img_count = len(list(images_dir.glob("*"))) if images_dir.exists() else 0
            node_attrs.append({
                "movie_id": mid,
                "markdown_length": len(text),
                "image_count": img_count,
                "markdown_size": md_path.stat().st_size,
            })
            total_images += img_count
            skipped += 1
            continue

        movie_dir.mkdir(parents=True, exist_ok=True)

        text = build_text_content(movie)
        md_path.write_text(text, encoding='utf-8')

        npy_path = movie_dir / f"{mid}.npy"
        np.save(str(npy_path), np.array(text), allow_pickle=True)

        image_count = 0
        poster_path = movie.get("poster_path", "")
        if poster_path:
            images_dir = movie_dir / "images"
            images_dir.mkdir(exist_ok=True)
            img_url = f"{TMDB_IMG_BASE}{poster_path}"
            if download_image(session, img_url, images_dir / "0.jpg"):
                image_count += 1
                total_images += 1

        node_attrs.append({
            "movie_id": mid,
            "markdown_length": len(text),
            "image_count": image_count,
            "markdown_size": md_path.stat().st_size,
        })

        if (idx + 1) % 200 == 0 or idx == len(movies) - 1:
            logger.info(f"[{idx+1}/{len(movies)}] 图片总计: {total_images} (跳过: {skipped})")

        time.sleep(0.05)

    logger.info(f"cs_lo 目录完成: {len(node_attrs)} 电影, {total_images} 图片")
    return node_attrs


# ==================== Step 4: 构建同构图 ====================

def build_homogeneous_graph(movies, node_attrs):
    logger.info("构建 TMDb 电影同构图...")

    valid_ids = {na["movie_id"] for na in node_attrs}
    mid_to_movie = {str(m["movie_id"]): m for m in movies}

    all_genres = set()
    for m in movies:
        all_genres.update(m.get("genres", []))
    genres_sorted = sorted(all_genres)
    label_map = {g: i for i, g in enumerate(genres_sorted)}
    logger.info(f"标签: {len(label_map)} 个流派")

    nodes = []
    for na in node_attrs:
        mid = na["movie_id"]
        movie = mid_to_movie.get(mid, {})
        movie_genres = movie.get("genres", [])
        label = movie_genres[0] if movie_genres else "Unknown"
        nodes.append({
            "id": mid,
            "type": "movie",
            "attributes": {
                "movie_id": mid,
                "title": movie.get("title", ""),
                "markdown_length": na["markdown_length"],
                "image_count": na["image_count"],
                "markdown_size": na["markdown_size"],
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
                "source": key[0],
                "target": key[1],
                "type": "co_attribute",
                "attributes": {"relation": relation}
            })
            edge_type_counts[relation] += 1

    # --- shared_genre 边 ---
    logger.info("构建 shared_genre 边...")
    genre_to_ids = defaultdict(list)
    for na in node_attrs:
        movie = mid_to_movie.get(na["movie_id"], {})
        for genre in movie.get("genres", []):
            genre_to_ids[genre].append(na["movie_id"])

    for genre, mids in genre_to_ids.items():
        if len(mids) < 2:
            continue
        if len(mids) > MAX_NODES_PER_GENRE:
            sampled = random.Random(42).sample(mids, MAX_NODES_PER_GENRE)
            for a, b in combinations(sampled, 2):
                add_edge(a, b, "shared_genre")
            remaining = set(mids) - set(sampled)
            for rid in remaining:
                partner = random.Random(hash(rid)).choice(sampled)
                add_edge(rid, partner, "shared_genre")
        else:
            for a, b in combinations(mids, 2):
                add_edge(a, b, "shared_genre")

    logger.info(f"shared_genre: {edge_type_counts['shared_genre']}")

    # --- same_director 边 ---
    logger.info("构建 same_director 边...")
    director_to_ids = defaultdict(list)
    for na in node_attrs:
        movie = mid_to_movie.get(na["movie_id"], {})
        for director in movie.get("directors", []):
            if director:
                director_to_ids[director].append(na["movie_id"])

    for director, mids in director_to_ids.items():
        if len(mids) < 2:
            continue
        if len(mids) > MAX_NODES_PER_DIRECTOR:
            sampled = random.Random(42).sample(mids, MAX_NODES_PER_DIRECTOR)
            for a, b in combinations(sampled, 2):
                add_edge(a, b, "same_director")
            remaining = set(mids) - set(sampled)
            for rid in remaining:
                partner = random.Random(hash(rid)).choice(sampled)
                add_edge(rid, partner, "same_director")
        else:
            for a, b in combinations(mids, 2):
                add_edge(a, b, "same_director")

    logger.info(f"same_director: {edge_type_counts['same_director']}")

    # --- shared_actor 边 ---
    logger.info("构建 shared_actor 边...")
    actor_to_ids = defaultdict(list)
    for na in node_attrs:
        movie = mid_to_movie.get(na["movie_id"], {})
        for actor in movie.get("actors", [])[:3]:
            if actor:
                actor_to_ids[actor].append(na["movie_id"])

    for actor, mids in actor_to_ids.items():
        if len(mids) < 2:
            continue
        if len(mids) > MAX_NODES_PER_ACTOR:
            sampled = random.Random(42).sample(mids, MAX_NODES_PER_ACTOR)
            for a, b in combinations(sampled, 2):
                add_edge(a, b, "shared_actor")
            remaining = set(mids) - set(sampled)
            for rid in remaining:
                partner = random.Random(hash(rid)).choice(sampled)
                add_edge(rid, partner, "shared_actor")
        else:
            for a, b in combinations(mids, 2):
                add_edge(a, b, "shared_actor")

    logger.info(f"shared_actor: {edge_type_counts['shared_actor']}")

    logger.info(f"节点: {len(nodes)}, 边: {len(edges)}")
    logger.info(f"边分布: {dict(edge_type_counts)}")

    if edges:
        id_to_label = {n["id"]: n["attributes"]["label"] for n in nodes}
        same_label = sum(1 for e in edges
                        if id_to_label.get(e["source"]) == id_to_label.get(e["target"]))
        homophily = same_label / len(edges)
        logger.info(f"Edge homophily: {homophily:.4f}")

    return nodes, edges, label_map


# ==================== Step 5: 保存 ====================

def save_graph_json(nodes, edges, label_map):
    total_images = sum(n["attributes"]["image_count"] for n in nodes)
    total_md_chars = sum(n["attributes"]["markdown_length"] for n in nodes)
    nodes_with_images = sum(1 for n in nodes if n["attributes"]["image_count"] > 0)

    label_dist = defaultdict(int)
    for n in nodes:
        label_dist[n["attributes"]["label"]] += 1
    top_labels = sorted(label_dist.items(), key=lambda x: -x[1])

    dataset = {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "name": "tmdb_multimodal_graph",
            "created_at": datetime.now().strftime("%Y-%m-%d"),
            "domain": "tmdb",
            "statistics": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "total_images": total_images,
                "movies_with_images": nodes_with_images,
                "total_markdown_chars": total_md_chars,
                "avg_images_per_movie": round(total_images / max(len(nodes), 1), 1),
                "avg_md_length": round(total_md_chars / max(len(nodes), 1), 0),
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
                "modalities": ["text", "image"],
            }
        }
    }

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_JSON, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    logger.info(f"图 JSON 已保存: {GRAPH_JSON}")
    return dataset


def save_edges_csv(edges):
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EDGES_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["source_id", "target_id", "edge_type", "relation"])
        for e in edges:
            writer.writerow([e["source"], e["target"], e["type"],
                           e["attributes"].get("relation", "")])
    logger.info(f"边 CSV 已保存: {EDGES_CSV} ({len(edges)} 边)")


# ==================== 主函数 ====================

def main():
    logger.info("=" * 60)
    logger.info("TMDb 电影多模态属性图构建 (从 TMDb API 主动爬取)")
    logger.info("=" * 60)

    session = get_session()

    # 1. 爬取原始数据
    movies = load_or_crawl_raw(session)
    logger.info(f"共 {len(movies)} 部电影")

    # 2. 创建 cs_lo 目录 + 下载图片
    node_attrs = create_movie_directories(movies)

    # 3. 构建同构图
    nodes, edges, label_map = build_homogeneous_graph(movies, node_attrs)

    # 4. 保存
    dataset = save_graph_json(nodes, edges, label_map)
    save_edges_csv(edges)

    # 5. 汇总
    meta = dataset["metadata"]["statistics"]
    logger.info("=" * 60)
    logger.info("✅ TMDb 数据集构建完成")
    logger.info(f"   节点 (movie): {meta['node_count']}")
    logger.info(f"   边 (co_attribute): {meta['edge_count']}")
    logger.info(f"   图片: {meta['total_images']} ({meta['movies_with_images']} 电影)")
    logger.info(f"   标签: {meta['num_labels']} 个流派")
    logger.info(f"   目录: {MOVIES_DIR}")
    logger.info(f"   图: {GRAPH_JSON}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
