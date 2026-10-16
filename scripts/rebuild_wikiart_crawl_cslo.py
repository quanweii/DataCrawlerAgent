#!/usr/bin/env python3
"""
WikiArt 画作多模态属性图数据集构建
- 从 wikiart.org API 主动爬取画作数据
- 每幅画一个目录: wikiart_artworks/{artwork_id}/{artwork_id}.md, {artwork_id}.npy, images/*.jpg
- 同构图: artwork 节点 + co_attribute 边 (same_artist / shared_genre / shared_style)
- 多模态: text (标题+描述) + image (画作图片)
- 标签: style (印象派/文艺复兴/...)
"""
import os
import sys
import json
import csv
import time
import random
import re
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
RAW_DIR = BASE_DIR / "wikiart_raw"
ARTWORKS_DIR = BASE_DIR / "wikiart_artworks"
DATASETS_DIR = BASE_DIR / "wikiart_datasets"
GRAPHS_DIR = BASE_DIR / "wikiart_graphs"

GRAPH_JSON = DATASETS_DIR / "wikiart_multimodal_graph.json"
EDGES_CSV = GRAPHS_DIR / "co_attribute_edges.csv"

RAW_JSON = RAW_DIR / "wikiart_paintings_raw.json"
CHECKPOINT_JSON = RAW_DIR / "wikiart_checkpoint.json"

PROXY = "http://127.0.0.1:7890"
MAX_PAINTINGS = 25000      # 目标节点数
MAX_NODES_PER_ARTIST = 200
MAX_NODES_PER_GENRE = 300
MAX_NODES_PER_STYLE = 500
REQUEST_DELAY = 0.5
IMAGE_TIMEOUT = 30
MAX_RETRIES = 3


def get_session():
    s = requests.Session()
    s.proxies = {"http": PROXY, "https": PROXY}
    s.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"})
    return s


# ==================== Step 1: 从 WikiArt API 爬取 ====================

def get_artists(session):
    """获取画家列表"""
    logger.info("获取 WikiArt 画家列表...")
    artists = []

    # 方法1: 从字母索引页面获取
    for letter in "abcdefghijklmnopqrstuvwxyz":
        try:
            url = f"https://www.wikiart.org/en/alphabet/{letter}/text-list"
            resp = session.get(url, timeout=20, verify=False)
            if resp.status_code == 200:
                # 解析画家链接
                links = re.findall(r'href="(/en/[^"]+)"', resp.text)
                for link in links:
                    artist_url = link.split("/en/")[-1]
                    # 过滤非画家页面
                    if any(x in artist_url for x in ['painting', 'blog', 'about', 'terms', 'App']):
                        continue
                    artists.append(artist_url)
        except Exception as e:
            logger.debug(f"字母 {letter} 获取失败: {e}")
        time.sleep(0.3)

    # 去重
    artists = sorted(set(artists))
    logger.info(f"共获取 {len(artists)} 个画家")
    return artists


def get_artist_paintings(session, artist_url):
    """获取单个画家的所有画作"""
    try:
        resp = session.get(
            f"https://www.wikiart.org/en/App/Painting/PaintingsByArtist?artistUrl={artist_url}&json=2",
            timeout=20, verify=False
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.debug(f"获取 {artist_url} 画作失败: {e}")
    return []


def fix_image_url(url):
    """修复 WikiArt 图片 URL"""
    if not url:
        return ""
    # 去掉 !Large.jpg 等后缀
    url = re.sub(r'!.*\.jpg$', '', url)
    url = re.sub(r'!.*\.png$', '', url)
    # 确保有扩展名
    if not url.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
        url += '.jpg'
    return url


def load_or_crawl_raw(session):
    """加载或爬取原始画作数据"""
    if RAW_JSON.exists():
        with open(RAW_JSON, 'r', encoding='utf-8') as f:
            paintings = json.load(f)
        logger.info(f"加载已有数据: {len(paintings)} 幅画")
        return paintings

    artists = get_artists(session)
    logger.info(f"开始爬取画作, 目标 {MAX_PAINTINGS} 幅")

    # 加载 checkpoint
    done_artists = set()
    if CHECKPOINT_JSON.exists():
        with open(CHECKPOINT_JSON, 'r', encoding='utf-8') as f:
            done_artists = set(json.load(f).get("done", []))
        logger.info(f"Checkpoint: {len(done_artists)} 个画家已完成")

    paintings = []
    partial_path = RAW_DIR / "wikiart_paintings_partial.json"
    if partial_path.exists():
        with open(partial_path, 'r', encoding='utf-8') as f:
            paintings = json.load(f)
        logger.info(f"加载部分结果: {len(paintings)} 幅画")

    for artist_url in artists:
        if len(paintings) >= MAX_PAINTINGS:
            logger.info(f"达到目标 {MAX_PAINTINGS}, 停止")
            break
        if artist_url in done_artists:
            continue

        arts = get_artist_paintings(session, artist_url)
        if not arts:
            done_artists.add(artist_url)
            continue

        for art in arts:
            if not isinstance(art, dict):
                continue
            # 构建 painting 记录
            title = art.get("title", "").strip()
            if not title:
                continue

            image_url = art.get("image", "")
            image_url = fix_image_url(image_url)

            paintings.append({
                "painting_id": f"{artist_url}_{art.get('contentId', hash(title))}",
                "title": title,
                "artist": art.get("artistName", artist_url.replace("-", " ").title()),
                "artist_url": artist_url,
                "year": art.get("yearAsString", ""),
                "style": art.get("style", ""),
                "genre": art.get("genre", ""),
                "image_url": image_url,
                "width": art.get("width", 0),
                "height": art.get("height", 0),
            })

        done_artists.add(artist_url)
        logger.info(f"[{len(done_artists)}/{len(artists)}] {artist_url}: +{len(arts)} 画作 (总计 {len(paintings)})")

        # Checkpoint
        if len(done_artists) % 10 == 0:
            with open(CHECKPOINT_JSON, 'w') as f:
                json.dump({"done": list(done_artists)}, f)
            with open(partial_path, 'w', encoding='utf-8') as f:
                json.dump(paintings, f, ensure_ascii=False)

        time.sleep(REQUEST_DELAY)

    # 保存
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAW_JSON, 'w', encoding='utf-8') as f:
        json.dump(paintings, f, ensure_ascii=False)
    logger.info(f"原始数据已保存: {len(paintings)} 幅画")

    return paintings


# ==================== Step 2: 文本内容构建 ====================

def build_text_content(painting):
    parts = []
    title = painting.get("title", "").strip()
    if title:
        parts.append(f"# {title}")

    meta_lines = []
    artist = painting.get("artist", "")
    if artist:
        meta_lines.append(f"Artist: {artist}")
    style = painting.get("style", "")
    if style:
        meta_lines.append(f"Style: {style}")
    genre = painting.get("genre", "")
    if genre:
        meta_lines.append(f"Genre: {genre}")
    year = painting.get("year", "")
    if year:
        meta_lines.append(f"Year: {year}")

    if meta_lines:
        parts.append("\n".join(meta_lines))

    return "\n\n".join(parts) if parts else title or "Untitled"


# ==================== Step 3: 创建 cs_lo 目录 + 下载图片 ====================

def download_image(session, url, save_path):
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, timeout=IMAGE_TIMEOUT, verify=False,
                             allow_redirects=True)
            if resp.status_code == 200 and len(resp.content) > 500:
                img = Image.open(BytesIO(resp.content))
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                img.save(str(save_path), format='JPEG', quality=85)
                return True
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
    return False


def create_artwork_directories(paintings):
    ARTWORKS_DIR.mkdir(parents=True, exist_ok=True)
    session = get_session()

    node_attrs = []
    total_images = 0
    skipped = 0

    for idx, painting in enumerate(paintings):
        pid = painting["painting_id"]
        # 清理 ID 中的特殊字符
        pid = re.sub(r'[^\w\-.]', '_', pid)
        art_dir = ARTWORKS_DIR / pid

        md_path = art_dir / f"{pid}.md"
        if md_path.exists():
            text = md_path.read_text(encoding='utf-8')
            images_dir = art_dir / "images"
            img_count = len(list(images_dir.glob("*"))) if images_dir.exists() else 0
            node_attrs.append({
                "painting_id": pid,
                "markdown_length": len(text),
                "image_count": img_count,
                "markdown_size": md_path.stat().st_size,
            })
            total_images += img_count
            skipped += 1
            continue

        art_dir.mkdir(parents=True, exist_ok=True)

        # 写 .md
        text = build_text_content(painting)
        md_path.write_text(text, encoding='utf-8')

        # 写 .npy
        npy_path = art_dir / f"{pid}.npy"
        np.save(str(npy_path), np.array(text), allow_pickle=True)

        # 下载图片
        image_count = 0
        image_url = painting.get("image_url", "")
        if image_url:
            images_dir = art_dir / "images"
            images_dir.mkdir(exist_ok=True)
            jpg_path = images_dir / "0.jpg"
            if download_image(session, image_url, jpg_path):
                image_count += 1
                total_images += 1
            else:
                # 尝试带 !Large.jpg 后缀
                large_url = image_url.replace('.jpg', '!Large.jpg')
                if download_image(session, large_url, jpg_path):
                    image_count += 1
                    total_images += 1

        node_attrs.append({
            "painting_id": pid,
            "markdown_length": len(text),
            "image_count": image_count,
            "markdown_size": md_path.stat().st_size,
        })

        if (idx + 1) % 200 == 0 or idx == len(paintings) - 1:
            logger.info(f"[{idx+1}/{len(paintings)}] 图片总计: {total_images} (跳过: {skipped})")

        time.sleep(0.1)

    logger.info(f"cs_lo 目录完成: {len(node_attrs)} 画作, {total_images} 图片")
    return node_attrs


# ==================== Step 4: 构建同构图 ====================

def build_homogeneous_graph(paintings, node_attrs):
    logger.info("构建 WikiArt 画作同构图...")

    valid_ids = {na["painting_id"] for na in node_attrs}
    pid_to_painting = {}
    for p in paintings:
        pid = re.sub(r'[^\w\-.]', '_', p["painting_id"])
        pid_to_painting[pid] = p

    # 标签映射: style → label_index
    all_styles = set()
    for p in paintings:
        s = p.get("style", "")
        if s:
            all_styles.add(s)
    styles_sorted = sorted(all_styles)
    label_map = {s: i for i, s in enumerate(styles_sorted)}
    logger.info(f"标签: {len(label_map)} 个风格")

    # 构建节点
    nodes = []
    for na in node_attrs:
        pid = na["painting_id"]
        painting = pid_to_painting.get(pid, {})
        style = painting.get("style", "Unknown")
        nodes.append({
            "id": pid,
            "type": "artwork",
            "attributes": {
                "painting_id": pid,
                "title": painting.get("title", ""),
                "markdown_length": na["markdown_length"],
                "image_count": na["image_count"],
                "markdown_size": na["markdown_size"],
                "label": style,
                "label_index": label_map.get(style, -1),
            }
        })

    # ========== 边构建 ==========
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

    # --- same_artist 边 ---
    logger.info("构建 same_artist 边...")
    artist_to_ids = defaultdict(list)
    for na in node_attrs:
        painting = pid_to_painting.get(na["painting_id"], {})
        artist = painting.get("artist", "")
        if artist:
            artist_to_ids[artist].append(na["painting_id"])

    for artist, pids in artist_to_ids.items():
        if len(pids) < 2:
            continue
        if len(pids) > MAX_NODES_PER_ARTIST:
            sampled = random.Random(42).sample(pids, MAX_NODES_PER_ARTIST)
            for a, b in combinations(sampled, 2):
                add_edge(a, b, "same_artist")
            remaining = set(pids) - set(sampled)
            for rid in remaining:
                partner = random.Random(hash(rid)).choice(sampled)
                add_edge(rid, partner, "same_artist")
        else:
            for a, b in combinations(pids, 2):
                add_edge(a, b, "same_artist")

    logger.info(f"same_artist: {edge_type_counts['same_artist']}")

    # --- shared_genre 边 ---
    logger.info("构建 shared_genre 边...")
    genre_to_ids = defaultdict(list)
    for na in node_attrs:
        painting = pid_to_painting.get(na["painting_id"], {})
        genre = painting.get("genre", "")
        if genre:
            genre_to_ids[genre].append(na["painting_id"])

    for genre, pids in genre_to_ids.items():
        if len(pids) < 2:
            continue
        if len(pids) > MAX_NODES_PER_GENRE:
            sampled = random.Random(42).sample(pids, MAX_NODES_PER_GENRE)
            for a, b in combinations(sampled, 2):
                add_edge(a, b, "shared_genre")
            remaining = set(pids) - set(sampled)
            for rid in remaining:
                partner = random.Random(hash(rid)).choice(sampled)
                add_edge(rid, partner, "shared_genre")
        else:
            for a, b in combinations(pids, 2):
                add_edge(a, b, "shared_genre")

    logger.info(f"shared_genre: {edge_type_counts['shared_genre']}")

    # --- shared_style 边 (采样控制) ---
    logger.info("构建 shared_style 边...")
    style_to_ids = defaultdict(list)
    for na in node_attrs:
        painting = pid_to_painting.get(na["painting_id"], {})
        style = painting.get("style", "")
        if style:
            style_to_ids[style].append(na["painting_id"])

    for style, pids in style_to_ids.items():
        if len(pids) < 2:
            continue
        if len(pids) > MAX_NODES_PER_STYLE:
            sampled = random.Random(42).sample(pids, MAX_NODES_PER_STYLE)
            for a, b in combinations(sampled, 2):
                add_edge(a, b, "shared_style")
            remaining = set(pids) - set(sampled)
            for rid in remaining:
                partner = random.Random(hash(rid)).choice(sampled)
                add_edge(rid, partner, "shared_style")
        else:
            for a, b in combinations(pids, 2):
                add_edge(a, b, "shared_style")

    logger.info(f"shared_style: {edge_type_counts['shared_style']}")

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
            "name": "wikiart_multimodal_graph",
            "created_at": datetime.now().strftime("%Y-%m-%d"),
            "domain": "wikiart",
            "statistics": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "total_images": total_images,
                "artworks_with_images": nodes_with_images,
                "total_markdown_chars": total_md_chars,
                "avg_images_per_artwork": round(total_images / max(len(nodes), 1), 1),
                "avg_md_length": round(total_md_chars / max(len(nodes), 1), 0),
                "label_distribution": dict(label_dist),
                "num_labels": len(label_map),
                "top_labels": [{"label": l, "count": c} for l, c in top_labels[:15]],
            },
            "schema": {
                "node_types": ["artwork"],
                "edge_types": ["co_attribute"],
                "node_attributes": ["painting_id", "title", "markdown_length", "image_count",
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
    logger.info("WikiArt 画作多模态属性图构建 (从 wikiart.org 主动爬取)")
    logger.info("=" * 60)

    session = get_session()

    # 1. 爬取原始数据
    paintings = load_or_crawl_raw(session)
    logger.info(f"共 {len(paintings)} 幅画")

    # 2. 创建 cs_lo 目录 + 下载图片
    node_attrs = create_artwork_directories(paintings)

    # 3. 构建同构图
    nodes, edges, label_map = build_homogeneous_graph(paintings, node_attrs)

    # 4. 保存
    dataset = save_graph_json(nodes, edges, label_map)
    save_edges_csv(edges)

    # 5. 汇总
    meta = dataset["metadata"]["statistics"]
    logger.info("=" * 60)
    logger.info("✅ WikiArt 数据集构建完成")
    logger.info(f"   节点 (artwork): {meta['node_count']}")
    logger.info(f"   边 (co_attribute): {meta['edge_count']}")
    logger.info(f"   图片: {meta['total_images']} ({meta['artworks_with_images']} 画作)")
    logger.info(f"   标签: {meta['num_labels']} 个风格")
    logger.info(f"   目录: {ARTWORKS_DIR}")
    logger.info(f"   图: {GRAPH_JSON}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
