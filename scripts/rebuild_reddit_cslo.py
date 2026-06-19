#!/usr/bin/env python3
"""
将 Reddit 数据集重建为 cs_lo 格式
- 每个帖子一个目录: reddit_posts/{post_id}/{post_id}.md, {post_id}.npy, images/*.png
- 同构图: post 节点 + co_comment 边 (同用户发帖关系)
- 仅保留 text + image 两个模态
- 标签: subreddit (10 类)

仿照 cs_lo_papers 的目录结构和 cs_lo_multimodal_graph.json 的图格式
"""
import os
import sys
import json
import csv
import time
import numpy as np
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

from loguru import logger
import requests
import urllib3
urllib3.disable_warnings()
from PIL import Image

# ==================== 路径配置 ====================
BASE_DIR = Path("./data")
RAW_DIR = BASE_DIR / "reddit_raw"
POSTS_DIR = BASE_DIR / "reddit_posts"
DATASETS_DIR = BASE_DIR / "reddit_datasets"
GRAPHS_DIR = BASE_DIR / "reddit_graphs"

GRAPH_JSON = DATASETS_DIR / "reddit_multimodal_graph.json"
EDGES_CSV = GRAPHS_DIR / "co_comment_edges.csv"

MAX_IMAGE_WORKERS = 8
IMAGE_TIMEOUT = 15


def load_raw_posts():
    """加载 Pushshift 预处理后的帖子数据"""
    path = RAW_DIR / "posts.json"
    if not path.exists() or path.stat().st_size <= 2:
        logger.error(f"帖子数据不存在: {path}")
        sys.exit(1)
    with open(path, 'r', encoding='utf-8') as f:
        posts = json.load(f)
    if isinstance(posts, dict):
        posts = list(posts.values())
    logger.info(f"加载 {len(posts)} 条帖子")
    return posts


def build_text_content(post):
    """构建帖子的 Markdown 文本 (title + selftext)"""
    parts = []
    title = post.get("title", "").strip()
    if title:
        parts.append(f"# {title}")
    selftext = post.get("selftext", "").strip()
    if selftext:
        parts.append(selftext)
    return "\n\n".join(parts) if parts else title or " "


def download_single_image(url, png_path):
    """下载单张图片并保存为 PNG"""
    try:
        resp = requests.get(url, timeout=IMAGE_TIMEOUT, verify=False,
                           allow_redirects=True, headers={
                               "User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"
                           })
        if resp.status_code == 200 and len(resp.content) > 100:
            img = Image.open(BytesIO(resp.content))
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            img.save(str(png_path), format='PNG')
            return True
    except Exception as e:
        logger.debug(f"下载失败 {url}: {type(e).__name__}")
    return False


def is_direct_image_url(url):
    """判断 URL 是否直接指向图片"""
    if not url or not isinstance(url, str):
        return False
    url_lower = url.lower()
    return any(ext in url_lower for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp'])


def create_post_directory(post, idx, total):
    """为单个帖子创建目录、写入文本、下载图片"""
    post_id = post["post_id"]
    post_dir = POSTS_DIR / post_id
    post_dir.mkdir(parents=True, exist_ok=True)

    # 写 .md 文件
    text = build_text_content(post)
    md_path = post_dir / f"{post_id}.md"
    md_path.write_text(text, encoding='utf-8')

    # 写 .npy 文件
    npy_path = post_dir / f"{post_id}.npy"
    np.save(str(npy_path), np.array(text), allow_pickle=True)

    # 收集图片 URL
    image_urls = []
    url = post.get("url", "")
    thumbnail = post.get("thumbnail", "")

    if is_direct_image_url(url):
        image_urls.append(url)
    if isinstance(thumbnail, str) and thumbnail.startswith("http") and is_direct_image_url(thumbnail):
        if thumbnail not in image_urls:
            image_urls.append(thumbnail)

    # 下载图片
    image_count = 0
    if image_urls:
        images_dir = post_dir / "images"
        images_dir.mkdir(exist_ok=True)
        for i, img_url in enumerate(image_urls):
            png_path = images_dir / f"{i}.png"
            if download_single_image(img_url, png_path):
                image_count += 1

    md_size = md_path.stat().st_size
    text_length = len(text)

    if idx % 500 == 0 or idx == total - 1:
        logger.info(f"[{idx+1}/{total}] {post_id}: text={text_length}chars, images={image_count}")

    return {
        "post_id": post_id,
        "markdown_length": text_length,
        "image_count": image_count,
        "markdown_size": md_size,
    }


def build_homogeneous_graph(posts, node_attrs):
    """构建同构图 (post 节点 + co_comment 边)"""
    logger.info("构建同构图...")

    # 构建 user → posts 映射 (从作者关系)
    user_to_posts = defaultdict(set)
    for post in posts:
        author = post.get("author", "")
        if author and author != "[deleted]":
            user_to_posts[author].add(post["post_id"])

    # 标签映射: subreddit → label index
    subreddit_set = sorted(set(p.get("subreddit", "") for p in posts if p.get("subreddit")))
    label_map = {s: i for i, s in enumerate(subreddit_set)}
    logger.info(f"标签: {len(label_map)} 个 subreddit: {label_map}")

    # 构建节点
    post_id_to_subreddit = {p["post_id"]: p.get("subreddit", "") for p in posts}
    valid_post_ids = {na["post_id"] for na in node_attrs}
    post_id_to_attrs = {na["post_id"]: na for na in node_attrs}

    nodes = []
    for post in posts:
        pid = post["post_id"]
        if pid not in valid_post_ids:
            continue
        na = post_id_to_attrs[pid]
        sub = post_id_to_subreddit.get(pid, "")
        nodes.append({
            "id": pid,
            "type": "post",
            "attributes": {
                "post_id": pid,
                "markdown_length": na["markdown_length"],
                "image_count": na["image_count"],
                "markdown_size": na["markdown_size"],
                "label": sub,
                "label_index": label_map.get(sub, -1),
            }
        })

    # 构建 co_comment 边 (同一用户在 >=2 个帖子中发帖 → 无向边)
    edges = []
    edge_set = set()
    for user, pids in user_to_posts.items():
        if len(pids) < 2:
            continue
        sorted_pids = sorted(pids)
        for p1, p2 in combinations(sorted_pids, 2):
            if p1 in valid_post_ids and p2 in valid_post_ids:
                key = (p1, p2)
                if key not in edge_set:
                    edge_set.add(key)
                    edges.append({
                        "source": p1,
                        "target": p2,
                        "type": "co_comment",
                        "attributes": {}
                    })

    logger.info(f"节点: {len(nodes)}, 边: {len(edges)} (来自 {sum(1 for p in user_to_posts.values() if len(p)>=2)} 个活跃用户)")

    return nodes, edges, label_map


def save_graph_json(nodes, edges, label_map):
    """保存图 JSON (仿 cs_lo 格式)"""
    total_images = sum(n["attributes"]["image_count"] for n in nodes)
    total_md_chars = sum(n["attributes"]["markdown_length"] for n in nodes)
    nodes_with_images = sum(1 for n in nodes if n["attributes"]["image_count"] > 0)
    avg_images = total_images / max(len(nodes), 1)
    avg_md = total_md_chars / max(len(nodes), 1)

    label_dist = defaultdict(int)
    for n in nodes:
        label_dist[n["attributes"]["label"]] += 1

    dataset = {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "name": "reddit_multimodal_graph",
            "created_at": datetime.now().strftime("%Y-%m-%d"),
            "domain": "reddit",
            "statistics": {
                "total_post_dirs": len(nodes),
                "node_count": len(nodes),
                "edge_count": len(edges),
                "total_images": total_images,
                "posts_with_images": nodes_with_images,
                "total_markdown_chars": total_md_chars,
                "avg_images_per_post": round(avg_images, 1),
                "avg_md_length": round(avg_md, 0),
                "label_distribution": dict(label_dist),
                "num_labels": len(label_map),
            },
            "schema": {
                "node_types": ["post"],
                "edge_types": ["co_comment"],
                "node_attributes": ["post_id", "markdown_length", "image_count", "markdown_size", "label", "label_index"],
                "edge_attributes": [],
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
    """保存边 CSV"""
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EDGES_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["source_id", "target_id", "edge_type"])
        for e in edges:
            writer.writerow([e["source"], e["target"], e["type"]])
    logger.info(f"边 CSV 已保存: {EDGES_CSV} ({len(edges)} 边)")


def main():
    logger.info("=" * 60)
    logger.info("Reddit 数据集重建 (cs_lo 格式)")
    logger.info("=" * 60)

    # 1. 加载原始数据
    posts = load_raw_posts()

    # 2. 为每个帖子创建目录 + 文本 + 图片
    logger.info(f"创建帖子目录: {POSTS_DIR}")
    POSTS_DIR.mkdir(parents=True, exist_ok=True)

    node_attrs = []
    for idx, post in enumerate(posts):
        try:
            na = create_post_directory(post, idx, len(posts))
            node_attrs.append(na)
        except Exception as e:
            logger.warning(f"帖子 {post.get('post_id', '?')} 创建失败: {e}")

    logger.info(f"成功创建 {len(node_attrs)} 个帖子目录")

    # 3. 构建同构图
    nodes, edges, label_map = build_homogeneous_graph(posts, node_attrs)

    # 4. 保存
    dataset = save_graph_json(nodes, edges, label_map)
    save_edges_csv(edges)

    # 5. 汇总
    meta = dataset["metadata"]["statistics"]
    logger.info("=" * 60)
    logger.info("✅ Reddit 数据集重建完成")
    logger.info(f"   节点 (post): {meta['node_count']}")
    logger.info(f"   边 (co_comment): {meta['edge_count']}")
    logger.info(f"   文本总量: {meta['total_markdown_chars']:,} chars (avg {meta['avg_md_length']:.0f})")
    logger.info(f"   图片: {meta['total_images']} ({meta['posts_with_images']} 帖子)")
    logger.info(f"   标签: {meta['num_labels']} 类")
    logger.info(f"   目录: {POSTS_DIR}")
    logger.info(f"   图: {GRAPH_JSON}")
    logger.info(f"   边: {EDGES_CSV}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
