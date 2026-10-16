#!/usr/bin/env python3
"""
补下 Reddit 图片 —— 通过 mihomo 代理下载 i.redd.it / i.imgur.com 图片
仅下载图片，不重建文本/图结构
"""
import json
import time
import sys
from pathlib import Path
from io import BytesIO
from collections import defaultdict

import numpy as np
import requests
import urllib3
urllib3.disable_warnings()
from PIL import Image
from loguru import logger

# ==================== 配置 ====================
BASE_DIR = Path("./data")
RAW_DIR = BASE_DIR / "reddit_raw"
POSTS_DIR = BASE_DIR / "reddit_posts"
DATASETS_DIR = BASE_DIR / "reddit_datasets"

PROXY = "http://127.0.0.1:7890"
IMAGE_TIMEOUT = 20
MAX_RETRIES = 3


def is_direct_image_url(url):
    """判断 URL 是否直接指向图片"""
    if not url or not isinstance(url, str):
        return False
    url_lower = url.lower()
    return any(ext in url_lower for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp'])


def download_image(session, url, png_path):
    """下载单张图片并保存为 PNG，通过代理"""
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, timeout=IMAGE_TIMEOUT, verify=False,
                               allow_redirects=True)
            if resp.status_code == 200 and len(resp.content) > 100:
                img = Image.open(BytesIO(resp.content))
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                img.save(str(png_path), format='PNG')
                return True
            else:
                logger.debug(f"  失败 {url}: status={resp.status_code}, size={len(resp.content)}")
        except Exception as e:
            logger.debug(f"  异常 {url} (attempt {attempt+1}): {type(e).__name__}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
    return False


def main():
    logger.info("=" * 60)
    logger.info("Reddit 图片补下 (通过 mihomo 代理)")
    logger.info("=" * 60)

    # 1. 加载原始帖子
    posts_path = RAW_DIR / "posts.json"
    with open(posts_path, 'r', encoding='utf-8') as f:
        posts = json.load(f)

    # 2. 筛选有图片 URL 的帖子
    image_posts = []
    for p in posts:
        pid = p["post_id"]
        url = p.get("url", "")
        thumbnail = p.get("thumbnail", "")

        urls = []
        if is_direct_image_url(url):
            urls.append(url)
        if (isinstance(thumbnail, str) and thumbnail.startswith("http")
                and is_direct_image_url(thumbnail) and thumbnail not in urls):
            urls.append(thumbnail)

        if urls:
            image_posts.append((pid, urls))

    logger.info(f"共 {len(posts)} 帖子, {len(image_posts)} 个有图片 URL")

    # 3. 检查已有图片（跳过已下载的）
    already_done = 0
    todo = []
    for pid, urls in image_posts:
        images_dir = POSTS_DIR / pid / "images"
        if images_dir.exists():
            existing = list(images_dir.glob("*.png"))
            if existing:
                already_done += 1
                continue
        todo.append((pid, urls))

    logger.info(f"已有图片: {already_done}, 待下载: {len(todo)}")

    if not todo:
        logger.info("无需下载")
        return

    # 4. 使用代理 Session 下载
    session = requests.Session()
    session.proxies = {"http": PROXY, "https": PROXY}
    session.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"})

    success = 0
    fail = 0
    total_urls = sum(len(urls) for _, urls in todo)

    for idx, (pid, urls) in enumerate(todo):
        post_dir = POSTS_DIR / pid
        images_dir = post_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        for i, url in enumerate(urls):
            png_path = images_dir / f"{i}.png"
            if download_image(session, url, png_path):
                success += 1
            else:
                fail += 1

        if (idx + 1) % 20 == 0 or idx == len(todo) - 1:
            logger.info(f"[{idx+1}/{len(todo)}] 成功: {success}, 失败: {fail}")

    logger.info(f"下载完成: 成功 {success}, 失败 {fail}")

    # 5. 更新 graph JSON 中的 image_count 和 image 统计
    logger.info("更新 graph JSON...")
    graph_path = DATASETS_DIR / "reddit_multimodal_graph.json"
    with open(graph_path, 'r', encoding='utf-8') as f:
        graph = json.load(f)

    # 重新统计每个节点的图片数
    updated = 0
    for node in graph["nodes"]:
        pid = node["id"]
        images_dir = POSTS_DIR / pid / "images"
        count = len(list(images_dir.glob("*.png"))) if images_dir.exists() else 0
        if count != node["attributes"]["image_count"]:
            node["attributes"]["image_count"] = count
            updated += 1

    # 更新 metadata 统计
    total_images = sum(n["attributes"]["image_count"] for n in graph["nodes"])
    posts_with_images = sum(1 for n in graph["nodes"] if n["attributes"]["image_count"] > 0)
    graph["metadata"]["statistics"]["total_images"] = total_images
    graph["metadata"]["statistics"]["posts_with_images"] = posts_with_images
    graph["metadata"]["statistics"]["avg_images_per_post"] = round(
        total_images / max(len(graph["nodes"]), 1), 1
    )

    with open(graph_path, 'w', encoding='utf-8') as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    logger.info(f"更新了 {updated} 个节点的 image_count")
    logger.info(f"总计: {total_images} 张图片 ({posts_with_images} 个帖子有图片)")

    logger.info("=" * 60)
    logger.info("✅ 完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
