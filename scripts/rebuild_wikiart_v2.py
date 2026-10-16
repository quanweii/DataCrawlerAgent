#!/usr/bin/env python3
"""
WikiArt v2 — 从风格页重新爬取 (修复方案A)
- 直接爬取 paintings-by-style 页面，每幅画天然带风格标签
- 包含所有大师作品(莫奈/梵高/雷诺阿等)
- 并发下载图片(比串行快10倍)
- 同构图: same_artist / shared_style / shared_genre
- 标签: style

之前的 artist-crawl 方案只拿到501个冷门艺术家，缺所有大师，故重爬
"""
import json
import csv
import re
import time
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from datetime import datetime
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
import requests
import urllib3
urllib3.disable_warnings()
from PIL import Image

# ==================== 路径配置 ====================
BASE_DIR = Path("./data")
RAW_DIR = BASE_DIR / "wikiart_raw"
ARTWORKS_DIR = BASE_DIR / "wikiart_artworks_v2"   # 新目录，避免覆盖旧的
DATASETS_DIR = BASE_DIR / "wikiart_datasets"
GRAPHS_DIR = BASE_DIR / "wikiart_graphs"

GRAPH_JSON = DATASETS_DIR / "wikiart_multimodal_graph_v2.json"
EDGES_CSV = GRAPHS_DIR / "co_attribute_edges_v2.csv"
RAW_JSON = RAW_DIR / "wikiart_v2_raw.json"
CHECKPOINT_JSON = RAW_DIR / "wikiart_v2_checkpoint.json"

PROXY = "http://127.0.0.1:7890"
MAX_PAINTINGS = 22000       # 目标节点数
PAGE_SIZE = 60
REQUEST_DELAY = 0.25

# 图密度参数
MAX_NODES_PER_ARTIST = 100
MAX_NODES_PER_STYLE = 50     # 较低: style 是标签,避免同质性泄漏
MAX_NODES_PER_GENRE = 100

# 图片并发
IMAGE_WORKERS = 12
IMAGE_TIMEOUT = 20


def get_session():
    s = requests.Session()
    s.proxies = {"http": PROXY, "https": PROXY}
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    return s


# ==================== Step 1: 从风格页爬取 ====================

def get_style_list(session):
    """获取所有艺术运动 slug (从 artists-by-art-movement 页,含主流运动)"""
    resp = session.get("https://www.wikiart.org/en/artists-by-art-movement", timeout=20, verify=False)
    links = re.findall(r'href="/en/artists-by-art-movement/([a-z0-9\-]+)"', resp.text)
    slugs = sorted(set(links))
    logger.info(f"共 {len(slugs)} 个艺术运动")
    return slugs


def fix_image_url(url):
    if not url:
        return ""
    url = re.sub(r'!.*\.jpg$', '', url)
    url = re.sub(r'!.*\.png$', '', url)
    return url


def load_or_crawl_raw(session):
    if RAW_JSON.exists():
        with open(RAW_JSON, encoding="utf-8") as f:
            paintings = json.load(f)
        logger.info(f"加载已有数据: {len(paintings)} 幅画")
        return paintings

    style_slugs = get_style_list(session)
    logger.info(f"开始爬取风格页, 目标 {MAX_PAINTINGS} 幅画")

    paintings = []
    seen_urls = set()  # paintingUrl 去重
    done_styles = set()

    if CHECKPOINT_JSON.exists():
        with open(CHECKPOINT_JSON) as f:
            cp = json.load(f)
            done_styles = set(cp.get("done_styles", []))
        partial = RAW_DIR / "wikiart_v2_partial.json"
        if partial.exists():
            with open(partial, encoding="utf-8") as f:
                paintings = json.load(f)
                seen_urls = set(p.get("painting_url") for p in paintings if p.get("painting_url"))
        logger.info(f"Checkpoint: {len(done_styles)} 风格已完成, {len(paintings)} 幅画")

    for si, style_slug in enumerate(style_slugs):
        if len(paintings) >= MAX_PAINTINGS:
            logger.info(f"达到目标 {MAX_PAINTINGS}")
            break
        if style_slug in done_styles:
            continue

        style_name = style_slug.replace('-', ' ').title()
        page = 1
        style_added = 0
        MAX_PER_STYLE = 500   # 每运动上限,平衡标签

        while True:
            try:
                url = f"https://www.wikiart.org/en/paintings-by-style/{style_slug}?json=2&page={page}"
                resp = session.get(url, timeout=20, verify=False)
                if resp.status_code != 200:
                    break
                data = resp.json()
                page_paintings = data.get("Paintings", [])
                total = data.get("AllPaintingsCount", 0)
                if not page_paintings:
                    break

                for p in page_paintings:
                    purl = p.get("paintingUrl", "")
                    if not purl or purl in seen_urls:
                        continue
                    title = p.get("title", "").strip()
                    if not title:
                        continue
                    seen_urls.add(purl)

                    # 解析 artist_url
                    m = re.match(r'/en/([^/]+)/', purl)
                    artist_url = m.group(1) if m else ""
                    artist_name = p.get("artistName", artist_url.replace('-', ' ').title())

                    paintings.append({
                        "painting_url": purl,
                        "painting_id": f"{artist_url}_{p.get('id', hash(title)%100000)}",
                        "title": title,
                        "artist": artist_name,
                        "artist_url": artist_url,
                        "year": str(p.get("year", "")),
                        "style": style_name,   # 天然带风格标签!
                        "style_slug": style_slug,
                        "genre": "",
                        "image_url": fix_image_url(p.get("image", "")),
                        "width": p.get("width", 0),
                        "height": p.get("height", 0),
                    })
                    style_added += 1
                    if len(paintings) >= MAX_PAINTINGS:
                        break

                if page * PAGE_SIZE >= total or len(page_paintings) < PAGE_SIZE:
                    break
                if len(paintings) >= MAX_PAINTINGS:
                    break
                if style_added >= MAX_PER_STYLE:   # 每风格达到上限,停止翻页
                    break
                page += 1
                time.sleep(REQUEST_DELAY)
            except Exception as e:
                logger.debug(f"  {style_slug} p={page}: {e}")
                break

        done_styles.add(style_slug)
        if style_added > 0:
            logger.info(f"[{si+1}/{len(style_slugs)}] {style_name}: +{style_added} (总 {len(paintings)})")

        # Checkpoint 每3个风格
        if (si + 1) % 3 == 0:
            with open(CHECKPOINT_JSON, 'w') as f:
                json.dump({"done_styles": list(done_styles)}, f)
            with open(RAW_DIR / "wikiart_v2_partial.json", 'w', encoding="utf-8") as f:
                json.dump(paintings, f, ensure_ascii=False)

        time.sleep(REQUEST_DELAY)

    with open(RAW_JSON, 'w', encoding="utf-8") as f:
        json.dump(paintings, f, ensure_ascii=False)
    logger.info(f"原始数据已保存: {len(paintings)} 幅画, {len(set(p['style'] for p in paintings))} 风格")
    return paintings


# ==================== Step 2: 文本构建 ====================

def build_text_content(p):
    parts = []
    title = p.get("title", "").strip()
    if title:
        parts.append(f"# {title}")
    meta = []
    if p.get("artist"):
        meta.append(f"Artist: {p['artist']}")
    if p.get("style"):
        meta.append(f"Style: {p['style']}")
    if p.get("genre"):
        meta.append(f"Genre: {p['genre']}")
    if p.get("year"):
        meta.append(f"Year: {p['year']}")
    if meta:
        parts.append("\n".join(meta))
    return "\n\n".join(parts) if parts else title or "Untitled"


# ==================== Step 3: 并发下载图片 + 创建目录 ====================

def download_one(painting, session):
    """下载单幅画图片, 创建目录, 返回属性"""
    pid = re.sub(r'[^\w\-.]', '_', painting["painting_id"])
    art_dir = ARTWORKS_DIR / pid
    md_path = art_dir / f"{pid}.md"

    if md_path.exists():
        text = md_path.read_text(encoding="utf-8")
        images_dir = art_dir / "images"
        img_count = len(list(images_dir.glob("*"))) if images_dir.exists() else 0
        return {"painting_id": pid, "markdown_length": len(text),
                "image_count": img_count, "markdown_size": md_path.stat().st_size}

    art_dir.mkdir(parents=True, exist_ok=True)
    text = build_text_content(painting)
    md_path.write_text(text, encoding="utf-8")
    np.save(str(art_dir / f"{pid}.npy"), np.array(text), allow_pickle=True)

    image_count = 0
    img_url = painting.get("image_url", "")
    if img_url:
        images_dir = art_dir / "images"
        images_dir.mkdir(exist_ok=True)
        jpg_path = images_dir / "0.jpg"
        for attempt in range(2):
            try:
                resp = session.get(img_url, timeout=IMAGE_TIMEOUT, verify=False, allow_redirects=True)
                if resp.status_code == 200 and len(resp.content) > 500:
                    img = Image.open(BytesIO(resp.content))
                    if img.mode not in ('RGB', 'L'):
                        img = img.convert('RGB')
                    img.save(str(jpg_path), format='JPEG', quality=85)
                    image_count = 1
                    break
            except Exception:
                time.sleep(0.5)
        # 尝试 !Large.jpg 后缀
        if image_count == 0:
            large_url = img_url.replace('.jpg', '!Large.jpg') if '.jpg' in img_url else img_url
            try:
                resp = session.get(large_url, timeout=IMAGE_TIMEOUT, verify=False)
                if resp.status_code == 200 and len(resp.content) > 500:
                    img = Image.open(BytesIO(resp.content))
                    if img.mode not in ('RGB', 'L'):
                        img = img.convert('RGB')
                    img.save(str(jpg_path), format='JPEG', quality=85)
                    image_count = 1
            except Exception:
                pass

    return {"painting_id": pid, "markdown_length": len(text),
            "image_count": image_count, "markdown_size": md_path.stat().st_size}


def create_directories_concurrent(paintings):
    ARTWORKS_DIR.mkdir(parents=True, exist_ok=True)

    # 创建 per-thread session
    import threading
    tls = threading.local()

    def worker(painting):
        if not hasattr(tls, 'session'):
            tls.session = get_session()
        return download_one(painting, tls.session)

    node_attrs = []
    total_images = 0
    done = 0
    with ThreadPoolExecutor(max_workers=IMAGE_WORKERS) as executor:
        futures = {executor.submit(worker, p): p for p in paintings}
        for future in as_completed(futures):
            try:
                na = future.result()
                node_attrs.append(na)
                total_images += na["image_count"]
            except Exception:
                pass
            done += 1
            if done % 500 == 0:
                logger.info(f"[{done}/{len(paintings)}] 图片总计: {total_images}")

    logger.info(f"cs_lo 完成: {len(node_attrs)} 画作, {total_images} 图片")
    return node_attrs


# ==================== Step 4: 构建图 ====================

def build_graph(paintings, node_attrs):
    logger.info("构建 WikiArt v2 同构图...")
    valid_ids = {na["painting_id"] for na in node_attrs}
    # 重新 slug 化 painting_id 以匹配
    pid_to_painting = {}
    for p in paintings:
        pid = re.sub(r'[^\w\-.]', '_', p["painting_id"])
        pid_to_painting[pid] = p

    # 标签映射: style → label_index
    styles = sorted(set(p.get("style", "") for p in paintings if p.get("style")))
    label_map = {s: i for i, s in enumerate(styles)}
    logger.info(f"标签: {len(label_map)} 个风格")

    nodes = []
    for na in node_attrs:
        pid = na["painting_id"]
        p = pid_to_painting.get(pid, {})
        style = p.get("style", "Unknown")
        nodes.append({
            "id": pid, "type": "artwork",
            "attributes": {**na, "title": p.get("title", ""),
                          "label": style, "label_index": label_map.get(style, -1)}
        })

    edges = []
    edge_set = set()
    edge_counts = defaultdict(int)

    def add_edge(a, b, rel):
        key = tuple(sorted([a, b]))
        if key not in edge_set and key[0] != key[1]:
            edge_set.add(key)
            edges.append({"source": key[0], "target": key[1], "type": "co_attribute",
                         "attributes": {"relation": rel}})
            edge_counts[rel] += 1

    def build_attr_edges(attr_to_ids, rel, cap):
        for val, ids in attr_to_ids.items():
            ids = [i for i in ids if i in valid_ids]
            if len(ids) < 2:
                continue
            if len(ids) <= cap:
                for a, b in combinations(ids, 2):
                    add_edge(a, b, rel)
            else:
                sampled = random.Random(42).sample(ids, cap)
                for a, b in combinations(sampled, 2):
                    add_edge(a, b, rel)
                for rid in set(ids) - set(sampled):
                    add_edge(rid, random.Random(hash(rid) % 2**32).choice(sampled), rel)

    # same_artist
    logger.info("same_artist 边...")
    a2i = defaultdict(list)
    for na in node_attrs:
        p = pid_to_painting.get(na["painting_id"], {})
        if p.get("artist"):
            a2i[p["artist"]].append(na["painting_id"])
    build_attr_edges(a2i, "same_artist", MAX_NODES_PER_ARTIST)
    logger.info(f"  {edge_counts['same_artist']:,}")

    # shared_style (注意: style 是标签, 但保留少量边增加连通性)
    logger.info("shared_style 边...")
    s2i = defaultdict(list)
    for na in node_attrs:
        p = pid_to_painting.get(na["painting_id"], {})
        if p.get("style"):
            s2i[p["style"]].append(na["painting_id"])
    build_attr_edges(s2i, "shared_style", MAX_NODES_PER_STYLE)
    logger.info(f"  {edge_counts['shared_style']:,}")

    # shared_genre
    logger.info("shared_genre 边...")
    g2i = defaultdict(list)
    for na in node_attrs:
        p = pid_to_painting.get(na["painting_id"], {})
        if p.get("genre"):
            g2i[p["genre"]].append(na["painting_id"])
    build_attr_edges(g2i, "shared_genre", MAX_NODES_PER_GENRE)
    logger.info(f"  {edge_counts['shared_genre']:,}")

    logger.info(f"节点: {len(nodes)}, 边: {len(edges):,}, 平均度: {2*len(edges)/max(len(nodes),1):.1f}")

    if edges:
        id2l = {n["id"]: n["attributes"]["label"] for n in nodes}
        same = sum(1 for e in edges if id2l.get(e["source"]) == id2l.get(e["target"]))
        logger.info(f"Edge homophily: {same/len(edges):.4f}")

    return nodes, edges, label_map


def save(nodes, edges, label_map):
    total_img = sum(n["attributes"]["image_count"] for n in nodes)
    total_md = sum(n["attributes"]["markdown_length"] for n in nodes)
    nwi = sum(1 for n in nodes if n["attributes"]["image_count"] > 0)
    ld = defaultdict(int)
    for n in nodes:
        ld[n["attributes"]["label"]] += 1
    top = sorted(ld.items(), key=lambda x: -x[1])

    dataset = {"nodes": nodes, "edges": edges,
        "metadata": {"name": "wikiart_multimodal_graph_v2",
            "created_at": datetime.now().strftime("%Y-%m-%d"), "domain": "wikiart",
            "statistics": {"node_count": len(nodes), "edge_count": len(edges),
                "total_images": total_img, "artworks_with_images": nwi,
                "total_markdown_chars": total_md,
                "avg_images_per_artwork": round(total_img/max(len(nodes),1),1),
                "avg_md_length": round(total_md/max(len(nodes),1),0),
                "avg_degree": round(2*len(edges)/max(len(nodes),1),1),
                "label_distribution": dict(ld), "num_labels": len(label_map),
                "top_labels": [{"label": l, "count": c} for l, c in top[:15]]},
            "schema": {"node_types": ["artwork"], "edge_types": ["co_attribute"],
                "node_attributes": ["painting_id","title","markdown_length","image_count",
                                   "markdown_size","label","label_index"],
                "edge_relations": ["same_artist","shared_style","shared_genre"],
                "modalities": ["text","image"]}}}
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_JSON, 'w', encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    logger.info(f"图 JSON: {GRAPH_JSON}")

    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EDGES_CSV, 'w', newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_id","target_id","edge_type","relation"])
        for e in edges:
            w.writerow([e["source"], e["target"], e["type"], e["attributes"].get("relation","")])
    logger.info(f"边 CSV: {EDGES_CSV}")


def main():
    logger.info("=" * 60)
    logger.info("WikiArt v2 (风格页爬取 + 并发下载)")
    logger.info("=" * 60)
    session = get_session()
    paintings = load_or_crawl_raw(session)
    logger.info(f"共 {len(paintings)} 幅画")
    node_attrs = create_directories_concurrent(paintings)
    nodes, edges, label_map = build_graph(paintings, node_attrs)
    save(nodes, edges, label_map)
    m = None
    with open(GRAPH_JSON) as f:
        m = json.load(f)["metadata"]["statistics"]
    logger.info("=" * 60)
    logger.info("✅ WikiArt v2 完成")
    logger.info(f"   节点: {m['node_count']}, 边: {m['edge_count']:,}")
    logger.info(f"   图片: {m['total_images']}, 标签: {m['num_labels']} 风格")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
