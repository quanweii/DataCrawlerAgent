#!/usr/bin/env python3
"""
将 Amazon 数据集重建为 cs_lo 格式
- 每个产品一个目录: amazon_products/{asin}/{asin}.md, {asin}.npy, images/*.png
- 同构图: product 节点 + co_purchase 边 (also_buy + also_view)
- 仅保留 text + image 两个模态
- 标签: leaf_category (叶品类分类)

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
from datetime import datetime
from io import BytesIO

from loguru import logger
import requests
import urllib3
urllib3.disable_warnings()
from PIL import Image

# ==================== 路径配置 ====================
BASE_DIR = Path("./data")
RAW_DIR = BASE_DIR / "amazon_raw"
PRODUCTS_DIR = BASE_DIR / "amazon_products"
DATASETS_DIR = BASE_DIR / "amazon_datasets"
GRAPHS_DIR = BASE_DIR / "amazon_graphs"

GRAPH_JSON = DATASETS_DIR / "amazon_multimodal_graph.json"
EDGES_CSV = GRAPHS_DIR / "co_purchase_edges.csv"

MAX_IMAGE_WORKERS = 8
IMAGE_TIMEOUT = 15


def load_raw_products():
    """加载产品元数据"""
    path = RAW_DIR / "products.json"
    if not path.exists() or path.stat().st_size <= 2:
        logger.error(f"产品数据不存在: {path}")
        sys.exit(1)
    with open(path, 'r', encoding='utf-8') as f:
        products = json.load(f)
    if isinstance(products, dict):
        products = list(products.values())
    logger.info(f"加载 {len(products)} 个产品")
    return products


def build_text_content(product):
    """构建产品的 Markdown 文本 (title + description)"""
    parts = []
    title = product.get("title", "").strip()
    if title:
        parts.append(f"# {title}")
    desc = product.get("description", "").strip()
    if desc:
        parts.append(desc)
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
        else:
            logger.debug(f"下载失败 {url}: status={resp.status_code}, size={len(resp.content)}")
    except Exception as e:
        logger.debug(f"下载异常 {url}: {type(e).__name__}: {e}")
    return False


def collect_image_urls(product):
    """收集产品的所有图片 URL（去重，优先 highres）"""
    urls = []
    seen = set()

    # 优先 highres
    for url in product.get("image_urls_highres", []):
        if isinstance(url, str) and url not in seen:
            urls.append(url)
            seen.add(url)

    # 补充 image_url
    iu = product.get("image_url", "")
    if isinstance(iu, str) and iu and iu not in seen:
        urls.append(iu)
        seen.add(iu)

    return urls


def create_product_directory(product, idx, total):
    """为单个产品创建目录、写入文本、下载图片"""
    asin = product["asin"]
    prod_dir = PRODUCTS_DIR / asin
    prod_dir.mkdir(parents=True, exist_ok=True)

    # 写 .md 文件
    text = build_text_content(product)
    md_path = prod_dir / f"{asin}.md"
    md_path.write_text(text, encoding='utf-8')

    # 写 .npy 文件
    npy_path = prod_dir / f"{asin}.npy"
    np.save(str(npy_path), np.array(text), allow_pickle=True)

    # 收集并下载图片
    image_urls = collect_image_urls(product)
    image_count = 0

    if image_urls:
        images_dir = prod_dir / "images"
        images_dir.mkdir(exist_ok=True)
        for i, img_url in enumerate(image_urls):
            png_path = images_dir / f"{i}.png"
            if download_single_image((img_url, png_path)):
                image_count += 1

    md_size = md_path.stat().st_size
    text_length = len(text)

    if idx % 500 == 0 or idx == total - 1:
        logger.info(f"[{idx+1}/{total}] {asin}: text={text_length}chars, images={image_count}")

    return {
        "asin": asin,
        "markdown_length": text_length,
        "image_count": image_count,
        "markdown_size": md_size,
    }


def create_products_sequential(products):
    """串行创建产品目录 + 下载图片（使用 requests.Session 更可靠）"""
    PRODUCTS_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"})

    node_attrs = []
    total_images = 0
    total_products_with_images = 0

    for idx, product in enumerate(products):
        asin = product["asin"]
        prod_dir = PRODUCTS_DIR / asin
        prod_dir.mkdir(parents=True, exist_ok=True)

        # 写 .md 文件
        text = build_text_content(product)
        md_path = prod_dir / f"{asin}.md"
        md_path.write_text(text, encoding='utf-8')

        # 写 .npy 文件
        npy_path = prod_dir / f"{asin}.npy"
        np.save(str(npy_path), np.array(text), allow_pickle=True)

        # 下载图片
        image_urls = collect_image_urls(product)
        image_count = 0

        if image_urls:
            images_dir = prod_dir / "images"
            images_dir.mkdir(exist_ok=True)
            for i, url in enumerate(image_urls):
                png_path = images_dir / f"{i}.png"
                try:
                    resp = session.get(url, timeout=IMAGE_TIMEOUT, verify=False,
                                      allow_redirects=True)
                    if resp.status_code == 200 and len(resp.content) > 100:
                        img = Image.open(BytesIO(resp.content))
                        if img.mode not in ('RGB', 'L'):
                            img = img.convert('RGB')
                        img.save(str(png_path), format='PNG')
                        image_count += 1
                except Exception:
                    pass

        if image_count > 0:
            total_products_with_images += 1
            total_images += image_count

        md_size = md_path.stat().st_size
        text_length = len(text)

        node_attrs.append({
            "asin": asin,
            "markdown_length": text_length,
            "image_count": image_count,
            "markdown_size": md_size,
        })

        if (idx + 1) % 200 == 0 or idx == len(products) - 1:
            logger.info(f"[{idx+1}/{len(products)}] 图片总计: {total_images} "
                        f"({total_products_with_images} 产品)")

    logger.info(f"完成: {len(node_attrs)} 产品, {total_images} 图片 "
                f"({total_products_with_images} 产品有图片)")

    return node_attrs


def get_leaf_category(product):
    """提取叶品类作为标签"""
    cat_path = product.get("category_path", "")
    if cat_path:
        parts = [p.strip() for p in cat_path.split("|") if p.strip()]
        if parts:
            return parts[-1]
    return product.get("main_cat", "unknown")


def build_homogeneous_graph(products, node_attrs):
    """构建同构图 (product 节点 + co_purchase 边)"""
    logger.info("构建同构图...")

    # 构建有效 ASIN 集合
    valid_asins = {na["asin"] for na in node_attrs}
    asin_to_attrs = {na["asin"]: na for na in node_attrs}

    # 构建标签映射: leaf_category → label index
    leaf_cats = sorted(set(get_leaf_category(p) for p in products))
    label_map = {cat: i for i, cat in enumerate(leaf_cats)}
    logger.info(f"标签: {len(label_map)} 个叶品类")

    # 构建节点
    asin_to_product = {p["asin"]: p for p in products}
    nodes = []
    for na in node_attrs:
        asin = na["asin"]
        product = asin_to_product.get(asin, {})
        leaf_cat = get_leaf_category(product)
        nodes.append({
            "id": asin,
            "type": "product",
            "attributes": {
                "asin": asin,
                "markdown_length": na["markdown_length"],
                "image_count": na["image_count"],
                "markdown_size": na["markdown_size"],
                "label": leaf_cat,
                "label_index": label_map.get(leaf_cat, -1),
            }
        })

    # 构建 co_purchase 边 (also_buy + also_view, 无向, 去重)
    edges = []
    edge_set = set()
    edge_type_counts = defaultdict(int)

    for product in products:
        asin = product["asin"]
        if asin not in valid_asins:
            continue

        # also_buy 边
        for related_asin in product.get("also_buy", []):
            if related_asin in valid_asins and related_asin != asin:
                key = tuple(sorted([asin, related_asin]))
                if key not in edge_set:
                    edge_set.add(key)
                    edges.append({
                        "source": key[0],
                        "target": key[1],
                        "type": "co_purchase",
                        "attributes": {"relation": "also_buy"}
                    })
                    edge_type_counts["also_buy"] += 1

        # also_view 边
        for related_asin in product.get("also_view", []):
            if related_asin in valid_asins and related_asin != asin:
                key = tuple(sorted([asin, related_asin]))
                if key not in edge_set:
                    edge_set.add(key)
                    edges.append({
                        "source": key[0],
                        "target": key[1],
                        "type": "co_purchase",
                        "attributes": {"relation": "also_view"}
                    })
                    edge_type_counts["also_view"] += 1

    logger.info(f"节点: {len(nodes)}, 边: {len(edges)} "
                f"(also_buy: {edge_type_counts.get('also_buy', 0)}, "
                f"also_view: {edge_type_counts.get('also_view', 0)})")

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

    # 按数量排序取 top 标签
    top_labels = sorted(label_dist.items(), key=lambda x: -x[1])[:20]

    dataset = {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "name": "amazon_multimodal_graph",
            "created_at": datetime.now().strftime("%Y-%m-%d"),
            "domain": "amazon",
            "statistics": {
                "total_product_dirs": len(nodes),
                "node_count": len(nodes),
                "edge_count": len(edges),
                "total_images": total_images,
                "products_with_images": nodes_with_images,
                "total_markdown_chars": total_md_chars,
                "avg_images_per_product": round(avg_images, 1),
                "avg_md_length": round(avg_md, 0),
                "label_distribution": dict(label_dist),
                "num_labels": len(label_map),
                "top_labels": [{"label": l, "count": c} for l, c in top_labels],
            },
            "schema": {
                "node_types": ["product"],
                "edge_types": ["co_purchase"],
                "node_attributes": ["asin", "markdown_length", "image_count", "markdown_size", "label", "label_index"],
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
    """保存边 CSV"""
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EDGES_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["source_id", "target_id", "edge_type", "relation"])
        for e in edges:
            writer.writerow([
                e["source"], e["target"], e["type"],
                e["attributes"].get("relation", "")
            ])
    logger.info(f"边 CSV 已保存: {EDGES_CSV} ({len(edges)} 边)")


def main():
    logger.info("=" * 60)
    logger.info("Amazon 数据集重建 (cs_lo 格式)")
    logger.info("=" * 60)

    # 1. 加载原始数据
    products = load_raw_products()

    # 2. 创建目录 + 文本 + 图片
    node_attrs = create_products_sequential(products)

    # 3. 构建同构图
    nodes, edges, label_map = build_homogeneous_graph(products, node_attrs)

    # 4. 保存
    dataset = save_graph_json(nodes, edges, label_map)
    save_edges_csv(edges)

    # 5. 汇总
    meta = dataset["metadata"]["statistics"]
    logger.info("=" * 60)
    logger.info("✅ Amazon 数据集重建完成")
    logger.info(f"   节点 (product): {meta['node_count']}")
    logger.info(f"   边 (co_purchase): {meta['edge_count']}")
    logger.info(f"   文本总量: {meta['total_markdown_chars']:,} chars (avg {meta['avg_md_length']:.0f})")
    logger.info(f"   图片: {meta['total_images']} ({meta['products_with_images']} 产品)")
    logger.info(f"   标签: {meta['num_labels']} 类")
    logger.info(f"   目录: {PRODUCTS_DIR}")
    logger.info(f"   图: {GRAPH_JSON}")
    logger.info(f"   边: {EDGES_CSV}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
