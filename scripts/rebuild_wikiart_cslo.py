#!/usr/bin/env python3
"""
将 WikiArt 数据集重建为 cs_lo 格式
- 每幅画一个目录: wikiart_artworks/{artwork_id}/{artwork_id}.md, {artwork_id}.npy, images/0.jpg
- 同构图: artwork 节点 + co_attribute 边 (同艺术家 / 同题材)
- 仅保留 text + image 两个模态
- 标签: style (27 类)

仿照 rebuild_amazon_cslo.py 的结构和 cs_lo 格式

数据来源: Kaggle steubk/wikiart
  wikiart_raw/
    ├── wikiart/                    # 图片 (按 style 子目录)
    │   ├── Impressionism/
    │   │   └── xxx_yyy.jpg
    │   └── .../
    └── wikiart_csv/
        ├── style_class.txt
        ├── genre_class.txt
        └── artist_class.txt
"""
import os
import sys
import json
import csv
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from datetime import datetime
from shutil import copy2

from loguru import logger
from PIL import Image

# ==================== 路径配置 ====================
BASE_DIR = Path("./data")
RAW_DIR = BASE_DIR / "wikiart_raw"
ARTWORKS_DIR = BASE_DIR / "wikiart_artworks"
DATASETS_DIR = BASE_DIR / "wikiart_datasets"
GRAPHS_DIR = BASE_DIR / "wikiart_graphs"

GRAPH_JSON = DATASETS_DIR / "wikiart_multimodal_graph.json"
EDGES_CSV = GRAPHS_DIR / "co_attribute_edges.csv"

# 边密度控制: same_style 边在每个风格内最多采样的节点数
# 全连接边数 = min(N, MAX_NODES_PER_STYLE) choose 2
MAX_NODES_PER_STYLE = 500

# 同一艺术家的画作超过此阈值时，采样边而非全连接
MAX_NODES_PER_ARTIST = 200

# 同一题材的画作超过此阈值时，采样边而非全连接
MAX_NODES_PER_GENRE = 300


# ==================== Step 1: 加载原始数据 ====================

def load_class_mapping(txt_path):
    """
    读取 style_class.txt / genre_class.txt / artist_class.txt
    格式: 每行 "class_name\tclass_index" 或 "class_name class_index"
    返回 {index: name} 和 {name: index}
    """
    idx2name = {}
    name2idx = {}
    if not txt_path.exists():
        logger.warning(f"映射文件不存在: {txt_path}")
        return idx2name, name2idx
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit('\t', 1) if '\t' in line else line.rsplit(' ', 1)
            if len(parts) == 2:
                name, idx_str = parts
                try:
                    idx = int(idx_str)
                    idx2name[idx] = name
                    name2idx[name] = idx
                except ValueError:
                    continue
    return idx2name, name2idx


def load_raw_artworks():
    """
    扫描 wikiart_raw/wikiart/ 目录，收集所有画作信息
    目录结构: wikiart/{style_name}/{filename}.jpg

    同时读取 CSV 映射文件获取 artist 和 genre 信息
    """
    wikiart_images_dir = RAW_DIR / "wikiart"
    csv_dir = RAW_DIR / "wikiart_csv"

    if not wikiart_images_dir.exists():
        logger.error(f"图片目录不存在: {wikiart_images_dir}")
        logger.error("请先运行 download_wikiart_data.py 下载数据")
        sys.exit(1)

    # 读取类别映射
    style_idx2name, style_name2idx = load_class_mapping(csv_dir / "style_class.txt")
    genre_idx2name, genre_name2idx = load_class_mapping(csv_dir / "genre_class.txt")
    artist_idx2name, artist_name2idx = load_class_mapping(csv_dir / "artist_class.txt")

    logger.info(f"Style 映射: {len(style_idx2name)} 类")
    logger.info(f"Genre 映射: {len(genre_idx2name)} 类")
    logger.info(f"Artist 映射: {len(artist_idx2name)} 个")

    # 从 CSV 读取图片到 artist/genre 的映射
    # Kaggle 数据集的 CSV 文件格式: filename (不含扩展名) -> class_index
    csv_files = {
        "artist": csv_dir / "artist_train.txt",
        "genre": csv_dir / "genre_train.txt",
    }
    # 尝试也读取 val/test 文件
    for split in ["val", "test"]:
        for category in ["artist", "genre"]:
            alt = csv_dir / f"{category}_{split}.txt"
            if alt.exists():
                # 合并到同一个 dict
                pass  # 下面统一处理

    # 构建 filename -> artist_index 映射
    # CSV 格式: 每行 "filename class_index" 或 "filename\tclass_index"
    file_to_artist = {}
    file_to_genre = {}

    for label_type, target_dict in [("artist", file_to_artist), ("genre", file_to_genre)]:
        for split in ["train", "val", "test"]:
            csv_path = csv_dir / f"{label_type}_{split}.txt"
            if not csv_path.exists():
                continue
            with open(csv_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.rsplit('\t', 1) if '\t' in line else line.rsplit(' ', 1)
                    if len(parts) == 2:
                        fname, idx_str = parts
                        try:
                            target_dict[fname] = int(idx_str)
                        except ValueError:
                            continue

    logger.info(f"Artist 映射: {len(file_to_artist)} 文件")
    logger.info(f"Genre 映射: {len(file_to_genre)} 文件")

    # 扫描图片目录
    artworks = []
    style_dirs = sorted([d for d in wikiart_images_dir.iterdir() if d.is_dir()])
    logger.info(f"找到 {len(style_dirs)} 个风格目录")

    for style_dir in style_dirs:
        style_name = style_dir.name
        style_idx = style_name2idx.get(style_name, -1)

        image_files = sorted([f for f in style_dir.iterdir()
                             if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp')])
        for img_path in image_files:
            # 用 风格名_文件名（无扩展名）作为 artwork_id
            stem = img_path.stem
            artwork_id = f"{style_name}_{stem}"
            # 清理 artwork_id 中的特殊字符
            artwork_id = artwork_id.replace(" ", "_").replace("(", "").replace(")", "")

            # 查找 artist 和 genre
            # CSV 中的 filename 可能包含子目录前缀或不包含
            artist_idx = file_to_artist.get(stem, file_to_artist.get(img_path.name, -1))
            genre_idx = file_to_genre.get(stem, file_to_genre.get(img_path.name, -1))

            artist_name = artist_idx2name.get(artist_idx, "unknown") if artist_idx >= 0 else "unknown"
            genre_name = genre_idx2name.get(genre_idx, "unknown") if genre_idx >= 0 else "unknown"

            # 从文件名提取可能的标题信息
            # 有些文件名格式: "artist-name_title-of-painting_year.jpg"
            title = stem.replace("_", " ").replace("-", " ").strip()
            # 截取到合理长度
            if len(title) > 200:
                title = title[:200]

            artworks.append({
                "artwork_id": artwork_id,
                "title": title,
                "artist": artist_name,
                "style": style_name,
                "style_index": style_idx,
                "genre": genre_name,
                "genre_index": genre_idx,
                "image_path": str(img_path),
            })

    logger.info(f"共加载 {len(artworks)} 幅画作")
    return artworks


# ==================== Step 2: 文本内容构建 ====================

def build_text_content(artwork):
    """构建画作的 Markdown 文本"""
    parts = []

    title = artwork.get("title", "").strip()
    if title:
        parts.append(f"# {title}")

    meta_lines = []
    artist = artwork.get("artist", "")
    if artist and artist != "unknown":
        meta_lines.append(f"Artist: {artist}")

    style = artwork.get("style", "")
    if style:
        meta_lines.append(f"Style: {style}")

    genre = artwork.get("genre", "")
    if genre and genre != "unknown":
        meta_lines.append(f"Genre: {genre}")

    if meta_lines:
        parts.append("\n".join(meta_lines))

    # 组合
    return "\n\n".join(parts) if parts else title or "Untitled artwork"


# ==================== Step 3: 创建 cs_lo 目录 ====================

def create_artwork_directories(artworks):
    """为所有画作创建 cs_lo 格式目录"""
    ARTWORKS_DIR.mkdir(parents=True, exist_ok=True)

    node_attrs = []
    total_images = 0
    skipped = 0

    for idx, artwork in enumerate(artworks):
        artwork_id = artwork["artwork_id"]
        art_dir = ARTWORKS_DIR / artwork_id

        # 检查是否已完成
        md_path = art_dir / f"{artwork_id}.md"
        if md_path.exists():
            # 已存在，跳过创建但读取统计
            text = md_path.read_text(encoding='utf-8')
            images_dir = art_dir / "images"
            img_count = len(list(images_dir.glob("*"))) if images_dir.exists() else 0
            node_attrs.append({
                "artwork_id": artwork_id,
                "markdown_length": len(text),
                "image_count": img_count,
                "markdown_size": md_path.stat().st_size,
            })
            total_images += img_count
            skipped += 1
            if skipped % 1000 == 0:
                logger.debug(f"跳过已存在: {skipped}")
            continue

        art_dir.mkdir(parents=True, exist_ok=True)

        # 写 .md 文件
        text = build_text_content(artwork)
        md_path.write_text(text, encoding='utf-8')

        # 写 .npy 文件
        npy_path = art_dir / f"{artwork_id}.npy"
        np.save(str(npy_path), np.array(text), allow_pickle=True)

        # 复制图片 (保持原始格式, 不转 PNG 节省空间)
        src_path = Path(artwork["image_path"])
        images_dir = art_dir / "images"
        images_dir.mkdir(exist_ok=True)

        image_count = 0
        if src_path.exists():
            # 确定目标扩展名 (保持原格式)
            suffix = src_path.suffix.lower()
            if suffix not in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
                suffix = '.jpg'  # 默认
            dst_path = images_dir / f"0{suffix}"
            try:
                copy2(src_path, dst_path)
                image_count = 1
                total_images += 1
            except Exception as e:
                logger.debug(f"复制图片失败 {src_path}: {e}")

        md_size = md_path.stat().st_size
        text_length = len(text)

        node_attrs.append({
            "artwork_id": artwork_id,
            "markdown_length": text_length,
            "image_count": image_count,
            "markdown_size": md_size,
        })

        if (idx + 1) % 2000 == 0 or idx == len(artworks) - 1:
            logger.info(f"[{idx+1}/{len(artworks)}] 图片总计: {total_images} "
                        f"(跳过已存在: {skipped})")

    logger.info(f"cs_lo 目录创建完成: {len(node_attrs)} 幅画, "
                f"{total_images} 张图片, 跳过 {skipped} 个已存在")
    return node_attrs


# ==================== Step 4: 构建同构图 ====================

def build_homogeneous_graph(artworks, node_attrs):
    """
    构建同构图: artwork 节点 + co_attribute 边

    边类型 (均为无向):
    - same_artist: 同一艺术家的画作
    - same_genre: 同一题材的画作

    标签: style (27 类)

    注意: 不使用 same_style 边，因为 style 就是分类标签，
         用标签连边会导致同质化过高 (edge homophily ~1.0)，
         GNN 分类任务将失去意义。
    """
    logger.info("构建同构图...")

    # 构建有效 artwork_id 集合
    valid_ids = {na["artwork_id"] for na in node_attrs}
    id_to_attrs = {na["artwork_id"]: na for na in node_attrs}

    # 构建标签映射: style -> label_index
    styles = sorted(set(a["style"] for a in artworks))
    label_map = {s: i for i, s in enumerate(styles)}
    logger.info(f"标签: {len(label_map)} 个风格")

    # 构建节点
    id_to_artwork = {a["artwork_id"]: a for a in artworks}
    nodes = []
    for na in node_attrs:
        artwork_id = na["artwork_id"]
        artwork = id_to_artwork.get(artwork_id, {})
        style_name = artwork.get("style", "unknown")
        nodes.append({
            "id": artwork_id,
            "type": "artwork",
            "attributes": {
                "artwork_id": artwork_id,
                "markdown_length": na["markdown_length"],
                "image_count": na["image_count"],
                "markdown_size": na["markdown_size"],
                "label": style_name,
                "label_index": label_map.get(style_name, -1),
            }
        })

    # ========== 边构建 ==========
    edges = []
    edge_set = set()
    edge_type_counts = defaultdict(int)

    def add_edge(id_a, id_b, relation):
        """添加无向边 (去重)"""
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
    for a in artworks:
        if a["artwork_id"] in valid_ids and a["artist"] != "unknown":
            artist_to_ids[a["artist"]].append(a["artwork_id"])

    for artist, art_ids in artist_to_ids.items():
        if len(art_ids) < 2:
            continue
        if len(art_ids) > MAX_NODES_PER_ARTIST:
            # 采样: 从该艺术家的画作中随机选 MAX_NODES_PER_ARTIST 个做全连接
            sampled = random.Random(42).sample(art_ids, MAX_NODES_PER_ARTIST)
            for a, b in combinations(sampled, 2):
                add_edge(a, b, "same_artist")
            # 剩余的每个画作与一个随机已有节点连边
            remaining = set(art_ids) - set(sampled)
            for rid in remaining:
                partner = random.Random(hash(rid)).choice(sampled)
                add_edge(rid, partner, "same_artist")
        else:
            for a, b in combinations(art_ids, 2):
                add_edge(a, b, "same_artist")

    logger.info(f"same_artist 边: {edge_type_counts['same_artist']}")

    # --- same_genre 边 ---
    logger.info("构建 same_genre 边...")
    genre_to_ids = defaultdict(list)
    for a in artworks:
        if a["artwork_id"] in valid_ids and a["genre"] != "unknown":
            genre_to_ids[a["genre"]].append(a["artwork_id"])

    for genre, art_ids in genre_to_ids.items():
        if len(art_ids) < 2:
            continue
        if len(art_ids) > MAX_NODES_PER_GENRE:
            sampled = random.Random(42).sample(art_ids, MAX_NODES_PER_GENRE)
            for a, b in combinations(sampled, 2):
                add_edge(a, b, "same_genre")
            remaining = set(art_ids) - set(sampled)
            for rid in remaining:
                partner = random.Random(hash(rid)).choice(sampled)
                add_edge(rid, partner, "same_genre")
        else:
            for a, b in combinations(art_ids, 2):
                add_edge(a, b, "same_genre")

    logger.info(f"same_genre 边: {edge_type_counts['same_genre']}")

    logger.info(f"节点: {len(nodes)}, 边: {len(edges)}")
    logger.info(f"边分布: {dict(edge_type_counts)}")

    # 计算同质性 (edge homophily)
    if edges:
        id_to_label = {n["id"]: n["attributes"]["label"] for n in nodes}
        same_label = sum(1 for e in edges
                        if id_to_label.get(e["source"]) == id_to_label.get(e["target"]))
        homophily = same_label / len(edges)
        logger.info(f"Edge homophily: {homophily:.4f} ({same_label}/{len(edges)} 边两端同标签)")

    return nodes, edges, label_map


# ==================== Step 5: 保存 ====================

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

    top_labels = sorted(label_dist.items(), key=lambda x: -x[1])

    dataset = {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "name": "wikiart_multimodal_graph",
            "created_at": datetime.now().strftime("%Y-%m-%d"),
            "domain": "wikiart",
            "statistics": {
                "total_artwork_dirs": len(nodes),
                "node_count": len(nodes),
                "edge_count": len(edges),
                "total_images": total_images,
                "artworks_with_images": nodes_with_images,
                "total_markdown_chars": total_md_chars,
                "avg_images_per_artwork": round(avg_images, 1),
                "avg_md_length": round(avg_md, 0),
                "label_distribution": dict(label_dist),
                "num_labels": len(label_map),
                "top_labels": [{"label": l, "count": c} for l, c in top_labels],
            },
            "schema": {
                "node_types": ["artwork"],
                "edge_types": ["co_attribute"],
                "node_attributes": ["artwork_id", "markdown_length", "image_count",
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


# ==================== 主函数 ====================

def main():
    logger.info("=" * 60)
    logger.info("WikiArt 数据集重建 (cs_lo 格式)")
    logger.info("=" * 60)

    # 1. 加载原始数据
    artworks = load_raw_artworks()

    # 2. 创建 cs_lo 目录 + 复制图片
    node_attrs = create_artwork_directories(artworks)

    # 3. 构建同构图
    nodes, edges, label_map = build_homogeneous_graph(artworks, node_attrs)

    # 4. 保存
    dataset = save_graph_json(nodes, edges, label_map)
    save_edges_csv(edges)

    # 5. 汇总
    meta = dataset["metadata"]["statistics"]
    logger.info("=" * 60)
    logger.info("✅ WikiArt 数据集重建完成")
    logger.info(f"   节点 (artwork): {meta['node_count']}")
    logger.info(f"   边 (co_attribute): {meta['edge_count']}")
    logger.info(f"   文本总量: {meta['total_markdown_chars']:,} chars (avg {meta['avg_md_length']:.0f})")
    logger.info(f"   图片: {meta['total_images']} ({meta['artworks_with_images']} 幅画)")
    logger.info(f"   标签: {meta['num_labels']} 个风格")
    logger.info(f"   目录: {ARTWORKS_DIR}")
    logger.info(f"   图: {GRAPH_JSON}")
    logger.info(f"   边: {EDGES_CSV}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
