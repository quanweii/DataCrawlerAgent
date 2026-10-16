#!/usr/bin/env python3
"""
Steam 游戏多模态属性图数据集构建
- 从 Steam Store API 主动爬取游戏数据
- 每个游戏一个目录: steam_games/{appid}/{appid}.md, {appid}.npy, images/*.jpg
- 同构图: game 节点 + co_attribute 边 (shared_genre / same_developer / same_publisher)
- 多模态: text (描述) + image (截图)
- 标签: genre (动作/策略/RPG/...)
"""
import os
import sys
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

from loguru import logger
import requests
import urllib3
urllib3.disable_warnings()
from PIL import Image

# ==================== 路径配置 ====================
BASE_DIR = Path("./data")
RAW_DIR = BASE_DIR / "steam_raw"
GAMES_DIR = BASE_DIR / "steam_games"
DATASETS_DIR = BASE_DIR / "steam_datasets"
GRAPHS_DIR = BASE_DIR / "steam_graphs"

GRAPH_JSON = DATASETS_DIR / "steam_multimodal_graph.json"
EDGES_CSV = GRAPHS_DIR / "co_attribute_edges.csv"

RAW_JSON = RAW_DIR / "steam_games_raw.json"
CHECKPOINT_JSON = RAW_DIR / "steam_checkpoint.json"

PROXY = "http://127.0.0.1:7890"
MAX_GAMES = 20000          # 目标节点数
MAX_NODES_PER_DEVELOPER = 200  # 边密度控制
MAX_NODES_PER_GENRE = 500
REQUEST_DELAY = 0.3        # API 请求间隔 (秒)
IMAGE_TIMEOUT = 20
MAX_RETRIES = 3


def get_session():
    s = requests.Session()
    s.proxies = {"http": PROXY, "https": PROXY}
    s.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"})
    return s


# ==================== Step 1: 发现游戏 App IDs ====================

def discover_app_ids(session):
    """获取 Steam 游戏 App IDs — 通过 Store 搜索分页"""
    logger.info("发现 Steam App IDs (Store 搜索分页)...")
    app_ids = []
    seen = set()

    # 方法1: Store 搜索分页 (每页100个, 最多200页=20000个)
    sort_methods = ["Price_ASC", "Price_DESC", "Released_ASC", "Released_DESC",
                    "Relevance_ASC", "Relevance_DESC", "Name_ASC", "Name_DESC",
                    "Reviews_DESC", "Steam_Reviews"]
    for sort_by in sort_methods:
        for start in range(0, 10000, 100):
            if len(app_ids) >= MAX_GAMES * 2:  # 多取一些供过滤
                break
            try:
                url = (f"https://store.steampowered.com/search/results/?query=&start={start}"
                       f"&count=100&sort_by={sort_by}&category1=998&os=win&infinite=1")
                resp = session.get(url, timeout=15, verify=False)
                if resp.status_code != 200:
                    break
                data = resp.json()
                html = data.get("results_html", "")
                ids = re.findall(r'data-ds-appid="(\d+)"', html)
                new_count = 0
                for aid_str in ids:
                    aid = int(aid_str)
                    if aid not in seen:
                        seen.add(aid)
                        app_ids.append(aid)
                        new_count += 1
                if new_count == 0 and start > 0:
                    break  # 没有新结果, 停止此排序
            except Exception as e:
                logger.debug(f"  搜索失败 start={start}: {e}")
                break
            time.sleep(0.2)
        if len(app_ids) >= MAX_GAMES * 2:
            break
        logger.info(f"  {sort_by}: 累计 {len(app_ids)} 个 ID")

    # 方法2: Steam Spy top100 补充
    try:
        for endpoint in ["top100forever", "top100in2weeks", "top100owned"]:
            resp = session.get(f"https://steamspy.com/api.php?request={endpoint}",
                             timeout=15, verify=False)
            if resp.status_code == 200:
                data = resp.json()
                for k, v in data.items():
                    aid = int(v.get("appid", k))
                    if aid not in seen:
                        seen.add(aid)
                        app_ids.append(aid)
    except Exception:
        pass

    logger.info(f"共发现 {len(app_ids)} 个 App ID")
    return app_ids


def load_or_crawl_raw(session):
    """加载或爬取原始数据"""
    # 尝试加载已有的原始数据
    if RAW_JSON.exists():
        with open(RAW_JSON, 'r', encoding='utf-8') as f:
            games = json.load(f)
        logger.info(f"加载已有原始数据: {len(games)} 个游戏")
        return games

    # 爬取新数据
    app_ids = discover_app_ids(session)
    logger.info(f"共 {len(app_ids)} 个 App ID, 目标采集 {MAX_GAMES} 个游戏")

    # 加载 checkpoint
    done_ids = set()
    if CHECKPOINT_JSON.exists():
        with open(CHECKPOINT_JSON, 'r', encoding='utf-8') as f:
            done_ids = set(json.load(f).get("done", []))
        logger.info(f"Checkpoint: {len(done_ids)} 已完成")

    # 过滤已完成的
    todo_ids = [aid for aid in app_ids if aid not in done_ids]
    logger.info(f"待爬取: {len(todo_ids)}")

    games = []
    # 先加载已完成的 (如果有部分结果)
    partial_path = RAW_DIR / "steam_games_partial.json"
    if partial_path.exists():
        with open(partial_path, 'r', encoding='utf-8') as f:
            games = json.load(f)
        logger.info(f"加载部分结果: {len(games)} 个游戏")

    session2 = get_session()
    fail_streak = 0

    for idx, appid in enumerate(todo_ids):
        if len(games) >= MAX_GAMES:
            logger.info(f"已达到目标 {MAX_GAMES} 个游戏, 停止爬取")
            break

        try:
            resp = session2.get(
                f"https://store.steampowered.com/api/appdetails?appids={appid}&l=english",
                timeout=10, verify=False
            )
            if resp.status_code != 200:
                fail_streak += 1
                if fail_streak > 50:
                    logger.warning("连续失败过多, 暂停 60 秒...")
                    time.sleep(60)
                    fail_streak = 0
                continue

            data = resp.json().get(str(appid), {})
            if not data.get("success"):
                continue

            info = data["data"]
            # 只保留 "game" 类型
            if info.get("type") != "game":
                continue

            game = {
                "appid": appid,
                "name": info.get("name", ""),
                "type": info.get("type", ""),
                "short_description": info.get("short_description", ""),
                "about_the_game": info.get("about_the_game", ""),
                "developers": info.get("developers", []),
                "publishers": info.get("publishers", []),
                "genres": [g["description"] for g in info.get("genres", [])],
                "categories": [c["description"] for c in info.get("categories", [])],
                "header_image": info.get("header_image", ""),
                "screenshots": [s["path_full"] for s in info.get("screenshots", [])],
                "release_date": info.get("release_date", {}).get("date", ""),
                "metacritic": info.get("metacritic", {}).get("score", 0) if info.get("metacritic") else 0,
                "price": info.get("price_overview", {}).get("final_formatted", "Free") if info.get("price_overview") else "Free",
            }

            games.append(game)
            fail_streak = 0
            done_ids.add(appid)

            if len(games) % 100 == 0:
                logger.info(f"[{len(games)}/{MAX_GAMES}] 最近: {game['name'][:40]}")

            # Checkpoint 每 500 个
            if len(games) % 500 == 0:
                with open(CHECKPOINT_JSON, 'w') as f:
                    json.dump({"done": list(done_ids)}, f)
                with open(partial_path, 'w', encoding='utf-8') as f:
                    json.dump(games, f, ensure_ascii=False)

            time.sleep(REQUEST_DELAY)

        except Exception as e:
            fail_streak += 1
            if fail_streak > 50:
                time.sleep(60)
                fail_streak = 0
            continue

    # 保存最终原始数据
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAW_JSON, 'w', encoding='utf-8') as f:
        json.dump(games, f, ensure_ascii=False)
    logger.info(f"原始数据已保存: {RAW_JSON} ({len(games)} 个游戏)")

    return games


# ==================== Step 2: 文本内容构建 ====================

def build_text_content(game):
    parts = []
    name = game.get("name", "").strip()
    if name:
        parts.append(f"# {name}")

    meta_lines = []
    if game.get("developers"):
        meta_lines.append(f"Developer: {', '.join(game['developers'])}")
    if game.get("publishers"):
        meta_lines.append(f"Publisher: {', '.join(game['publishers'])}")
    if game.get("genres"):
        meta_lines.append(f"Genres: {', '.join(game['genres'])}")
    if game.get("release_date"):
        meta_lines.append(f"Release: {game['release_date']}")
    if game.get("price"):
        meta_lines.append(f"Price: {game['price']}")

    if meta_lines:
        parts.append("\n".join(meta_lines))

    desc = game.get("short_description", "").strip()
    if desc:
        parts.append(desc)

    about = game.get("about_the_game", "").strip()
    # 去除 HTML 标签
    import re
    about = re.sub(r'<[^>]+>', ' ', about).strip()
    about = re.sub(r'\s+', ' ', about)
    if about and about != desc:
        if len(about) > 2000:
            about = about[:2000]
        parts.append(about)

    return "\n\n".join(parts) if parts else name or "Untitled Game"


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


def create_game_directories(games):
    GAMES_DIR.mkdir(parents=True, exist_ok=True)
    session = get_session()

    node_attrs = []
    total_images = 0
    skipped = 0

    for idx, game in enumerate(games):
        appid = str(game["appid"])
        game_dir = GAMES_DIR / appid

        # 检查是否已完成
        md_path = game_dir / f"{appid}.md"
        if md_path.exists():
            text = md_path.read_text(encoding='utf-8')
            images_dir = game_dir / "images"
            img_count = len(list(images_dir.glob("*"))) if images_dir.exists() else 0
            node_attrs.append({
                "appid": appid,
                "markdown_length": len(text),
                "image_count": img_count,
                "markdown_size": md_path.stat().st_size,
            })
            total_images += img_count
            skipped += 1
            continue

        game_dir.mkdir(parents=True, exist_ok=True)

        # 写 .md
        text = build_text_content(game)
        md_path.write_text(text, encoding='utf-8')

        # 写 .npy
        npy_path = game_dir / f"{appid}.npy"
        np.save(str(npy_path), np.array(text), allow_pickle=True)

        # 下载截图 (最多3张)
        image_count = 0
        screenshots = game.get("screenshots", [])
        if screenshots:
            images_dir = game_dir / "images"
            images_dir.mkdir(exist_ok=True)
            for i, ss_url in enumerate(screenshots[:3]):
                jpg_path = images_dir / f"{i}.jpg"
                if download_image(session, ss_url, jpg_path):
                    image_count += 1
                    total_images += 1

        # 如果无截图，尝试下载 header_image
        if image_count == 0 and game.get("header_image"):
            images_dir = game_dir / "images"
            images_dir.mkdir(exist_ok=True)
            jpg_path = images_dir / "0.jpg"
            if download_image(session, game["header_image"], jpg_path):
                image_count += 1
                total_images += 1

        node_attrs.append({
            "appid": appid,
            "markdown_length": len(text),
            "image_count": image_count,
            "markdown_size": md_path.stat().st_size,
        })

        if (idx + 1) % 200 == 0 or idx == len(games) - 1:
            logger.info(f"[{idx+1}/{len(games)}] 图片总计: {total_images} (跳过: {skipped})")

        time.sleep(0.1)  # 图片下载限速

    logger.info(f"cs_lo 目录完成: {len(node_attrs)} 游戏, {total_images} 图片")
    return node_attrs


# ==================== Step 4: 构建同构图 ====================

def build_homogeneous_graph(games, node_attrs):
    logger.info("构建 Steam 游戏同构图...")

    valid_ids = {na["appid"] for na in node_attrs}
    id_to_game = {str(g["appid"]): g for g in games}

    # 标签映射: genre → label_index
    all_genres = set()
    for g in games:
        all_genres.update(g.get("genres", []))
    genres_sorted = sorted(all_genres)
    label_map = {g: i for i, g in enumerate(genres_sorted)}
    logger.info(f"标签: {len(label_map)} 个流派")

    # 构建节点 (标签取第一个 genre)
    nodes = []
    for na in node_attrs:
        appid = na["appid"]
        game = id_to_game.get(appid, {})
        game_genres = game.get("genres", [])
        label = game_genres[0] if game_genres else "Unknown"
        nodes.append({
            "id": appid,
            "type": "game",
            "attributes": {
                "appid": appid,
                "name": game.get("name", ""),
                "markdown_length": na["markdown_length"],
                "image_count": na["image_count"],
                "markdown_size": na["markdown_size"],
                "label": label,
                "label_index": label_map.get(label, -1),
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

    # --- shared_genre 边 ---
    logger.info("构建 shared_genre 边...")
    genre_to_ids = defaultdict(list)
    for na in node_attrs:
        game = id_to_game.get(na["appid"], {})
        for genre in game.get("genres", []):
            genre_to_ids[genre].append(na["appid"])

    for genre, gids in genre_to_ids.items():
        if len(gids) < 2:
            continue
        if len(gids) > MAX_NODES_PER_GENRE:
            sampled = random.Random(42).sample(gids, MAX_NODES_PER_GENRE)
            for a, b in combinations(sampled, 2):
                add_edge(a, b, "shared_genre")
            remaining = set(gids) - set(sampled)
            for rid in remaining:
                partner = random.Random(hash(rid)).choice(sampled)
                add_edge(rid, partner, "shared_genre")
        else:
            for a, b in combinations(gids, 2):
                add_edge(a, b, "shared_genre")

    logger.info(f"shared_genre: {edge_type_counts['shared_genre']}")

    # --- same_developer 边 ---
    logger.info("构建 same_developer 边...")
    dev_to_ids = defaultdict(list)
    for na in node_attrs:
        game = id_to_game.get(na["appid"], {})
        for dev in game.get("developers", []):
            if dev:
                dev_to_ids[dev].append(na["appid"])

    for dev, gids in dev_to_ids.items():
        if len(gids) < 2:
            continue
        if len(gids) > MAX_NODES_PER_DEVELOPER:
            sampled = random.Random(42).sample(gids, MAX_NODES_PER_DEVELOPER)
            for a, b in combinations(sampled, 2):
                add_edge(a, b, "same_developer")
            remaining = set(gids) - set(sampled)
            for rid in remaining:
                partner = random.Random(hash(rid)).choice(sampled)
                add_edge(rid, partner, "same_developer")
        else:
            for a, b in combinations(gids, 2):
                add_edge(a, b, "same_developer")

    logger.info(f"same_developer: {edge_type_counts['same_developer']}")

    # --- same_publisher 边 ---
    logger.info("构建 same_publisher 边...")
    pub_to_ids = defaultdict(list)
    for na in node_attrs:
        game = id_to_game.get(na["appid"], {})
        for pub in game.get("publishers", []):
            if pub:
                pub_to_ids[pub].append(na["appid"])

    for pub, gids in pub_to_ids.items():
        if len(gids) < 2:
            continue
        if len(gids) > MAX_NODES_PER_DEVELOPER:
            sampled = random.Random(42).sample(gids, MAX_NODES_PER_DEVELOPER)
            for a, b in combinations(sampled, 2):
                add_edge(a, b, "same_publisher")
            remaining = set(gids) - set(sampled)
            for rid in remaining:
                partner = random.Random(hash(rid)).choice(sampled)
                add_edge(rid, partner, "same_publisher")
        else:
            for a, b in combinations(gids, 2):
                add_edge(a, b, "same_publisher")

    logger.info(f"same_publisher: {edge_type_counts['same_publisher']}")

    logger.info(f"节点: {len(nodes)}, 边: {len(edges)}")
    logger.info(f"边分布: {dict(edge_type_counts)}")

    # 同质性
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
            "name": "steam_multimodal_graph",
            "created_at": datetime.now().strftime("%Y-%m-%d"),
            "domain": "steam",
            "statistics": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "total_images": total_images,
                "games_with_images": nodes_with_images,
                "total_markdown_chars": total_md_chars,
                "avg_images_per_game": round(total_images / max(len(nodes), 1), 1),
                "avg_md_length": round(total_md_chars / max(len(nodes), 1), 0),
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
    logger.info("Steam 游戏多模态属性图构建 (从 Steam API 主动爬取)")
    logger.info("=" * 60)

    session = get_session()

    # 1. 爬取原始数据
    games = load_or_crawl_raw(session)
    logger.info(f"共 {len(games)} 个游戏")

    # 2. 创建 cs_lo 目录 + 下载图片
    node_attrs = create_game_directories(games)

    # 3. 构建同构图
    nodes, edges, label_map = build_homogeneous_graph(games, node_attrs)

    # 4. 保存
    dataset = save_graph_json(nodes, edges, label_map)
    save_edges_csv(edges)

    # 5. 汇总
    meta = dataset["metadata"]["statistics"]
    logger.info("=" * 60)
    logger.info("✅ Steam 数据集构建完成")
    logger.info(f"   节点 (game): {meta['node_count']}")
    logger.info(f"   边 (co_attribute): {meta['edge_count']}")
    logger.info(f"   图片: {meta['total_images']} ({meta['games_with_images']} 游戏)")
    logger.info(f"   标签: {meta['num_labels']} 个流派")
    logger.info(f"   目录: {GAMES_DIR}")
    logger.info(f"   图: {GRAPH_JSON}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
