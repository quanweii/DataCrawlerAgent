#!/usr/bin/env python3
"""
WikiArt style/genre 元数据补充爬取
- 从 wikiart.org 的"按风格/题材浏览"页面爬取
- 通过 paintingUrl 匹配原始数据中的画作，填充 style/genre 字段
- 然后重建图

匹配策略: paintingUrl 格式 /en/{artist_url}/{title-slug}
         与原始数据的 artist_url + title 进行 slug 匹配
"""
import json
import re
import time
from pathlib import Path
from collections import defaultdict
from loguru import logger
import requests
import urllib3
urllib3.disable_warnings()

BASE_DIR = Path("./data")
RAW_DIR = BASE_DIR / "wikiart_raw"
RAW_JSON = RAW_DIR / "wikiart_paintings_raw.json"
ENRICHED_JSON = RAW_DIR / "wikiart_paintings_enriched.json"
CHECKPOINT_STYLE = RAW_DIR / "wikiart_style_checkpoint.json"
CHECKPOINT_GENRE = RAW_DIR / "wikiart_genre_checkpoint.json"

PROXY = "http://127.0.0.1:7890"
REQUEST_DELAY = 0.3
PAGE_SIZE = 60


def get_session():
    s = requests.Session()
    s.proxies = {"http": PROXY, "https": PROXY}
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    return s


def slugify(text):
    """将标题转为 slug (与 WikiArt URL 格式一致)"""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)  # 去标点
    text = re.sub(r'[\s_]+', '-', text)    # 空格转连字符
    text = re.sub(r'-+', '-', text)        # 合并多连字符
    text = text.strip('-')
    return text


def build_match_index(raw):
    """构建匹配索引: (artist_url, title_slug) → painting_index"""
    index = {}
    for i, p in enumerate(raw):
        au = p.get("artist_url", "")
        title = p.get("title", "")
        if au and title:
            key = (au, slugify(title))
            index[key] = i
    return index


def match_painting(painting_url, artist_url_field, title_field, match_index):
    """将风格页的画作与原始数据匹配
    painting_url: /en/{artist_url}/{title-slug}
    返回匹配的原始数据 index 或 None
    """
    if not painting_url:
        return None
    # 解析 paintingUrl
    m = re.match(r'/en/([^/]+)/(.+)', painting_url)
    if not m:
        return None
    au_slug = m.group(1)
    title_slug = m.group(2)
    # 去掉尾部年份 (如 -1889, -1853-1853)
    title_slug_clean = re.sub(r'-\d{4}(-\d{4})?$', '', title_slug)

    # 精确匹配
    key = (au_slug, title_slug_clean)
    if key in match_index:
        return match_index[key]

    # 模糊匹配: title_slug_clean 可能是原始 slug 的前缀 (或反之)
    for (au, ts), idx in match_index.items():
        if au == au_slug:
            if ts == title_slug_clean or ts.startswith(title_slug_clean) or title_slug_clean.startswith(ts):
                return idx
    return None


def crawl_category_pages(session, category_type, match_index, checkpoint_path):
    """
    category_type: 'style' or 'genre'
    爬取所有风格/题材页面，返回 {painting_index: category_name}
    """
    cat_to_paintings = {}  # painting_index → category_name
    page_base = f"https://www.wikiart.org/en/paintings-by-{category_type}"

    # 获取分类列表
    logger.info(f"获取 {category_type} 列表...")
    resp = session.get(page_base, timeout=20, verify=False)
    cat_links = re.findall(rf'href="(/en/paintings-by-{category_type}/[^"?#]+)"', resp.text)
    cat_slugs = sorted(set(l.split('/')[-1] for l in cat_links))
    logger.info(f"共 {len(cat_slugs)} 个 {category_type}")

    # 加载 checkpoint
    done_cats = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            done_cats = set(json.load(f).get("done", []))
        # 恢复已匹配的
        partial = RAW_DIR / f"wikiart_{category_type}_mapping.json"
        if partial.exists():
            with open(partial) as f:
                for k, v in json.load(f).items():
                    cat_to_paintings[int(k)] = v

    for ci, cat_slug in enumerate(cat_slugs):
        if cat_slug in done_cats:
            continue

        cat_name = cat_slug.replace('-', ' ').title()
        page = 1
        cat_count = 0
        while True:
            try:
                url = f"{page_base}/{cat_slug}?json=2&p={page}"
                resp = session.get(url, timeout=20, verify=False)
                if resp.status_code != 200:
                    break
                data = resp.json()
                paintings = data.get("Paintings", [])
                total = data.get("AllPaintingsCount", 0)
                if not paintings:
                    break

                for p in paintings:
                    purl = p.get("paintingUrl", "")
                    idx = match_painting(purl, None, None, match_index)
                    if idx is not None:
                        if idx not in cat_to_paintings:
                            cat_to_paintings[idx] = cat_name
                            cat_count += 1

                # 是否还有更多页
                if page * PAGE_SIZE >= total or len(paintings) < PAGE_SIZE:
                    break
                page += 1
                time.sleep(REQUEST_DELAY)
            except Exception as e:
                logger.debug(f"  {cat_slug} p={page} 失败: {e}")
                break

        done_cats.add(cat_slug)

        if (ci + 1) % 5 == 0 or ci == len(cat_slugs) - 1:
            logger.info(f"[{ci+1}/{len(cat_slugs)}] {cat_slug}: 匹配 {cat_count} 画作 "
                       f"(总已匹配 {len(cat_to_paintings)})")
            # Checkpoint
            with open(checkpoint_path, 'w') as f:
                json.dump({"done": list(done_cats)}, f)
            with open(RAW_DIR / f"wikiart_{category_type}_mapping.json", 'w') as f:
                json.dump({str(k): v for k, v in cat_to_paintings.items()}, f)

        time.sleep(REQUEST_DELAY)

    return cat_to_paintings


def main():
    logger.info("=" * 60)
    logger.info("WikiArt style/genre 元数据补充爬取")
    logger.info("=" * 60)

    session = get_session()

    # 加载原始数据
    with open(RAW_JSON, encoding="utf-8") as f:
        raw = json.load(f)
    logger.info(f"原始数据: {len(raw)} 幅画")

    # 构建匹配索引
    match_index = build_match_index(raw)
    logger.info(f"匹配索引: {len(match_index)} 条")

    # 1. 爬取 style
    logger.info("\n--- 阶段1: 爬取风格 ---")
    style_map = crawl_category_pages(session, "style", match_index, CHECKPOINT_STYLE)
    logger.info(f"风格匹配: {len(style_map)}/{len(raw)} ({100*len(style_map)/len(raw):.1f}%)")

    # 2. 爬取 genre
    logger.info("\n--- 阶段2: 爬取题材 ---")
    genre_map = crawl_category_pages(session, "genre", match_index, CHECKPOINT_GENRE)
    logger.info(f"题材匹配: {len(genre_map)}/{len(raw)} ({100*len(genre_map)/len(raw):.1f}%)")

    # 3. 合并到原始数据
    logger.info("\n--- 阶段3: 合并元数据 ---")
    for i, p in enumerate(raw):
        if i in style_map:
            p["style"] = style_map[i]
        if i in genre_map:
            p["genre"] = genre_map[i]

    has_style = sum(1 for p in raw if p.get("style"))
    has_genre = sum(1 for p in raw if p.get("genre"))
    logger.info(f"填充完成: style={has_style}/{len(raw)} ({100*has_style/len(raw):.1f}%), "
               f"genre={has_genre}/{len(raw)} ({100*has_genre/len(raw):.1f}%)")

    # 4. 保存
    with open(ENRICHED_JSON, 'w', encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False)
    logger.info(f"已保存: {ENRICHED_JSON}")

    # 5. 统计
    from collections import Counter
    styles = Counter(p["style"] for p in raw if p.get("style"))
    genres = Counter(p["genre"] for p in raw if p.get("genre"))
    logger.info(f"\n风格数: {len(styles)}, Top5: {styles.most_common(5)}")
    logger.info(f"题材数: {len(genres)}, Top5: {genres.most_common(5)}")

    logger.info("=" * 60)
    logger.info("✅ 元数据补充完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
