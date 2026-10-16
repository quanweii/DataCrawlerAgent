#!/usr/bin/env python3
"""
WikiArt 文本补充 — 爬取艺术家简介 + 画作 meta description
- 艺术家 bio (1370个艺术家, 每个爬一次, 含真实散文)
- 画作 meta description (补充 genre 字段 + 模板描述)
- 过滤 <20 样本的小类别
- 重建 .md 文本 + 图

文本结构: # {title}
         Artist: {artist} | Style: {style} | Genre: {genre} | Year: {year}
         {画作 meta description}
         ## About the Artist
         {艺术家 bio}
"""
import json, re, time
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
import requests, urllib3
from bs4 import BeautifulSoup
urllib3.disable_warnings()
import numpy as np

BASE_DIR = Path("./data")
RAW_DIR = BASE_DIR / "wikiart_raw"
ARTWORKS_DIR = BASE_DIR / "wikiart_artworks_v2"
RAW_JSON = RAW_DIR / "wikiart_v2_raw.json"
ENRICHED_JSON = RAW_DIR / "wikiart_v2_enriched.json"
ARTIST_BIO_JSON = RAW_DIR / "wikiart_artist_bios.json"
META_DESC_JSON = RAW_DIR / "wikiart_meta_desc.json"

PROXY = "http://127.0.0.1:7890"
MIN_CLASS_SIZE = 20   # 过滤阈值
BIO_WORKERS = 8
DESC_WORKERS = 12

import threading
_tls = threading.local()
def get_session():
    if not hasattr(_tls, 'session'):
        s = requests.Session()
        s.proxies = {"http": PROXY, "https": PROXY}
        s.headers.update({"User-Agent": "Mozilla/5.0"})
        _tls.session = s
    return _tls.session


def fetch_artist_bio(artist_url):
    """爬单个艺术家的简介"""
    try:
        r = get_session().get(f"https://www.wikiart.org/en/{artist_url}", timeout=20, verify=False)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, 'html.parser')
        # 找 bio 段: 跳过Wikipedia模板/导航/页脚, 找第一个实质性段落
        skip_kw = ['wikipedia article', 'creative commons', 'wikiart.org',
                   'signup', 'sign in', 'cookie', 'privacy', 'terms of']
        for p in soup.find_all('p'):
            t = p.get_text(strip=True)
            t_low = t.lower()
            if (80 < len(t) < 1200
                    and not any(k in t_low for k in skip_kw)
                    and not t.startswith('{{')):
                return t
        # 退化: 用 meta description
        meta = soup.find('meta', {'name':'description'})
        if meta:
            c = meta.get('content', '')
            # "lived in the XIX cent., a remarkable figure of Dutch Post-Impressionism"
            return c.split('.')[0] + '.' if c else ""
        return ""
    except Exception:
        return ""


def fetch_painting_desc(painting_url):
    """爬单个画作的 meta description (同时含 genre 关键词)"""
    try:
        r = get_session().get(f"https://www.wikiart.org{painting_url}", timeout=15, verify=False)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, 'html.parser')
        meta = soup.find('meta', {'name':'description'})
        if meta:
            desc = meta.get('content', '')
            # 去掉尾部 "Find more prominent...Wikiart.org"
            desc = re.sub(r'\s*Find more prominent.*Wikiart\.org.*$', '', desc, flags=re.I|re.S)
            return desc.strip()
        return ""
    except Exception:
        return ""


def extract_genre_from_desc(desc):
    """从描述中提取 genre: 'in XXX style. Find more prominent pieces of GENRE at'"""
    if not desc:
        return ""
    m = re.search(r'prominent pieces of (.+?) at', desc)
    if m:
        return m.group(1).strip().title()
    return ""


def crawl_bios(artist_urls):
    """并发爬取所有艺术家 bio"""
    bios = {}
    if ARTIST_BIO_JSON.exists():
        with open(ARTIST_BIO_JSON) as f:
            bios = json.load(f)
        logger.info(f"已加载 {len(bios)} 个艺术家 bio")
    todo = [a for a in artist_urls if a not in bios]
    logger.info(f"待爬取 bio: {len(todo)} 个艺术家")
    if not todo:
        return bios

    done = 0
    with ThreadPoolExecutor(max_workers=BIO_WORKERS) as ex:
        futures = {ex.submit(fetch_artist_bio, a): a for a in todo}
        for fut in as_completed(futures):
            a = futures[fut]
            try:
                bios[a] = fut.result()
            except Exception:
                bios[a] = ""
            done += 1
            if done % 100 == 0:
                logger.info(f"bio: {done}/{len(todo)}")
                with open(ARTIST_BIO_JSON, 'w') as f:
                    json.dump(bios, f, ensure_ascii=False)
    with open(ARTIST_BIO_JSON, 'w') as f:
        json.dump(bios, f, ensure_ascii=False)
    has_bio = sum(1 for v in bios.values() if v)
    logger.info(f"bio 完成: {has_bio}/{len(bios)} 有内容")
    return bios


def crawl_descs(paintings):
    """并发爬取画作 meta description"""
    descs = {}
    if META_DESC_JSON.exists():
        with open(META_DESC_JSON) as f:
            descs = json.load(f)
        logger.info(f"已加载 {len(descs)} 个画作描述")
    todo = [(p['painting_url'], p['painting_id']) for p in paintings
            if p['painting_id'] not in descs]
    logger.info(f"待爬取描述: {len(todo)}")
    if not todo:
        return descs

    done = 0
    with ThreadPoolExecutor(max_workers=DESC_WORKERS) as ex:
        futures = {ex.submit(fetch_painting_desc, url): pid for url, pid in todo}
        for fut in as_completed(futures):
            pid = futures[fut]
            try:
                descs[pid] = fut.result()
            except Exception:
                descs[pid] = ""
            done += 1
            if done % 500 == 0:
                logger.info(f"desc: {done}/{len(todo)}")
                with open(META_DESC_JSON, 'w', encoding='utf-8') as f:
                    json.dump(descs, f, ensure_ascii=False)
    with open(META_DESC_JSON, 'w', encoding='utf-8') as f:
        json.dump(descs, f, ensure_ascii=False)
    has_desc = sum(1 for v in descs.values() if v)
    logger.info(f"描述完成: {has_desc}/{len(descs)} 有内容")
    return descs


def build_text(p, bio, desc):
    """构建丰富文本"""
    parts = []
    title = p.get("title", "").strip()
    if title:
        parts.append(f"# {title}")
    meta = []
    if p.get("artist"): meta.append(f"Artist: {p['artist']}")
    if p.get("style"): meta.append(f"Style: {p['style']}")
    if p.get("genre"): meta.append(f"Genre: {p['genre']}")
    if p.get("year"): meta.append(f"Year: {p['year']}")
    if meta: parts.append("\n".join(meta))
    if desc: parts.append(desc)
    if bio:
        parts.append("## About the Artist")
        parts.append(bio)
    return "\n\n".join(parts) if parts else title or "Untitled"


def main():
    logger.info("="*60)
    logger.info("WikiArt 文本补充 + 小类过滤")
    logger.info("="*60)

    with open(RAW_JSON, encoding="utf-8") as f:
        paintings = json.load(f)
    logger.info(f"原始画作: {len(paintings)}")

    # 1. 过滤小类别
    label_count = Counter(p["style"] for p in paintings)
    small = {l for l, c in label_count.items() if c < MIN_CLASS_SIZE}
    logger.info(f"过滤 <{MIN_CLASS_SIZE} 的小类: {len(small)} 个 → {sorted(small)}")
    paintings = [p for p in paintings if p["style"] not in small]
    logger.info(f"过滤后: {len(paintings)} 画作, {len(set(p['style'] for p in paintings))} 类")

    # 2. 爬艺术家 bio
    artist_urls = sorted(set(p["artist_url"] for p in paintings))
    logger.info(f"\n阶段1: 爬取 {len(artist_urls)} 个艺术家 bio")
    bios = crawl_bios(artist_urls)

    # 3. 爬画作描述
    logger.info(f"\n阶段2: 爬取 {len(paintings)} 个画作 meta description")
    descs = crawl_descs(paintings)

    # 4. 从描述提取 genre 补充空字段
    genre_filled = 0
    for p in paintings:
        if not p.get("genre"):
            g = extract_genre_from_desc(descs.get(p["painting_id"], ""))
            if g:
                p["genre"] = g
                genre_filled += 1
    logger.info(f"\n从描述补充 genre: {genre_filled} 个")

    # 5. 重建 .md 文本
    logger.info(f"\n阶段3: 重建 {len(paintings)} 个 .md 文本")
    def rebuild(p):
        bio = bios.get(p["artist_url"], "")
        desc = descs.get(p["painting_id"], "")
        text = build_text(p, bio, desc)
        pid = re.sub(r'[^\w\-.]', '_', p["painting_id"])
        md_path = ARTWORKS_DIR / pid / f"{pid}.md"
        if md_path.parent.exists():
            md_path.write_text(text, encoding="utf-8")
            np.save(str(ARTWORKS_DIR / pid / f"{pid}.npy"),
                    np.array(text), allow_pickle=True)
        return len(text)

    total_len = 0
    with ThreadPoolExecutor(max_workers=12) as ex:
        for i, L in enumerate(ex.map(rebuild, paintings)):
            total_len += L
            if (i+1) % 1000 == 0:
                logger.info(f"  重建 {i+1}/{len(paintings)}, avg={total_len//(i+1)}字符")

    logger.info(f"平均文本长度: {total_len/len(paintings):.0f} 字符 (原83)")

    # 6. 保存 enriched 数据
    with open(ENRICHED_JSON, 'w', encoding="utf-8") as f:
        json.dump(paintings, f, ensure_ascii=False)
    logger.info(f"enriched 已保存: {ENRICHED_JSON}")

    # 统计
    has_bio = sum(1 for p in paintings if bios.get(p["artist_url"]))
    has_desc = sum(1 for p in paintings if descs.get(p["painting_id"]))
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ 文本补充完成")
    logger.info(f"   有bio: {has_bio}/{len(paintings)} ({100*has_bio/len(paintings):.0f}%)")
    logger.info(f"   有desc: {has_desc}/{len(paintings)} ({100*has_desc/len(paintings):.0f}%)")
    logger.info(f"   有genre: {sum(1 for p in paintings if p.get('genre'))}/{len(paintings)}")
    logger.info(f"   类别: {len(set(p['style'] for p in paintings))}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
