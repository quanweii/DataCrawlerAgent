#!/usr/bin/env python3
"""
Steam 扩展到 ~10K 节点
- Steam Spy all?page=N 批量拿 appid/developer/publisher/score (快,每页1000)
- appdetails 逐个补 genre/categories/screenshots/description (慢,0.3s/个)
- 并发下载截图
- 复用现有 steam_games 目录(已有1842),补齐到10K
"""
import json, re, time, random
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
import threading
from loguru import logger
import requests, urllib3, numpy as np
from PIL import Image
urllib3.disable_warnings()

BASE_DIR = Path("./data")
RAW_DIR = BASE_DIR / "steam_raw"
GAMES_DIR = BASE_DIR / "steam_games"
RAW_JSON = RAW_DIR / "steam_games_10k.json"
STEAMSPY_JSON = RAW_DIR / "steamspy_cache.json"
CHECKPOINT_JSON = RAW_DIR / "steam_10k_checkpoint.json"
MAX_GAMES = 10000
PROXY = "http://127.0.0.1:7890"
DETAIL_WORKERS = 4   # appdetails 并发(Steam限速,别太高)
IMAGE_WORKERS = 10

_tls = threading.local()
def get_session():
    if not hasattr(_tls, 'session'):
        s = requests.Session()
        s.proxies = {"http": PROXY, "https": PROXY}
        s.headers.update({"User-Agent": "Mozilla/5.0"})
        _tls.session = s
    return _tls.session


def fetch_steamspy():
    """Steam Spy 批量拿游戏元数据"""
    if STEAMSPY_JSON.exists():
        with open(STEAMSPY_JSON) as f:
            data = json.load(f)
        logger.info(f"加载 Steam Spy 缓存: {len(data)} 游戏")
        return data
    logger.info("Steam Spy 批量爬取...")
    data = {}
    page = 0
    while len(data) < MAX_GAMES * 2:  # 多拿些供过滤
        try:
            r = get_session().get(f"https://steamspy.com/api.php?request=all&page={page}", timeout=20, verify=False)
            if r.status_code != 200: break
            d = r.json()
            if not d: break
            data.update({str(k):v for k,v in d.items()})
            if (page+1) % 5 == 0:
                logger.info(f"  Steam Spy page {page+1}: 累计 {len(data)}")
        except Exception as e:
            logger.debug(f"  page{page}: {e}")
            break
        page += 1
        time.sleep(0.3)
    with open(STEAMSPY_JSON, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    logger.info(f"Steam Spy 完成: {len(data)} 游戏")
    return data


def get_appdetails(appid):
    """appdetails 补 genre/categories/screenshots/description"""
    try:
        r = get_session().get(f"https://store.steampowered.com/api/appdetails?appids={appid}&l=english", timeout=10, verify=False)
        if r.status_code != 200: return None
        d = r.json().get(str(appid), {})
        if not d.get("success"): return None
        info = d["data"]
        if info.get("type") != "game": return None
        return {
            "genres": [g["description"] for g in info.get("genres", [])],
            "categories": [c["description"] for c in info.get("categories", [])],
            "short_description": info.get("short_description", ""),
            "about_the_game": info.get("about_the_game", ""),
            "header_image": info.get("header_image", ""),
            "screenshots": [s["path_full"] for s in info.get("screenshots", [])],
            "release_date": info.get("release_date", {}).get("date", ""),
        }
    except Exception:
        return None


def build_game(appid, spy_data, details):
    """合并 Steam Spy + appdetails"""
    spy = spy_data.get(str(appid), {})
    # 解析 developer/publisher (Steam Spy 是逗号分隔)
    dev = str(spy.get("developer","")).split(",")
    pub = str(spy.get("publisher","")).split(",")
    dev = [d.strip() for d in dev if d.strip()][:3]
    pub = [p.strip() for p in pub if p.strip()][:3]
    return {
        "appid": int(appid),
        "name": spy.get("name", ""),
        "developers": dev,
        "publishers": pub,
        "positive": spy.get("positive", 0),
        "negative": spy.get("negative", 0),
        "owners": spy.get("owners", ""),
        "price": spy.get("price", 0),
        "genres": details.get("genres", []) if details else [],
        "categories": details.get("categories", []) if details else [],
        "short_description": details.get("short_description", "") if details else "",
        "about_the_game": details.get("about_the_game", "") if details else "",
        "header_image": details.get("header_image", "") if details else "",
        "screenshots": details.get("screenshots", []) if details else [],
        "release_date": details.get("release_date", "") if details else "",
    }


def download_image(session, url, save_path):
    try:
        r = session.get(url, timeout=20, verify=False)
        if r.status_code == 200 and len(r.content) > 500:
            img = Image.open(BytesIO(r.content))
            if img.mode not in ('RGB','L'): img = img.convert('RGB')
            img.save(str(save_path), format='JPEG', quality=85)
            return True
    except Exception:
        pass
    return False


def build_text(game):
    import re as _re
    parts = []
    if game.get("name"): parts.append(f"# {game['name']}")
    meta = []
    if game.get("developers"): meta.append(f"Developer: {', '.join(game['developers'])}")
    if game.get("publishers"): meta.append(f"Publisher: {', '.join(game['publishers'])}")
    if game.get("genres"): meta.append(f"Genres: {', '.join(game['genres'])}")
    if game.get("release_date"): meta.append(f"Release: {game['release_date']}")
    if meta: parts.append("\n".join(meta))
    if game.get("short_description"): parts.append(game["short_description"])
    about = game.get("about_the_game","")
    if about:
        about = _re.sub(r'<[^>]+>',' ',about)
        about = _re.sub(r'\s+',' ',about).strip()
        if about and len(about) > len(game.get("short_description","")):
            parts.append(about[:2000])
    return "\n\n".join(parts) if parts else game.get("name","") or "Untitled"


def main():
    logger.info("="*60)
    logger.info(f"Steam 扩展到 ~{MAX_GAMES} 节点")
    logger.info("="*60)

    # 已有游戏(复用)
    if RAW_JSON.exists():
        with open(RAW_JSON) as f: games = json.load(f)
        logger.info(f"加载已有: {len(games)} 游戏")
    else:
        games = []

    spy = fetch_steamspy()

    # 选候选: 按positive排序,排除已有的
    have = {str(g["appid"]) for g in games}
    cands = [(int(k), v.get("positive",0)) for k,v in spy.items() if k not in have]
    cands.sort(key=lambda x: -x[1])
    cands = [c[0] for c in cands]
    logger.info(f"候选: {len(cands)} (排除已有 {len(have)})")

    # checkpoint
    done = set()
    if CHECKPOINT_JSON.exists():
        with open(CHECKPOINT_JSON) as f: done = set(json.load(f).get("done",[]))
    todo = [c for c in cands if c not in done]
    logger.info(f"待处理: {len(todo)}")

    # 阶段1: appdetails 补充 (并发,带限速)
    logger.info(f"\n阶段1: appdetails 补充 genre+截图URL (并发{DETAIL_WORKERS})")
    new_games = []

    def process_one(appid):
        d = get_appdetails(appid)
        g = build_game(appid, spy, d)
        return appid, g

    batch_done = 0
    with ThreadPoolExecutor(max_workers=DETAIL_WORKERS) as ex:
        futures = {ex.submit(process_one, aid): aid for aid in todo}
        for fut in as_completed(futures):
            aid = futures[fut]
            try:
                _, g = fut.result()
                # 需要有genre才能用(否则无法分类)
                if g and g["genres"]:
                    new_games.append(g)
                    done.add(aid)
            except Exception:
                done.add(aid)
            batch_done += 1
            if batch_done % 200 == 0:
                logger.info(f"  [{batch_done}/{len(todo)}] 新增有效: {len(new_games)} (总计 {len(games)+len(new_games)})")
                # checkpoint + 保存
                with open(CHECKPOINT_JSON,'w') as f: json.dump({"done":list(done)},f)
                games_all = games + new_games
                with open(RAW_JSON,'w',encoding='utf-8') as f: json.dump(games_all,f,ensure_ascii=False)
            if len(games) + len(new_games) >= MAX_GAMES:
                # 取消剩余
                for f2 in futures: f2.cancel()
                break

    games = games + new_games
    # 截到MAX
    games = games[:MAX_GAMES]
    with open(RAW_JSON,'w',encoding='utf-8') as f: json.dump(games,f,ensure_ascii=False)
    logger.info(f"原始数据: {len(games)} 游戏 (有genre)")

    # 阶段2: 创建 cs_lo 目录 + 下载截图
    logger.info(f"\n阶段2: 创建目录 + 下载截图 (并发{IMAGE_WORKERS})")
    GAMES_DIR.mkdir(parents=True, exist_ok=True)
    def create_dir(game):
        appid = str(game["appid"])
        gd = GAMES_DIR / appid
        md_path = gd / f"{appid}.md"
        if md_path.exists():
            return appid, True
        gd.mkdir(parents=True, exist_ok=True)
        text = build_text(game)
        md_path.write_text(text, encoding='utf-8')
        np.save(str(gd/f"{appid}.npy"), np.array(text), allow_pickle=True)
        # 下载截图(最多2张)
        cnt = 0
        for i, ss in enumerate(game.get("screenshots",[])[:2]):
            if download_image(get_session(), ss, gd/"images"/f"{i}.jpg"):
                cnt += 1
        (gd/"images").mkdir(exist_ok=True)
        if cnt == 0 and game.get("header_image"):
            if download_image(get_session(), game["header_image"], gd/"images"/"0.jpg"):
                cnt = 1
        return appid, cnt>0

    n_img = 0
    with ThreadPoolExecutor(max_workers=IMAGE_WORKERS) as ex:
        for i,(aid,ok) in enumerate(ex.map(create_dir, games)):
            if ok: n_img += 1
            if (i+1) % 500 == 0:
                logger.info(f"  目录创建 {i+1}/{len(games)}, 有图: {n_img}")
    logger.info(f"完成: {len(games)}游戏, {n_img}有图")

    logger.info("="*60)
    logger.info(f"✅ Steam 扩展完成: {len(games)} 节点")
    logger.info(f"   原始数据: {RAW_JSON}")
    logger.info(f"   cs_lo目录: {GAMES_DIR}")
    logger.info(f"{'='*60}\n下一步: 运行 rebuild_steam_graph.py (需指向10k数据) 重建图")


if __name__ == "__main__":
    main()
