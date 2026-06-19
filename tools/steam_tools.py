"""
Steam 游戏数据采集工具集 (仿 tools/amazon_tools.py 的 ToolManager 模式)

功能:
- load_raw(): 加载已爬取的 Steam 游戏元数据缓存 (steam_games_10k.json)
- crawl_live(): 实时爬取能力 (Steam Spy 批量 + appdetails 补充), 默认管线不走此路径
- 图片/文本已在 cs_lo 目录就绪, 此处仅负责元数据

设计: 默认 data_source="file" 走缓存 (数据已爬取验证); live 能力保留以支撑
论文「agent-constructed」主张与可复现性。
"""
import os
import json
from typing import Dict, Any, List, Optional
from loguru import logger


class SteamToolManager:
    """Steam 游戏元数据工具管理器"""

    DEFAULT_RAW = "steam_games_10k.json"
    DEFAULT_CSLO = "../data/steam_games"  # 相对 data_dir 的默认 cs_lo (会被 task 覆盖)

    def __init__(self, data_dir: str = "./data/steam_raw"):
        self.data_dir = data_dir
        self._cache: Optional[List[Dict]] = None

    # ==================== 数据加载 (缓存优先) ====================

    def load_raw(self, file_path: str = None, max_games: int = None) -> List[Dict]:
        """加载已爬取的 Steam 游戏元数据缓存.

        Returns: list[dict], 每项含 appid/name/genres/developers/publishers/categories 等
        """
        if self._cache is not None and not file_path:
            items = self._cache
        else:
            path = file_path or os.path.join(self.data_dir, self.DEFAULT_RAW)
            if not os.path.exists(path):
                logger.warning(f"[SteamTools] 缓存不存在: {path}")
                return []
            with open(path, encoding="utf-8") as f:
                items = json.load(f)
            self._cache = items
            logger.info(f"[SteamTools] 加载缓存: {len(items)} 游戏 ({path})")
        if max_games:
            return items[:max_games]
        return items

    def get_cslo_dir(self) -> str:
        """返回 cs_lo 目录 (默认 ./data/steam_games)"""
        return "./data/steam_games"

    # ==================== 实时爬取能力 (可选, 默认不走) ====================

    def crawl_live(self, max_games: int = 10000) -> List[Dict]:
        """实时爬取 Steam 游戏 (Steam Spy 批量 + appdetails 补充).

        复用 scripts/expand_steam_10k.py 的已验证逻辑. 需要代理 (mihomo).
        注意: 较慢 (appdetails ~0.3s/个, 受 Steam 限速), 默认管线用 load_raw() 缓存.
        """
        try:
            from scripts.expand_steam_10k import fetch_steamspy, get_appdetails, build_game
        except Exception as e:
            logger.error(f"[SteamTools] 无法加载实时爬取模块 (需要 scripts/expand_steam_10k.py): {e}")
            logger.error(f"[SteamTools] 请改用 data_source='file' 走缓存")
            return []
        logger.info(f"[SteamTools] 开始实时爬取 (目标 {max_games}, 预计耗时较长)")
        spy = fetch_steamspy()
        cands = sorted(spy.keys(), key=lambda k: -spy[k].get("positive", 0))
        games = []
        for appid in cands:
            if len(games) >= max_games:
                break
            details = get_appdetails(int(appid))
            g = build_game(appid, spy, details)
            if g and g.get("genres"):
                games.append(g)
        logger.info(f"[SteamTools] 实时爬取完成: {len(games)} 游戏")
        return games
