"""
TMDb 电影数据采集工具集 (仿 tools/amazon_tools.py 的 ToolManager 模式)

功能:
- load_raw(): 加载已爬取的 TMDb 电影元数据缓存 (tmdb_movies_raw.json)
- crawl_live(): 实时爬取能力 (TMDb API discover + details + credits), 默认管线不走此路径

设计: 默认 data_source="file" 走缓存; live 能力保留以支撑论文主张与可复现性。
"""
import os
import json
from typing import Dict, List, Optional
from loguru import logger


class TmdbToolManager:
    """TMDb 电影元数据工具管理器"""

    DEFAULT_RAW = "tmdb_movies_raw.json"

    def __init__(self, data_dir: str = "./data/tmdb_raw", api_key: str = None):
        self.data_dir = data_dir
        self.api_key = api_key or os.getenv("TMDB_API_KEY", "")
        self._cache: Optional[List[Dict]] = None

    def load_raw(self, file_path: str = None, max_movies: int = None) -> List[Dict]:
        """加载已爬取的 TMDb 电影元数据缓存.

        Returns: list[dict], 每项含 movie_id/title/genres/directors/actors 等
        """
        if self._cache is not None and not file_path:
            items = self._cache
        else:
            path = file_path or os.path.join(self.data_dir, self.DEFAULT_RAW)
            if not os.path.exists(path):
                logger.warning(f"[TmdbTools] 缓存不存在: {path}")
                return []
            with open(path, encoding="utf-8") as f:
                items = json.load(f)
            self._cache = items
            logger.info(f"[TmdbTools] 加载缓存: {len(items)} 电影 ({path})")
        if max_movies:
            return items[:max_movies]
        return items

    def get_cslo_dir(self) -> str:
        return "./data/tmdb_movies"

    def crawl_live(self, max_movies: int = 13000) -> List[Dict]:
        """实时爬取 TMDb 电影 (discover + /movie/{id} + /movie/{id}/credits). 需 TMDB_API_KEY."""
        if not self.api_key:
            logger.error("[TmdbTools] 缺少 TMDB_API_KEY, 无法实时爬取; 请改用 data_source='file'")
            return []
        try:
            from scripts.rebuild_tmdb_cslo import crawl_movies  # 复用已验证逻辑
        except Exception as e:
            logger.error(f"[TmdbTools] 无法加载实时爬取模块: {e}")
            return []
        logger.info(f"[TmdbTools] 开始实时爬取 (目标 {max_movies})")
        return crawl_movies(self.api_key, max_movies)[:max_movies]
