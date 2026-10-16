"""
WikiArt 画作数据采集工具集 (仿 tools/amazon_tools.py 的 ToolManager 模式)

功能:
- load_raw(): 加载已爬取+enriched 的 WikiArt 画作元数据缓存 (wikiart_v2_enriched.json)
- crawl_live(): 实时爬取能力 (style-pages + artist bio), 默认管线不走此路径

设计: 默认 data_source="file" 走缓存; live 能力保留以支撑论文主张。
id 清洗统一用 crawlers.wikiart.sanitize_pid。
"""
import os
import json
from typing import Dict, List, Optional
from loguru import logger


class WikiartToolManager:
    """WikiArt 画作元数据工具管理器"""

    DEFAULT_RAW = "wikiart_v2_enriched.json"

    def __init__(self, data_dir: str = "./data/wikiart_raw"):
        self.data_dir = data_dir
        self._cache: Optional[List[Dict]] = None

    def load_raw(self, file_path: str = None, max_artworks: int = None) -> List[Dict]:
        """加载已爬取+enriched 的 WikiArt 画作元数据缓存.

        Returns: list[dict], 每项含 painting_id/title/artist/style 等
        """
        if self._cache is not None and not file_path:
            items = self._cache
        else:
            path = file_path or os.path.join(self.data_dir, self.DEFAULT_RAW)
            if not os.path.exists(path):
                logger.warning(f"[WikiartTools] 缓存不存在: {path}")
                return []
            with open(path, encoding="utf-8") as f:
                items = json.load(f)
            self._cache = items
            logger.info(f"[WikiartTools] 加载缓存: {len(items)} 画作 ({path})")
        if max_artworks:
            return items[:max_artworks]
        return items

    def get_cslo_dir(self) -> str:
        return "./data/wikiart_artworks_v2"

    def crawl_live(self, max_artworks: int = 22000) -> List[Dict]:
        """实时爬取 WikiArt (style-pages + artist bio enrich). 需代理."""
        try:
            from scripts.rebuild_wikiart_v2 import crawl_all  # 复用已验证逻辑
            from scripts.enrich_wikiart_text import main as enrich_main
        except Exception as e:
            logger.error(f"[WikiartTools] 无法加载实时爬取模块: {e}; 请改用 data_source='file'")
            return []
        logger.info(f"[WikiartTools] 开始实时爬取 (目标 {max_artworks}, 含 artist bio enrich)")
        return crawl_all(max_artworks)[:max_artworks]
