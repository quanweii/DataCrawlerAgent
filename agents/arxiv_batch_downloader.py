"""
ArXiv批量下载Agent - 从arxiv元数据中筛选并批量下载论文

功能:
- 读取arxiv元数据JSON文件
- 按分类筛选论文(如cs.AI)
- 批量生成下载任务
- 支持限制下载数量
"""
from typing import Dict, Any, Optional, List
from openai import OpenAI
from loguru import logger
import json
import os

from .base_agent import BaseAgent


class ArxivBatchDownloaderAgent(BaseAgent):
    """ArXiv批量下载Agent - 负责从元数据中筛选论文并生成下载任务"""

    def __init__(
        self,
        client: OpenAI,
        system_prompt: str,
        mcp_tool_manager: Optional[Any] = None
    ):
        super().__init__(
            client=client,
            system_prompt=system_prompt,
            mcp_tool_manager=mcp_tool_manager,
            agent_name="ArxivBatchDownloaderAgent"
        )
        self.filtered_papers: List[Dict] = []

    def execute(
        self,
        task: Dict[str, Any],
        upstream_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        执行批量下载任务

        Args:
            task: 任务参数，包含：
                - metadata_path: arxiv元数据JSON文件路径
                - category: 筛选的分类(如"cs.AI")
                - limit: 下载数量限制(默认20)
                - output_dir: PDF保存目录
                - extract_multimodal: 是否提取多模态数据
                - device: 计算设备
            upstream_data: 上游数据

        Returns:
            Dict[str, Any]: 包含paper_sources的结果，供PaperCollectorAgent使用
        """
        logger.info(f"[{self.agent_name}] 开始执行ArXiv批量下载任务")

        metadata_path = task.get("metadata_path", "/home/QuanWei/arxiv-metadata-oai-snapshot.json")
        category = task.get("category", "cs.AI")
        limit = task.get("limit", 20)
        output_dir = task.get("output_dir", "./data/papers")

        results = {
            "status": "success",
            "category": category,
            "total_papers_found": 0,
            "papers_to_download": [],
            "paper_sources": [],  # 供PaperCollectorAgent使用
            "errors": []
        }

        # 1. 读取并筛选元数据
        try:
            filtered_papers = self._filter_papers_by_category(
                metadata_path=metadata_path,
                category=category,
                limit=limit,
                sort_by_date=task.get("sort_by_date", True)  # 默认按日期排序
            )
            results["total_papers_found"] = len(filtered_papers)
            results["papers_to_download"] = filtered_papers
            self.filtered_papers = filtered_papers

            logger.info(f"[{self.agent_name}] 筛选到 {len(filtered_papers)} 篇 {category} 论文")

        except Exception as e:
            error_msg = f"筛选论文失败: {str(e)}"
            logger.error(f"[{self.agent_name}] {error_msg}")
            results["status"] = "error"
            results["errors"].append(error_msg)
            return results

        # 2. 生成paper_sources供PaperCollectorAgent使用
        for paper in filtered_papers:
            paper_source = {
                "type": "arxiv",
                "id": paper["id"],
                "title": paper.get("title", ""),
                "authors": paper.get("authors", ""),
                "categories": paper.get("categories", ""),
                "abstract": paper.get("abstract", "")[:500] if paper.get("abstract") else ""
            }
            results["paper_sources"].append(paper_source)

        # 3. 使用LLM总结筛选结果
        summary = self._summarize_filtering(results)
        results["summary"] = summary

        logger.info(f"[{self.agent_name}] 批量下载任务准备完成: {len(results['paper_sources'])} 篇论文")

        return results

    def _filter_papers_by_category(
        self,
        metadata_path: str,
        category: str,
        limit: int,
        sort_by_date: bool = True
    ) -> List[Dict]:
        """
        从元数据文件中筛选指定分类的论文

        Args:
            metadata_path: 元数据JSON文件路径
            category: 目标分类(如"cs.AI")
            limit: 最大数量
            sort_by_date: 是否按日期排序(最新的在前)

        Returns:
            List[Dict]: 筛选出的论文列表
        """
        filtered = []

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")

        logger.info(f"[{self.agent_name}] 读取元数据文件: {metadata_path}")

        # 1. 先收集所有匹配分类的论文
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    paper = json.loads(line)

                    # 检查分类是否匹配
                    categories = paper.get("categories", "")
                    # 分类可能是空格分隔的多个分类
                    if category in categories.split():
                        filtered.append(paper)

                        if len(filtered) % 1000 == 0:
                            logger.debug(f"[{self.agent_name}] 已收集 {len(filtered)} 篇论文...")

                except json.JSONDecodeError as e:
                    logger.warning(f"[{self.agent_name}] 第 {line_num} 行JSON解析失败: {e}")
                    continue

        logger.info(f"[{self.agent_name}] 共收集到 {len(filtered)} 篇 {category} 论文")

        # 2. 按日期排序(最新的在前)
        if sort_by_date:
            def get_date(paper):
                """获取论文的更新日期"""
                update_date = paper.get("update_date", "")
                if update_date:
                    return update_date
                # 如果没有update_date，尝试从versions获取
                versions = paper.get("versions", [])
                if versions:
                    # 取最后一个版本的日期
                    last_version = versions[-1].get("created", "")
                    return last_version
                return "1970-01-01"  # 默认最早日期

            filtered.sort(key=get_date, reverse=True)
            logger.info(f"[{self.agent_name}] 已按日期排序(最新优先)")

        # 3. 返回指定数量
        result = filtered[:limit]
        logger.info(f"[{self.agent_name}] 返回最近的 {len(result)} 篇论文")

        # 打印日期范围
        if result:
            dates = [p.get("update_date", "unknown") for p in result]
            logger.info(f"[{self.agent_name}] 日期范围: {dates[-1]} ~ {dates[0]}")

        return result

    def _summarize_filtering(self, results: Dict) -> str:
        """使用LLM总结筛选结果"""
        papers = results["papers_to_download"][:5]  # 只展示前5篇

        paper_list = []
        for p in papers:
            paper_list.append({
                "id": p.get("id"),
                "title": p.get("title", "")[:100],
                "authors": p.get("authors", "")[:50]
            })

        prompt = f"""请总结以下ArXiv论文筛选结果：

        分类: {results['category']}
        总共筛选到: {results['total_papers_found']} 篇论文

        前5篇论文:
        {json.dumps(paper_list, ensure_ascii=False, indent=2)}

        请简要说明筛选的内容和接下来将要进行的下载任务。"""

        return self.generate(prompt, temperature=0.3, max_tokens=500)

    def get_filtered_papers(self) -> List[Dict]:
        """获取已筛选的论文列表"""
        return self.filtered_papers

    def get_paper_sources(self) -> List[Dict]:
        """获取paper_sources列表，供PaperCollectorAgent使用"""
        return [{
            "type": "arxiv",
            "id": p["id"],
            "title": p.get("title", "")
        } for p in self.filtered_papers]
