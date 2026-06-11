"""
ArXiv批量下载Agent - ReAct版
从arxiv元数据中筛选论文并生成下载任务，支持LLM驱动的迭代决策
"""
from typing import Dict, Any, Optional, List
from openai import OpenAI
from loguru import logger
import json
import os

from .react_agent import ReActAgent
from .messages import Blackboard, MessageBus, Message


class ArxivBatchDownloaderAgent(ReActAgent):
    """ArXiv批量下载Agent — ReAct循环驱动"""

    def __init__(self, client: OpenAI, system_prompt: str,
                 mcp_tool_manager=None, blackboard=None, message_bus=None):
        super().__init__(
            client=client,
            system_prompt=system_prompt,
            mcp_tool_manager=mcp_tool_manager,
            agent_name="ArxivBatchDownloaderAgent",
            blackboard=blackboard,
            message_bus=message_bus
        )
        self.filtered_papers: List[Dict] = []

    # ==================== ReAct 接口实现 ====================

    def get_available_actions(self) -> List[str]:
        return [
            "filter_metadata",          # 按分类筛选元数据
            "pass_through_sources",     # 直接传递已有的paper_sources
            "validate_results",         # 验证筛选结果
            "expand_search",            # 扩展搜索（放宽条件）
            "generate_sources",         # 生成paper_sources列表
            "done"                      # 完成
        ]

    def is_done(self, observation: Dict) -> bool:
        results = observation.get("accumulated_results", {})
        return bool(results.get("paper_sources"))

    def observe(self, task: Dict, blackboard: Blackboard,
                messages: List[Message]) -> Dict[str, Any]:
        observation = super().observe(task, blackboard, messages)

        # 从blackboard读取
        if blackboard:
            observation["filtered_count"] = blackboard.read("downloader", "filtered_count", 0)
            observation["paper_sources_count"] = blackboard.read("downloader", "paper_sources_count", 0)

        # 读取任务参数
        metadata_path = task.get("metadata_path") or ""
        observation["category"] = task.get("category", "cs.AI")
        observation["limit"] = task.get("limit", 20)
        observation["metadata_exists"] = os.path.exists(metadata_path)
        observation["has_paper_sources"] = bool(task.get("paper_sources"))

        return observation

    def act(self, action_plan: Dict) -> Dict[str, Any]:
        action = action_plan.get("action", "")
        params = action_plan.get("params", {})

        if action == "filter_metadata":
            return self.action_filter_metadata(self._current_task)
        elif action == "pass_through_sources":
            return self.action_pass_through_sources(self._current_task)
        elif action == "validate_results":
            return self.action_validate_results()
        elif action == "expand_search":
            return self.action_expand_search(self._current_task)
        elif action == "generate_sources":
            return self.action_generate_sources()
        elif action == "done":
            return {"status": "success", "message": "任务完成"}

        return {"status": "error", "message": f"未知行动: {action}"}

    # ==================== 行动方法 ====================

    def action_filter_metadata(self, task: Dict) -> Dict[str, Any]:
        """按分类筛选元数据"""
        metadata_path = task.get("metadata_path")
        category = task.get("category", "cs.AI")
        limit = task.get("limit", 20)
        sort_by_date = task.get("sort_by_date", True)

        try:
            filtered = self._filter_papers_by_category(
                metadata_path=metadata_path,
                category=category,
                limit=limit,
                sort_by_date=sort_by_date
            )
            self.filtered_papers = filtered

            if self.blackboard:
                self.blackboard.write("downloader", "filtered_count", len(filtered))
                self.blackboard.write("downloader", "category", category)
                self.blackboard.write("downloader", "papers", filtered)

            logger.info(
                f"[{self.agent_name}] 筛选完成: {len(filtered)} 篇 {category} 论文"
            )
            return {
                "status": "success",
                "total_papers_found": len(filtered),
                "message": f"筛选到 {len(filtered)} 篇论文"
            }
        except Exception as e:
            error_msg = f"筛选失败: {str(e)}"
            logger.error(f"[{self.agent_name}] {error_msg}")
            return {"status": "error", "errors": [error_msg]}

    def action_validate_results(self) -> Dict[str, Any]:
        """验证筛选结果"""
        if not self.filtered_papers:
            return {
                "status": "warning",
                "message": "没有筛选到论文，可能需要扩展搜索",
                "needs_expand": True
            }

        dates = [p.get("update_date", "unknown") for p in self.filtered_papers]
        return {
            "status": "success",
            "message": f"验证通过: {len(self.filtered_papers)} 篇，日期范围 {dates[-1]} ~ {dates[0]}"
        }

    def action_expand_search(self, task: Dict) -> Dict[str, Any]:
        """扩展搜索：放宽分类限制或增加数量"""
        category = task.get("category", "cs.AI")
        limit = task.get("limit", 20)

        # 尝试使用更宽泛的分类（去掉子分类）
        broader_category = category.split(".")[0] if "." in category else category

        if broader_category != category:
            logger.info(f"[{self.agent_name}] 尝试更宽泛的分类: {broader_category}")
            try:
                filtered = self._filter_papers_by_category(
                    metadata_path=task.get("metadata_path"),
                    category=broader_category,
                    limit=limit,
                    sort_by_date=True
                )
                self.filtered_papers = filtered

                if self.blackboard:
                    self.blackboard.write("downloader", "filtered_count", len(filtered))
                    self.blackboard.write("downloader", "papers", filtered)

                return {
                    "status": "success",
                    "total_papers_found": len(filtered),
                    "message": f"扩展搜索到 {broader_category}: {len(filtered)} 篇"
                }
            except Exception as e:
                return {"status": "error", "errors": [str(e)]}

        return {"status": "warning", "message": "无法进一步扩展搜索"}

    def action_pass_through_sources(self, task: Dict) -> Dict[str, Any]:
        """直接传递任务中已有的paper_sources到blackboard"""
        paper_sources = task.get("paper_sources", [])
        if not paper_sources:
            return {
                "status": "error",
                "message": "任务中没有paper_sources可传递"
            }

        if self.blackboard:
            self.blackboard.write("downloader", "paper_sources", paper_sources)
            self.blackboard.write("downloader", "paper_sources_count", len(paper_sources))

        logger.info(
            f"[{self.agent_name}] 直接传递 {len(paper_sources)} 个paper_sources"
        )
        return {
            "status": "success",
            "paper_sources": paper_sources,
            "total_papers_found": len(paper_sources),
            "message": f"传递了 {len(paper_sources)} 个论文源"
        }

    def action_generate_sources(self) -> Dict[str, Any]:
        """生成paper_sources列表供PaperCollectorAgent使用"""
        paper_sources = []
        for paper in self.filtered_papers:
            paper_source = {
                "type": "arxiv",
                "id": paper["id"],
                "title": paper.get("title", ""),
                "authors": paper.get("authors", ""),
                "categories": paper.get("categories", ""),
                "abstract": paper.get("abstract", "")[:500] if paper.get("abstract") else ""
            }
            paper_sources.append(paper_source)

        if self.blackboard:
            self.blackboard.write("downloader", "paper_sources", paper_sources)
            self.blackboard.write("downloader", "paper_sources_count", len(paper_sources))

        logger.info(f"[{self.agent_name}] 生成 {len(paper_sources)} 个paper_sources")

        # LLM总结
        summary = self._summarize_filtering({
            "status": "success",
            "category": self._current_task.get("category", ""),
            "total_papers_found": len(self.filtered_papers),
            "papers_to_download": self.filtered_papers[:5]
        })

        return {
            "status": "success",
            "paper_sources": paper_sources,
            "total_papers_found": len(self.filtered_papers),
            "summary": summary
        }

    # ==================== execute 入口（兼容旧接口）====================

    def execute(self, task: Dict[str, Any],
                upstream_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        执行入口 — 兼容旧接口
        如果有blackboard和message_bus，走ReAct循环；否则走原始逻辑
        """
        self._current_task = task

        if self.blackboard and self.message_bus:
            return self.run_react_loop(task, self.blackboard, self.message_bus)

        # 兼容模式：保持原始逻辑
        return self._execute_legacy(task)

    def _execute_legacy(self, task: Dict) -> Dict:
        """原始执行逻辑（向后兼容）"""
        logger.info(f"[{self.agent_name}] 开始执行ArXiv批量下载任务")

        analysis = self.reason_about_task(task)
        logger.info(f"[{self.agent_name}] 任务理解: {analysis.get('understanding', '')}")

        metadata_path = task.get("metadata_path")
        category = task.get("category", "cs.AI")
        limit = task.get("limit", 20)

        results = {
            "status": "success",
            "category": category,
            "total_papers_found": 0,
            "papers_to_download": [],
            "paper_sources": [],
            "errors": []
        }

        try:
            filtered = self._filter_papers_by_category(
                metadata_path=metadata_path, category=category,
                limit=limit, sort_by_date=task.get("sort_by_date", True)
            )
            results["total_papers_found"] = len(filtered)
            results["papers_to_download"] = filtered
            self.filtered_papers = filtered
        except Exception as e:
            results["status"] = "error"
            results["errors"].append(str(e))
            return results

        for paper in filtered:
            results["paper_sources"].append({
                "type": "arxiv",
                "id": paper["id"],
                "title": paper.get("title", ""),
                "authors": paper.get("authors", ""),
                "categories": paper.get("categories", ""),
                "abstract": paper.get("abstract", "")[:500] if paper.get("abstract") else ""
            })

        evaluation = self.evaluate_own_results(results)
        results["summary"] = self._summarize_filtering(results)

        return results

    # ==================== 核心逻辑（保持不变）====================

    def _filter_papers_by_category(self, metadata_path: str, category: str,
                                    limit: int, sort_by_date: bool = True) -> List[Dict]:
        """从元数据文件中筛选指定分类的论文（逻辑不变）"""
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")

        filtered = []
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    paper = json.loads(line)
                    categories = paper.get("categories", "")
                    primary_category = categories.split()[0] if categories.split() else ""
                    if primary_category == category:
                        filtered.append(paper)
                        if len(filtered) % 1000 == 0:
                            logger.debug(f"[{self.agent_name}] 已收集 {len(filtered)} 篇...")
                except json.JSONDecodeError as e:
                    logger.warning(f"[{self.agent_name}] 第 {line_num} 行JSON解析失败: {e}")

        logger.info(f"[{self.agent_name}] 共收集到 {len(filtered)} 篇 {category} 论文")

        if sort_by_date:
            def get_date(paper):
                update_date = paper.get("update_date", "")
                if update_date:
                    return update_date
                versions = paper.get("versions", [])
                if versions:
                    return versions[-1].get("created", "")
                return "1970-01-01"
            filtered.sort(key=get_date, reverse=True)

        result = filtered[:limit]
        if result:
            dates = [p.get("update_date", "unknown") for p in result]
            logger.info(f"[{self.agent_name}] 日期范围: {dates[-1]} ~ {dates[0]}")

        return result

    def _summarize_filtering(self, results: Dict) -> str:
        """使用LLM总结筛选结果"""
        papers = results.get("papers_to_download", [])[:5]
        paper_list = [
            {"id": p.get("id"), "title": p.get("title", "")[:100]}
            for p in papers
        ]

        prompt = f"""请总结以下ArXiv论文筛选结果：
分类: {results.get('category', '')}
总共筛选到: {results.get('total_papers_found', 0)} 篇论文
前5篇论文: {json.dumps(paper_list, ensure_ascii=False, indent=2)}
请简要说明筛选的内容和接下来的下载任务。"""

        return self.generate(prompt, temperature=0.3, max_tokens=500)

    def get_filtered_papers(self) -> List[Dict]:
        return self.filtered_papers

    def get_paper_sources(self) -> List[Dict]:
        return [{
            "type": "arxiv",
            "id": p["id"],
            "title": p.get("title", "")
        } for p in self.filtered_papers]
