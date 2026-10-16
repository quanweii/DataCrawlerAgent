"""
论文收集Agent - ReAct版
下载论文PDF并用pdf-extract-kit提取多模态数据
支持LLM驱动的迭代决策：下载失败→重试，提取失败→诊断
"""
from typing import Dict, Any, Optional, List
from openai import OpenAI
from loguru import logger
import os
import json

from .react_agent import ReActAgent
from .messages import Blackboard, MessageBus, Message


class PaperCollectorAgent(ReActAgent):
    """论文收集Agent — ReAct循环驱动"""

    def __init__(self, client: OpenAI, system_prompt: str,
                 mcp_tool_manager=None, blackboard=None, message_bus=None):
        super().__init__(
            client=client,
            system_prompt=system_prompt,
            mcp_tool_manager=mcp_tool_manager,
            agent_name="PaperCollectorAgent",
            blackboard=blackboard,
            message_bus=message_bus
        )
        self.collected_papers: List[Dict] = []
        self.collected_multimodal_data: List[Dict] = []

    # ==================== ReAct 接口实现 ====================

    def get_available_actions(self) -> List[str]:
        return [
            "download_batch",         # 批量并行下载PDF
            "download_single",        # 单篇下载（重试用）
            "extract_multimodal",     # 顺序提取多模态数据
            "extract_batch_gpu",      # 多GPU并行提取
            "retry_failed_downloads", # 重试失败的下载
            "validate_pdfs",          # 验证PDF文件
            "done"                    # 完成
        ]

    def is_done(self, observation: Dict) -> bool:
        results = observation.get("accumulated_results", {})
        papers = results.get("papers", [])
        need_extraction = observation.get("extract_multimodal", True)

        if not papers:
            return False

        total = observation.get("total_sources", 0)
        downloaded = len(papers)
        errors_count = len(results.get("errors", []))

        # 下载已完成（下载数 >= 来源数，考虑去重后可能有更多）
        download_done = (total > 0 and downloaded >= total) or (downloaded + errors_count >= total)

        if not download_done:
            return False

        # 不需要提取时直接完成
        if not need_extraction:
            return True

        # 需要多模态提取时，必须等提取完成
        extracted = len(results.get("extracted_folders", []))
        return extracted + errors_count >= downloaded

    def _default_action(self, observation: Dict) -> Dict[str, Any]:
        """
        LLM决策失败时，根据当前进度智能选择默认行动：
        - 下载未完成 → download_batch
        - 下载已完成但提取未开始 → extract_multimodal
        - 全部完成 → done
        """
        results = observation.get("accumulated_results", {})
        downloaded = len(results.get("papers", []))
        extracted = len(results.get("extracted_folders", []))
        total = observation.get("total_sources", 0)
        need_extraction = observation.get("extract_multimodal", True)

        # 下载已完成
        if total > 0 and downloaded >= total:
            if need_extraction and extracted < downloaded:
                return {
                    "action": "extract_multimodal",
                    "params": {},
                    "reasoning": "下载已完成，开始多模态提取"
                }
            return {"action": "done", "params": {}, "reasoning": "所有任务已完成"}

        # 下载未完成
        return {
            "action": "download_batch",
            "params": {},
            "reasoning": "下载未完成，继续批量下载"
        }

    def observe(self, task: Dict, blackboard: Blackboard,
                messages: List[Message]) -> Dict[str, Any]:
        observation = super().observe(task, blackboard, messages)

        # 读取任务参数
        paper_sources = self._normalize_sources(task.get("paper_sources", []))
        observation["total_sources"] = len(paper_sources)
        observation["extract_multimodal"] = task.get("extract_multimodal", True)
        observation["parallel_extraction"] = task.get("parallel_extraction", False)
        observation["num_gpus"] = task.get("num_gpus", 8)

        # 从blackboard读取downloader结果
        if blackboard:
            dl_papers = blackboard.read("downloader", "papers", [])
            if dl_papers and not paper_sources:
                observation["paper_sources_from_blackboard"] = len(dl_papers)

        # 统计当前状态
        results = observation.get("accumulated_results", {})
        downloaded = len(results.get("papers", []))
        extracted = len(results.get("extracted_folders", []))
        errors = len(results.get("errors", []))
        observation["downloaded_count"] = downloaded
        observation["extracted_count"] = extracted
        observation["error_count"] = errors
        observation["failed_downloads"] = results.get("failed_downloads", [])

        # 生成清晰的进度摘要，供 think() 使用
        observation["progress_summary"] = (
            f"总计{observation['total_sources']}篇，"
            f"已下载{downloaded}篇，"
            f"已提取{extracted}篇，"
            f"错误{errors}个"
        )

        # 明确标记当前阶段
        if downloaded >= observation["total_sources"] and extracted < downloaded:
            observation["current_phase"] = "extraction_needed"
            observation["phase_hint"] = (
                "下载阶段已完成！下一步必须选择 extract_multimodal "
                "进行多模态提取，不要再重复下载。"
            )
        elif extracted >= downloaded and downloaded > 0:
            observation["current_phase"] = "all_done"
            observation["phase_hint"] = "下载和提取均已完成，选择 done 结束。"
        else:
            observation["current_phase"] = "downloading"
            observation["phase_hint"] = "论文尚未下载完成，需要继续下载。"

        return observation

    def act(self, action_plan: Dict) -> Dict[str, Any]:
        action = action_plan.get("action", "")
        params = action_plan.get("params", {})

        if action == "download_batch":
            return self.action_download_batch(self._current_task)
        elif action == "download_single":
            return self.action_download_single(
                params.get("source"), params.get("output_dir", "./data/papers")
            )
        elif action == "extract_multimodal":
            return self.action_extract_multimodal()
        elif action == "extract_batch_gpu":
            return self.action_extract_batch_gpu(self._current_task)
        elif action == "retry_failed_downloads":
            return self.action_retry_failed_downloads()
        elif action == "validate_pdfs":
            return self.action_validate_pdfs()
        elif action == "done":
            return {"status": "success", "message": "收集完成"}

        return {"status": "error", "message": f"未知行动: {action}"}

    # ==================== 行动方法 ====================

    @staticmethod
    def _normalize_sources(paper_sources):
        """将 paper_sources 统一为 dict 列表，兼容各种输入格式"""
        if isinstance(paper_sources, str):
            try:
                paper_sources = json.loads(paper_sources)
            except (json.JSONDecodeError, ValueError):
                paper_sources = [{"type": "arxiv", "id": paper_sources.strip()}]

        if not isinstance(paper_sources, list):
            logger.warning(
                f"[PaperCollectorAgent] paper_sources类型异常: "
                f"{type(paper_sources).__name__}, 值: {str(paper_sources)[:100]}"
            )
            return []

        normalized = []
        for s in paper_sources:
            if isinstance(s, dict):
                normalized.append(s)
            elif isinstance(s, str):
                normalized.append({"type": "arxiv", "id": s.strip()})
        return normalized

    def action_download_batch(self, task: Dict) -> Dict[str, Any]:
        """批量并行下载PDF"""
        paper_sources = self._normalize_sources(task.get("paper_sources", []))
        output_dir = task.get("output_dir", "./data/papers")
        os.makedirs(output_dir, exist_ok=True)

        # 检查是否已经下载过（避免重复下载）
        already_downloaded = {p.get("arxiv_id") or p.get("name")
                              for p in self._accumulated_results.get("papers", [])}
        pending_sources = [s for s in paper_sources
                           if s.get("id") not in already_downloaded]

        if not pending_sources:
            logger.info(f"[{self.agent_name}] 所有论文已下载，跳过重复下载")
            return {"status": "success", "papers": [], "errors": [],
                    "message": "所有论文已下载，无需重复操作"}

        arxiv_sources = [s for s in pending_sources if s.get("type") == "arxiv"]
        other_sources = [s for s in pending_sources if s.get("type") != "arxiv"]

        results = {"status": "success", "papers": [], "errors": []}

        # 批量下载arxiv论文
        if arxiv_sources:
            arxiv_ids = [s.get("id") for s in arxiv_sources if s.get("id")]
            if len(arxiv_ids) > 1:
                logger.info(f"[{self.agent_name}] 批量下载 {len(arxiv_ids)} 篇arXiv论文")
                download_result = self.call_mcp_tool(
                    "download_arxiv_papers_batch",
                    paper_ids=arxiv_ids,
                    output_dir=output_dir,
                    max_workers=min(10, len(arxiv_ids))
                )
                if download_result:
                    for paper_id in download_result.get("success", []) + download_result.get("cached", []):
                        source = next((s for s in arxiv_sources if s.get("id") == paper_id), {})
                        results["papers"].append({
                            "source": source,
                            "path": os.path.join(output_dir, paper_id, f"{paper_id}.pdf"),
                            "arxiv_id": paper_id,
                            "name": paper_id,
                            "type": "arxiv",
                            "title": source.get("title", ""),
                            "cached": paper_id in download_result.get("cached", [])
                        })
                    for failed in download_result.get("failed", []):
                        results["errors"].append(
                            f"下载失败 {failed['paper_id']}: {failed.get('error', 'Unknown')}"
                        )
                    results["failed_downloads"] = [
                        f["paper_id"] for f in download_result.get("failed", [])
                    ]
            else:
                for source in arxiv_sources:
                    paper_result = self._collect_single_paper(source, output_dir)
                    if paper_result:
                        results["papers"].append(paper_result)

        # 处理其他来源
        for source in other_sources:
            try:
                source_type = source.get("type", "local")
                if source_type == "local":
                    pdf_path = source.get("path", "")
                    if os.path.exists(pdf_path):
                        paper_name = source.get("name") or os.path.splitext(os.path.basename(pdf_path))[0]
                        results["papers"].append({
                            "source": source, "path": pdf_path,
                            "name": paper_name, "type": "local"
                        })
                    else:
                        results["errors"].append(f"PDF文件不存在: {pdf_path}")
                else:
                    paper_result = self._collect_single_paper(source, output_dir)
                    if paper_result:
                        results["papers"].append(paper_result)
            except Exception as e:
                results["errors"].append(f"处理论文失败: {str(e)}")

        if self.blackboard:
            self.blackboard.write("collector", "papers", results["papers"])
            self.blackboard.write("collector", "download_errors", results["errors"])

        logger.info(
            f"[{self.agent_name}] 下载完成: {len(results['papers'])} 成功, "
            f"{len(results['errors'])} 失败"
        )
        return results

    def action_download_single(self, source: Dict, output_dir: str) -> Dict:
        """单篇下载（用于重试）"""
        if not source:
            return {"status": "error", "message": "未指定论文来源"}

        result = self._collect_single_paper(source, output_dir)
        if result:
            return {"status": "success", "paper": result}
        return {"status": "error", "message": f"下载失败: {source.get('id', 'unknown')}"}

    def action_extract_multimodal(self) -> Dict:
        """顺序提取多模态数据，跳过已提取的论文"""
        papers = self._accumulated_results.get("papers", [])
        if not papers:
            return {"status": "error", "message": "没有可提取的论文"}

        task = self._current_task
        output_dir = task.get("output_dir", "./data/papers")
        device = task.get("device", "cuda:0")

        # 获取已提取的论文集合，避免重复提取
        already_extracted = set(self._accumulated_results.get("extracted_folders", []))

        multimodal_data = []
        extracted_folders = []
        errors = []

        total = len(papers)
        for i, paper in enumerate(papers, 1):
            pdf_path = paper.get("path")
            if not pdf_path or not os.path.exists(pdf_path):
                continue

            paper_name = paper.get("name") or paper.get("arxiv_id", f"paper_{i}")
            paper_output_dir = os.path.join(output_dir, paper_name)

            # 跳过已提取的论文（检查输出文件夹是否存在且非空）
            if paper_output_dir in already_extracted or (
                os.path.isdir(paper_output_dir) and
                os.listdir(paper_output_dir) and
                os.path.exists(os.path.join(paper_output_dir, f"{paper_name}.md"))
            ):
                logger.info(f"[{self.agent_name}] [{i}/{total}] 跳过已提取: {paper_name}")
                # 补充到 extracted_folders 以防 _accumulate_result 去重丢失
                if paper_output_dir not in already_extracted:
                    extracted_folders.append(paper_output_dir)
                # 设置 markdown_path 供下游 graph builder 使用
                paper["markdown_path"] = os.path.join(
                    paper_output_dir, f"{paper_name}.md"
                )
                continue

            logger.info(f"[{self.agent_name}] [{i}/{total}] 提取: {paper_name}")

            try:
                multimodal_result = self._extract_multimodal_from_pdf(
                    pdf_path=pdf_path, output_dir=output_dir,
                    paper_name=paper_name, device=device
                )
                if multimodal_result.get("status") == "success":
                    multimodal_data.append(multimodal_result)
                    extracted_folders.append(multimodal_result.get("output_folder"))
                    paper["markdown_path"] = os.path.join(
                        multimodal_result.get("output_folder", ""),
                        f"{paper_name}.md"
                    )
                else:
                    errors.append(
                        f"多模态提取失败 {paper_name}: "
                        f"{multimodal_result.get('message', '未知错误')}"
                    )
            except Exception as e:
                errors.append(f"多模态提取异常 {paper_name}: {str(e)}")

        self.collected_multimodal_data = multimodal_data

        if self.blackboard:
            self.blackboard.write("collector", "multimodal_data", multimodal_data)
            self.blackboard.write("collector", "extracted_folders", extracted_folders)
            # 更新 papers 列表（包含提取后的 markdown_path）
            self.blackboard.write("collector", "papers", papers)

        return {
            "status": "success",
            "multimodal_data": multimodal_data,
            "extracted_folders": extracted_folders,
            "errors": errors
        }

    def action_extract_batch_gpu(self, task: Dict) -> Dict:
        """多GPU并行提取"""
        papers = self._accumulated_results.get("papers", [])
        if not papers:
            return {"status": "error", "message": "没有可提取的论文"}

        output_dir = task.get("output_dir", "./data/papers")
        num_gpus = task.get("num_gpus", 8)
        batch_size = task.get("batch_size", 50)

        from tools.gpu_pool import GPUExtractionPool

        extraction_tasks = []
        paper_index = []
        for i, paper in enumerate(papers):
            pdf_path = paper.get("path")
            if not pdf_path or not os.path.exists(pdf_path):
                continue
            paper_name = paper.get("name") or paper.get("arxiv_id", f"paper_{i}")
            extraction_tasks.append({
                "pdf_path": pdf_path,
                "output_dir": output_dir,
                "paper_name": paper_name
            })
            paper_index.append(i)

        progress_callback = task.get("progress_callback")
        pool = GPUExtractionPool(num_gpus=num_gpus, batch_size=batch_size)
        extraction_results = pool.extract_all(extraction_tasks, progress_callback=progress_callback)

        multimodal_data = []
        extracted_folders = []
        errors = []

        for idx, ext_result in zip(paper_index, extraction_results):
            paper = papers[idx]
            if ext_result.get("status") == "success":
                multimodal_data.append(ext_result)
                extracted_folders.append(ext_result.get("output_folder"))
                paper["markdown_path"] = os.path.join(
                    ext_result.get("output_folder", ""),
                    f"{paper.get('name', paper.get('arxiv_id', ''))}.md"
                )
            else:
                errors.append(
                    f"多模态提取失败 {ext_result.get('paper_name', '?')}: "
                    f"{ext_result.get('message', '未知错误')}"
                )

        self.collected_multimodal_data = multimodal_data

        if self.blackboard:
            self.blackboard.write("collector", "multimodal_data", multimodal_data)
            self.blackboard.write("collector", "extracted_folders", extracted_folders)
            # 更新 papers 列表（包含提取后的 markdown_path）
            self.blackboard.write("collector", "papers", papers)

        return {
            "status": "success",
            "multimodal_data": multimodal_data,
            "extracted_folders": extracted_folders,
            "errors": errors
        }

    def action_retry_failed_downloads(self) -> Dict:
        """重试失败的下载"""
        failed_ids = self._accumulated_results.get("failed_downloads", [])
        if not failed_ids:
            return {"status": "success", "message": "没有需要重试的下载"}

        output_dir = self._current_task.get("output_dir", "./data/papers")
        retried_papers = []
        still_failed = []

        for paper_id in failed_ids:
            logger.info(f"[{self.agent_name}] 重试下载: {paper_id}")
            result = self.call_mcp_tool(
                "download_arxiv_paper",
                paper_id=paper_id,
                save_path=os.path.join(output_dir, paper_id, f"{paper_id}.pdf")
            )
            if result and result.get("status") == "success":
                retried_papers.append({
                    "source": {"type": "arxiv", "id": paper_id},
                    "path": result.get("path"),
                    "arxiv_id": paper_id,
                    "name": paper_id,
                    "type": "arxiv",
                    "title": "",
                    "cached": result.get("cached", False)
                })
            else:
                still_failed.append(paper_id)

        # 更新累积结果
        self._accumulated_results.setdefault("papers", []).extend(retried_papers)
        self._accumulated_results["failed_downloads"] = still_failed

        return {
            "status": "success" if retried_papers else "error",
            "retried_success": len(retried_papers),
            "still_failed": len(still_failed),
            "papers": retried_papers
        }

    def action_validate_pdfs(self) -> Dict:
        """验证已下载的PDF文件"""
        papers = self._accumulated_results.get("papers", [])
        valid = 0
        invalid = []

        for paper in papers:
            pdf_path = paper.get("path", "")
            if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000:
                valid += 1
            else:
                invalid.append({
                    "name": paper.get("name", paper.get("arxiv_id", "")),
                    "path": pdf_path,
                    "reason": "文件不存在或过小"
                })

        if invalid and self.message_bus:
            self.message_bus.send(Message(
                sender=self.agent_name,
                recipient="broadcast",
                msg_type="quality_issue",
                content={"invalid_pdfs": invalid},
                priority=1
            ))

        return {
            "status": "success",
            "valid_count": valid,
            "invalid_count": len(invalid),
            "invalid_pdfs": invalid
        }

    # ==================== execute 入口（兼容旧接口）====================

    def execute(self, task: Dict[str, Any],
                upstream_data: Optional[Dict] = None) -> Dict[str, Any]:
        self._current_task = task

        if self.blackboard and self.message_bus:
            # ReAct模式
            result = self.run_react_loop(task, self.blackboard, self.message_bus)
            # 确保返回格式兼容
            self._ensure_compatible_result(result, task)
            return result

        # 兼容模式
        return self._execute_legacy(task, upstream_data)

    def _ensure_compatible_result(self, result: Dict, task: Dict):
        """确保ReAct结果与旧格式兼容"""
        if "papers" not in result:
            result["papers"] = self._accumulated_results.get("papers", [])
        if "multimodal_data" not in result:
            result["multimodal_data"] = self._accumulated_results.get("multimodal_data", [])
        if "extracted_folders" not in result:
            result["extracted_folders"] = self._accumulated_results.get("extracted_folders", [])
        if "errors" not in result:
            result["errors"] = self._accumulated_results.get("errors", [])

        self.collected_papers = result["papers"]
        self.collected_multimodal_data = result["multimodal_data"]

        evaluation = self.evaluate_own_results(result)
        result["summary"] = self._summarize_collection(result)

    def _execute_legacy(self, task: Dict, upstream_data=None) -> Dict:
        """原始执行逻辑（向后兼容）"""
        logger.info(f"[{self.agent_name}] 开始执行论文收集任务")

        analysis = self.reason_about_task(task)
        logger.info(f"[{self.agent_name}] 任务理解: {analysis.get('understanding', '')}")

        paper_sources = self._normalize_sources(task.get("paper_sources", []))
        output_dir = task.get("output_dir", "./data/papers")
        extract_multimodal = task.get("extract_multimodal", True)
        device = task.get("device", "cuda:0")
        batch_download = task.get("batch_download", True)

        results = {
            "status": "success",
            "papers": [],
            "multimodal_data": [],
            "extracted_folders": [],
            "errors": []
        }
        os.makedirs(output_dir, exist_ok=True)

        arxiv_sources = [s for s in paper_sources if s.get("type") == "arxiv"]
        other_sources = [s for s in paper_sources if s.get("type") != "arxiv"]

        if arxiv_sources:
            arxiv_ids = [s.get("id") for s in arxiv_sources if s.get("id")]
            if batch_download and len(arxiv_ids) > 1:
                download_result = self.call_mcp_tool(
                    "download_arxiv_papers_batch",
                    paper_ids=arxiv_ids,
                    output_dir=output_dir,
                    max_workers=min(10, len(arxiv_ids))
                )
                if download_result:
                    for paper_id in download_result.get("success", []) + download_result.get("cached", []):
                        source = next((s for s in arxiv_sources if s.get("id") == paper_id), {})
                        results["papers"].append({
                            "source": source,
                            "path": os.path.join(output_dir, paper_id, f"{paper_id}.pdf"),
                            "arxiv_id": paper_id, "name": paper_id,
                            "type": "arxiv", "title": source.get("title", ""),
                            "cached": paper_id in download_result.get("cached", [])
                        })
                    for failed in download_result.get("failed", []):
                        results["errors"].append(
                            f"下载失败 {failed['paper_id']}: {failed.get('error', 'Unknown')}"
                        )
            else:
                for source in arxiv_sources:
                    paper_result = self._collect_single_paper(source, output_dir)
                    if paper_result:
                        results["papers"].append(paper_result)

        for source in other_sources:
            try:
                source_type = source.get("type", "local")
                if source_type == "local":
                    pdf_path = source.get("path", "")
                    if os.path.exists(pdf_path):
                        paper_name = source.get("name") or os.path.splitext(os.path.basename(pdf_path))[0]
                        results["papers"].append({
                            "source": source, "path": pdf_path,
                            "name": paper_name, "type": "local"
                        })
                    else:
                        results["errors"].append(f"PDF文件不存在: {pdf_path}")
                else:
                    paper_result = self._collect_single_paper(source, output_dir)
                    if paper_result:
                        results["papers"].append(paper_result)
            except Exception as e:
                results["errors"].append(f"处理论文失败: {str(e)}")

        if extract_multimodal:
            total = len(results["papers"])
            parallel_extraction = task.get("parallel_extraction", False)
            num_gpus = task.get("num_gpus", 8)

            if parallel_extraction and total > 1:
                from tools.gpu_pool import GPUExtractionPool
                extraction_tasks = []
                paper_index = []
                for i, paper in enumerate(results["papers"]):
                    pdf_path = paper.get("path")
                    if not pdf_path or not os.path.exists(pdf_path):
                        continue
                    paper_name = paper.get("name") or paper.get("arxiv_id", f"paper_{i}")
                    extraction_tasks.append({
                        "pdf_path": pdf_path,
                        "output_dir": output_dir,
                        "paper_name": paper_name
                    })
                    paper_index.append(i)

                batch_size = task.get("batch_size", 50)
                progress_callback = task.get("progress_callback")
                pool = GPUExtractionPool(num_gpus=num_gpus, batch_size=batch_size)
                extraction_results = pool.extract_all(extraction_tasks, progress_callback=progress_callback)

                for idx, ext_result in zip(paper_index, extraction_results):
                    paper = results["papers"][idx]
                    if ext_result.get("status") == "success":
                        results["multimodal_data"].append(ext_result)
                        results["extracted_folders"].append(ext_result.get("output_folder"))
                        paper["markdown_path"] = os.path.join(
                            ext_result.get("output_folder", ""),
                            f"{paper.get('name', paper.get('arxiv_id', ''))}.md"
                        )
                    else:
                        results["errors"].append(
                            f"多模态提取失败 {ext_result.get('paper_name', '?')}: "
                            f"{ext_result.get('message', '未知错误')}"
                        )
            else:
                for i, paper in enumerate(results["papers"], 1):
                    pdf_path = paper.get("path")
                    if not pdf_path or not os.path.exists(pdf_path):
                        continue
                    paper_name = paper.get("name") or paper.get("arxiv_id", f"paper_{i}")
                    logger.info(f"[{self.agent_name}] [{i}/{total}] 提取: {paper_name}")
                    try:
                        multimodal_result = self._extract_multimodal_from_pdf(
                            pdf_path=pdf_path, output_dir=output_dir,
                            paper_name=paper_name, device=device
                        )
                        if multimodal_result.get("status") == "success":
                            results["multimodal_data"].append(multimodal_result)
                            results["extracted_folders"].append(multimodal_result.get("output_folder"))
                            paper["markdown_path"] = os.path.join(
                                multimodal_result.get("output_folder", ""),
                                f"{paper_name}.md"
                            )
                        else:
                            results["errors"].append(
                                f"多模态提取失败 {paper_name}: {multimodal_result.get('message', '未知错误')}"
                            )
                    except Exception as e:
                        results["errors"].append(f"多模态提取异常 {paper_name}: {str(e)}")

        self.collected_papers = results["papers"]
        self.collected_multimodal_data = results["multimodal_data"]

        evaluation = self.evaluate_own_results(results)
        results["summary"] = self._summarize_collection(results)

        return results

    # ==================== 辅助方法（保持不变）====================

    def _collect_single_paper(self, source: Dict, output_dir: str) -> Optional[Dict]:
        source_type = source.get("type", "arxiv")
        source_id = source.get("id", "")

        if source_type == "arxiv":
            result = self.call_mcp_tool(
                "download_arxiv_paper",
                paper_id=source_id,
                save_path=os.path.join(output_dir, f"{source_id}.pdf")
            )
        elif source_type == "doi":
            result = self.call_mcp_tool(
                "download_doi_paper",
                doi=source_id,
                save_path=os.path.join(output_dir, f"{source_id.replace('/', '_')}.pdf")
            )
        else:
            logger.warning(f"[{self.agent_name}] 未知的论文来源类型: {source_type}")
            return None

        if result and result.get("status") == "success":
            return {
                "source": source,
                "path": result.get("path"),
                "title": source.get("title", ""),
                "type": source_type
            }
        return None

    def _extract_multimodal_from_pdf(self, pdf_path: str, output_dir: str,
                                      paper_name: str = None,
                                      device: str = "cuda:0") -> Dict[str, Any]:
        logger.info(f"[{self.agent_name}] 提取多模态数据: {pdf_path}")

        result = self.call_mcp_tool(
            "extract_pdf_multimodal",
            pdf_path=pdf_path,
            output_dir=output_dir,
            paper_name=paper_name,
            device=device
        )

        if result and result.get("status") == "success":
            logger.info(f"[{self.agent_name}] 多模态提取成功: {result.get('image_count')}张图片")

        return result or {"status": "error", "message": "工具调用失败"}

    def _summarize_collection(self, results: Dict) -> str:
        prompt = f"""请总结以下论文收集结果：
论文数量: {len(results.get('papers', []))}
提取文件夹数量: {len(results.get('extracted_folders', []))}
错误数量: {len(results.get('errors', []))}
请简要总结收集的内容和数据质量。"""
        return self.generate(prompt, temperature=0.3, max_tokens=500)

    def get_collected_papers(self) -> List[Dict]:
        return self.collected_papers

    def get_multimodal_data(self) -> List[Dict]:
        return self.collected_multimodal_data

    def get_extracted_folders(self) -> List[str]:
        return [d.get("output_folder") for d in self.collected_multimodal_data if d.get("output_folder")]
