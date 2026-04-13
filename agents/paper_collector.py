"""
论文收集Agent - 负责论文PDF下载和多模态数据收集

使用 pdf-extract-kit 进行多模态数据提取：
- 布局检测（标题、正文、图片、表格、公式等）
- OCR文字识别
- 公式检测与识别（转LaTeX）
- 图片/表格提取

输出结构:
    output_dir/
    └── {paper_name}/
        ├── {paper_name}.md     # Markdown文件
        ├── {paper_name}.npy    # NPY格式数据
        └── images/             # 图片文件夹
            ├── 0.png
            ├── 1.png
            └── ...
"""
from typing import Dict, Any, Optional, List
from openai import OpenAI
from loguru import logger
import os
import json

from .base_agent import BaseAgent


class PaperCollectorAgent(BaseAgent):
    """论文收集Agent - 第一个分管Agent"""

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
            agent_name="PaperCollectorAgent"
        )
        self.collected_papers: List[Dict] = []
        self.collected_multimodal_data: List[Dict] = []

    def execute(
        self,
        task: Dict[str, Any],
        upstream_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        执行论文收集任务

        Args:
            task: 任务参数，包含：
                - paper_sources: 论文来源列表（支持两种格式）
                    格式1（下载+提取）: [{"type": "arxiv", "id": "2301.07041"}]
                    格式2（仅提取本地PDF）: [{"type": "local", "path": "/path/to/paper.pdf"}]
                - output_dir: 输出目录
                - extract_multimodal: 是否提取多模态数据
                - device: 计算设备 (cuda:0 / cpu)
                - batch_download: 是否批量并行下载（默认True）
            upstream_data: 上游数据（此Agent为第一个，通常为None）

        Returns:
            Dict[str, Any]: 收集结果
        """
        logger.info(f"[{self.agent_name}] 开始执行论文收集任务")

        paper_sources = task.get("paper_sources", [])
        output_dir = task.get("output_dir", "./data/papers")
        extract_multimodal = task.get("extract_multimodal", True)
        device = task.get("device", "cuda:0")
        batch_download = task.get("batch_download", True)  # 默认启用批量下载

        results = {
            "status": "success",
            "papers": [],
            "multimodal_data": [],
            "extracted_folders": [],
            "errors": []
        }

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 分离arxiv论文和其他来源
        arxiv_sources = [s for s in paper_sources if s.get("type") == "arxiv"]
        other_sources = [s for s in paper_sources if s.get("type") != "arxiv"]

        # 1. 批量并行下载arXiv论文
        if arxiv_sources:
            arxiv_ids = [s.get("id") for s in arxiv_sources if s.get("id")]

            if batch_download and len(arxiv_ids) > 1:
                # 使用批量并行下载
                logger.info(f"[{self.agent_name}] 批量并行下载 {len(arxiv_ids)} 篇arXiv论文")

                download_result = self.call_mcp_tool(
                    "download_arxiv_papers_batch",
                    paper_ids=arxiv_ids,
                    output_dir=output_dir,
                    max_workers=min(5, len(arxiv_ids))
                )

                if download_result:
                    # 记录下载成功的论文
                    for paper_id in download_result.get("success", []) + download_result.get("cached", []):
                        source = next((s for s in arxiv_sources if s.get("id") == paper_id), {})
                        paper_path = os.path.join(output_dir, paper_id, f"{paper_id}.pdf")
                        results["papers"].append({
                            "source": source,
                            "path": paper_path,
                            "arxiv_id": paper_id,
                            "name": paper_id,
                            "type": "arxiv",
                            "title": source.get("title", ""),
                            "cached": paper_id in download_result.get("cached", [])
                        })

                    # 记录失败的论文
                    for failed in download_result.get("failed", []):
                        results["errors"].append(f"下载失败 {failed['paper_id']}: {failed.get('error', 'Unknown')}")

                    logger.info(f"[{self.agent_name}] 下载完成: 成功{len(download_result.get('success', []))}, "
                               f"缓存{len(download_result.get('cached', []))}, "
                               f"失败{len(download_result.get('failed', []))}")
            else:
                # 单篇下载模式（兼容旧逻辑）
                for source in arxiv_sources:
                    paper_result = self._collect_single_paper(source, output_dir)
                    if paper_result:
                        results["papers"].append(paper_result)

        # 2. 处理其他来源（local, doi等）
        for source in other_sources:
            try:
                source_type = source.get("type", "local")

                if source_type == "local":
                    pdf_path = source.get("path", "")
                    if not os.path.exists(pdf_path):
                        results["errors"].append(f"PDF文件不存在: {pdf_path}")
                        continue

                    paper_name = source.get("name") or os.path.splitext(os.path.basename(pdf_path))[0]
                    results["papers"].append({
                        "source": source,
                        "path": pdf_path,
                        "name": paper_name,
                        "type": "local"
                    })
                else:
                    paper_result = self._collect_single_paper(source, output_dir)
                    if paper_result:
                        results["papers"].append(paper_result)

            except Exception as e:
                results["errors"].append(f"处理论文失败: {str(e)}")

        # 3. 批量提取多模态数据（按顺序，因为GPU操作）
        if extract_multimodal:
            total = len(results["papers"])
            logger.info(f"[{self.agent_name}] 开始提取 {total} 篇论文的多模态数据")

            for i, paper in enumerate(results["papers"], 1):
                pdf_path = paper.get("path")
                if not pdf_path or not os.path.exists(pdf_path):
                    continue

                paper_name = paper.get("name") or paper.get("arxiv_id", f"paper_{i}")
                logger.info(f"[{self.agent_name}] [{i}/{total}] 提取: {paper_name}")

                try:
                    multimodal_result = self._extract_multimodal_from_pdf(
                        pdf_path=pdf_path,
                        output_dir=output_dir,
                        paper_name=paper_name,
                        device=device
                    )
                    if multimodal_result.get("status") == "success":
                        results["multimodal_data"].append(multimodal_result)
                        results["extracted_folders"].append(multimodal_result.get("output_folder"))
                        # 添加markdown路径
                        paper["markdown_path"] = os.path.join(
                            multimodal_result.get("output_folder", ""),
                            f"{paper_name}.md"
                        )
                    else:
                        results["errors"].append(f"多模态提取失败 {paper_name}: {multimodal_result.get('message', '未知错误')}")
                except Exception as e:
                    results["errors"].append(f"多模态提取异常 {paper_name}: {str(e)}")

        # 保存收集结果
        self.collected_papers = results["papers"]
        self.collected_multimodal_data = results["multimodal_data"]

        # 使用LLM总结收集结果
        summary = self._summarize_collection(results)
        results["summary"] = summary

        logger.info(f"[{self.agent_name}] 论文收集完成: {len(results['papers'])}篇论文, {len(results['extracted_folders'])}个提取文件夹")

        return results

    def _collect_single_paper(self, source: Dict, output_dir: str) -> Optional[Dict]:
        """
        收集单篇论文（从arxiv/doi下载）

        Args:
            source: 论文来源信息
            output_dir: 输出目录

        Returns:
            Optional[Dict]: 论文收集结果
        """
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

    def _extract_multimodal_from_pdf(
        self,
        pdf_path: str,
        output_dir: str,
        paper_name: str = None,
        device: str = "cuda:0"
    ) -> Dict[str, Any]:
        """
        从PDF提取多模态数据（调用pdf-extract-kit）

        Args:
            pdf_path: PDF文件路径
            output_dir: 输出根目录
            paper_name: 论文名称（用于命名文件夹）
            device: 计算设备

        Returns:
            Dict: 提取结果，包含:
                - status: 'success' 或 'error'
                - paper_name: 论文名称
                - output_folder: 输出文件夹路径
                - md_path: Markdown文件路径
                - npy_path: NPY文件路径
                - images_folder: 图片文件夹路径
                - image_count: 图片数量
                - markdown_content: Markdown内容（部分）
        """
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
        """使用LLM总结收集结果"""
        prompt = f"""请总结以下论文收集结果：

论文数量: {len(results['papers'])}
提取文件夹数量: {len(results['extracted_folders'])}
错误数量: {len(results['errors'])}

论文列表:
{json.dumps([{'name': p.get('name', p.get('title', '')), 'type': p.get('type')} for p in results['papers']], ensure_ascii=False, indent=2)}

提取文件夹:
{json.dumps(results['extracted_folders'], ensure_ascii=False, indent=2)}

请简要总结收集的内容和数据质量。"""

        return self.generate(prompt, temperature=0.3, max_tokens=500)

    def get_collected_papers(self) -> List[Dict]:
        """获取已收集的论文列表"""
        return self.collected_papers

    def get_multimodal_data(self) -> List[Dict]:
        """获取已收集的多模态数据"""
        return self.collected_multimodal_data

    def get_extracted_folders(self) -> List[str]:
        """获取提取的文件夹路径列表"""
        return [d.get("output_folder") for d in self.collected_multimodal_data if d.get("output_folder")]
