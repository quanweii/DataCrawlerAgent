"""
MCP工具管理器 - 封装MCP工具调用
支持动态注册和调用各种MCP工具

包含:
- 论文下载工具 (arxiv, doi)
- PDF多模态提取工具 (pdf-extract-kit)
- 数据清洗工具
- 数据集封装工具
"""
from typing import Dict, Any, Optional, Callable, List
from loguru import logger
import json


class MCPToolManager:
    """MCP工具管理器，负责工具的注册、发现和调用"""

    def __init__(self):
        """初始化MCP工具管理器"""
        self._tools: Dict[str, Callable] = {}
        self._tool_schemas: Dict[str, Dict] = {}
        logger.info("MCP工具管理器初始化完成")

    def register_tool(
        self,
        name: str,
        func: Callable,
        description: str = "",
        parameters_schema: Optional[Dict] = None
    ):
        """
        注册MCP工具

        Args:
            name: 工具名称
            func: 工具函数
            description: 工具描述
            parameters_schema: 参数schema
        """
        self._tools[name] = func
        self._tool_schemas[name] = {
            "name": name,
            "description": description,
            "parameters": parameters_schema or {}
        }
        logger.info(f"已注册MCP工具: {name}")

    def unregister_tool(self, name: str):
        """注销工具"""
        if name in self._tools:
            del self._tools[name]
            del self._tool_schemas[name]
            logger.info(f"已注销MCP工具: {name}")

    def get_tool(self, name: str) -> Optional[Callable]:
        """获取工具函数"""
        return self._tools.get(name)

    def list_tools(self) -> Dict[str, Dict]:
        """列出所有已注册的工具"""
        return self._tool_schemas.copy()

    def call_tool(self, name: str, **kwargs) -> Any:
        """
        调用MCP工具

        Args:
            name: 工具名称
            **kwargs: 工具参数

        Returns:
            Any: 工具执行结果
        """
        if name not in self._tools:
            raise ValueError(f"工具 '{name}' 未注册")

        tool_func = self._tools[name]
        logger.debug(f"调用工具: {name}, 参数: {kwargs}")

        try:
            result = tool_func(**kwargs)
            logger.debug(f"工具 {name} 执行成功")
            return result
        except Exception as e:
            logger.error(f"工具 {name} 执行失败: {e}")
            raise

    # ==================== 预定义的MCP工具 ====================

    def register_paper_download_tools(self):
        """注册论文下载相关的MCP工具"""
        # arXiv单篇论文下载
        self.register_tool(
            name="download_arxiv_paper",
            func=self._download_arxiv_paper,
            description="从arXiv下载单篇论文PDF",
            parameters_schema={
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "arXiv论文ID"},
                    "save_path": {"type": "string", "description": "保存路径"}
                },
                "required": ["paper_id"]
            }
        )

        # arXiv批量并行下载
        self.register_tool(
            name="download_arxiv_papers_batch",
            func=self.download_arxiv_papers_batch,
            description="批量并行下载arXiv论文PDF，支持多线程",
            parameters_schema={
                "type": "object",
                "properties": {
                    "paper_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "arXiv论文ID列表"
                    },
                    "output_dir": {"type": "string", "description": "输出目录"},
                    "max_workers": {
                        "type": "integer",
                        "description": "最大并行数(建议3-5)",
                        "default": 5
                    }
                },
                "required": ["paper_ids", "output_dir"]
            }
        )

        # 示例：DOI论文下载
        self.register_tool(
            name="download_doi_paper",
            func=self._download_doi_paper,
            description="通过DOI下载论文PDF",
            parameters_schema={
                "type": "object",
                "properties": {
                    "doi": {"type": "string", "description": "论文DOI"},
                    "save_path": {"type": "string", "description": "保存路径"}
                },
                "required": ["doi"]
            }
        )

        logger.info("已注册论文下载相关工具")

    def register_multimodal_tools(self):
        """注册多模态数据收集相关的MCP工具"""
        # ==================== PDF多模态提取工具 (核心工具) ====================
        # 这是主要的PDF提取工具，调用pdf-extract-kit
        self.register_tool(
            name="extract_pdf_multimodal",
            func=self._extract_pdf_multimodal,
            description="从PDF论文中提取多模态数据（文字、公式、表格、图片），输出Markdown、NPY和图片文件夹",
            parameters_schema={
                "type": "object",
                "properties": {
                    "pdf_path": {"type": "string", "description": "PDF文件路径"},
                    "output_dir": {"type": "string", "description": "输出根目录"},
                    "paper_name": {"type": "string", "description": "论文名称（可选，用于命名输出文件夹）"},
                    "device": {"type": "string", "description": "计算设备，如cuda:1或cpu", "default": "cuda:1"}
                },
                "required": ["pdf_path"]
            }
        )

        # 批量PDF提取
        self.register_tool(
            name="batch_extract_pdf_multimodal",
            func=self._batch_extract_pdf_multimodal,
            description="批量从多个PDF论文中提取多模态数据",
            parameters_schema={
                "type": "object",
                "properties": {
                    "pdf_paths": {"type": "array", "description": "PDF文件路径列表"},
                    "output_dir": {"type": "string", "description": "输出根目录"},
                    "device": {"type": "string", "description": "计算设备", "default": "cuda:1"}
                },
                "required": ["pdf_paths"]
            }
        )

        # ==================== 旧版占位符工具（保留兼容性） ====================
        # PDF文本提取
        self.register_tool(
            name="extract_pdf_text",
            func=self._extract_pdf_text,
            description="从PDF文件中提取文本内容（简化版）",
            parameters_schema={
                "type": "object",
                "properties": {
                    "pdf_path": {"type": "string", "description": "PDF文件路径"}
                },
                "required": ["pdf_path"]
            }
        )

        # PDF图像提取
        self.register_tool(
            name="extract_pdf_images",
            func=self._extract_pdf_images,
            description="从PDF文件中提取图像（简化版）",
            parameters_schema={
                "type": "object",
                "properties": {
                    "pdf_path": {"type": "string", "description": "PDF文件路径"},
                    "output_dir": {"type": "string", "description": "图像输出目录"}
                },
                "required": ["pdf_path"]
            }
        )

        # PDF表格提取
        self.register_tool(
            name="extract_pdf_tables",
            func=self._extract_pdf_tables,
            description="从PDF文件中提取表格数据（简化版）",
            parameters_schema={
                "type": "object",
                "properties": {
                    "pdf_path": {"type": "string", "description": "PDF文件路径"}
                },
                "required": ["pdf_path"]
            }
        )

        logger.info("已注册多模态数据收集工具")

    def register_data_cleaning_tools(self):
        """注册数据清洗相关的MCP工具"""
        # 文本清洗
        self.register_tool(
            name="clean_text",
            func=self._clean_text,
            description="清洗文本数据",
            parameters_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "待清洗的文本"},
                    "remove_special_chars": {"type": "boolean", "description": "是否移除特殊字符"},
                    "normalize_whitespace": {"type": "boolean", "description": "是否规范化空白字符"}
                },
                "required": ["text"]
            }
        )

        # 去重
        self.register_tool(
            name="deduplicate_data",
            func=self._deduplicate_data,
            description="数据去重",
            parameters_schema={
                "type": "object",
                "properties": {
                    "data_list": {"type": "array", "description": "待去重的数据列表"},
                    "key_field": {"type": "string", "description": "用于判断重复的字段"}
                },
                "required": ["data_list"]
            }
        )

        # 数据验证
        self.register_tool(
            name="validate_data",
            func=self._validate_data,
            description="验证数据完整性和有效性",
            parameters_schema={
                "type": "object",
                "properties": {
                    "data": {"type": "object", "description": "待验证的数据"},
                    "schema": {"type": "object", "description": "数据schema"}
                },
                "required": ["data"]
            }
        )

        logger.info("已注册数据清洗工具")

    def register_dataset_tools(self):
        """注册数据集封装相关的MCP工具"""
        # 属性图格式化
        self.register_tool(
            name="format_attribute_graph",
            func=self._format_attribute_graph,
            description="将数据格式化为多模态属性图格式",
            parameters_schema={
                "type": "object",
                "properties": {
                    "nodes": {"type": "array", "description": "节点数据列表"},
                    "edges": {"type": "array", "description": "边数据列表"},
                    "metadata": {"type": "object", "description": "元数据"}
                },
                "required": ["nodes", "edges"]
            }
        )

        # 保存数据集
        self.register_tool(
            name="save_dataset",
            func=self._save_dataset,
            description="保存数据集到指定格式",
            parameters_schema={
                "type": "object",
                "properties": {
                    "dataset": {"type": "object", "description": "数据集对象"},
                    "format": {"type": "string", "enum": ["json", "parquet", "csv"], "description": "保存格式"},
                    "output_path": {"type": "string", "description": "输出路径"}
                },
                "required": ["dataset", "output_path"]
            }
        )

        logger.info("已注册数据集封装工具")

    # ==================== 工具实现占位符 ====================
    # 以下方法需要用户根据实际MCP工具进行实现

    def _extract_pdf_multimodal(
        self,
        pdf_path: str,
        output_dir: str = "./data/extracted",
        paper_name: str = None,
        device: str = "cuda:1"
    ) -> Dict:
        """
        从PDF论文提取多模态数据（核心工具）

        调用 pdf-extract-kit 进行：
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
        logger.info(f"[MCP] 提取PDF多模态数据: {pdf_path}")

        try:
            # 延迟导入避免启动时加载模型
            from .pdf_extractor_tool import extract_pdf_to_multimodal

            result = extract_pdf_to_multimodal(
                pdf_path=pdf_path,
                output_dir=output_dir,
                paper_name=paper_name,
                device=device
            )

            if result.get("status") == "success":
                logger.info(f"[MCP] PDF提取成功: {result.get('paper_name')}, {result.get('image_count')}张图片")

            return result

        except ImportError as e:
            logger.error(f"无法导入PDF提取工具: {e}")
            return {
                "status": "error",
                "message": f"PDF提取工具未正确配置: {e}",
                "pdf_path": pdf_path
            }
        except Exception as e:
            logger.error(f"PDF提取失败: {e}")
            return {
                "status": "error",
                "message": str(e),
                "pdf_path": pdf_path
            }

    def _batch_extract_pdf_multimodal(
        self,
        pdf_paths: List[str],
        output_dir: str = "./data/extracted",
        device: str = "cuda:1"
    ) -> Dict:
        """批量提取多个PDF的多模态数据"""
        logger.info(f"[MCP] 批量提取PDF: {len(pdf_paths)}个文件")

        try:
            from .pdf_extractor_tool import batch_extract_pdfs

            results = batch_extract_pdfs(
                pdf_paths=pdf_paths,
                output_dir=output_dir,
                device=device
            )

            success_count = sum(1 for r in results if r.get("status") == "success")
            logger.info(f"[MCP] 批量提取完成: {success_count}/{len(pdf_paths)}成功")

            return {
                "status": "success",
                "total": len(pdf_paths),
                "success_count": success_count,
                "results": results
            }

        except Exception as e:
            logger.error(f"批量PDF提取失败: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    def _download_arxiv_paper(self, paper_id: str, save_path: str = None) -> Dict:
        """从arXiv下载论文PDF"""
        import urllib.request
        import os

        logger.info(f"[MCP] 下载arXiv论文: {paper_id}")

        # 默认保存路径
        if save_path is None:
            save_path = f"./data/papers/{paper_id}.pdf"

        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 如果文件已存在，跳过下载
        if os.path.exists(save_path):
            logger.info(f"[MCP] PDF已存在，跳过下载: {save_path}")
            return {"status": "success", "paper_id": paper_id, "path": save_path, "cached": True}

        # 构建arXiv PDF URL
        # 注意: arXiv的PDF链接格式是 /pdf/{id} 而不是 /pdf/{id}.pdf
        # 支持新旧格式的paper_id (如 0704.1394 或 2301.07041)
        pdf_url = f"https://arxiv.org/pdf/{paper_id}"

        try:
            # 下载PDF
            logger.info(f"[MCP] 正在下载: {pdf_url}")
            req = urllib.request.Request(
                pdf_url,
                headers={'User-Agent': 'Mozilla/5.0 (compatible; DataCrawlerAgent/1.0)'}
            )

            with urllib.request.urlopen(req, timeout=120) as response:
                with open(save_path, 'wb') as f:
                    f.write(response.read())

            logger.info(f"[MCP] 下载完成: {save_path}")
            return {"status": "success", "paper_id": paper_id, "path": save_path, "cached": False}

        except urllib.error.HTTPError as e:
            error_msg = f"HTTP错误 {e.code}: {e.reason}"
            logger.error(f"[MCP] 下载失败: {error_msg}")
            return {"status": "error", "paper_id": paper_id, "message": error_msg}
        except urllib.error.URLError as e:
            error_msg = f"URL错误: {e.reason}"
            logger.error(f"[MCP] 下载失败: {error_msg}")
            return {"status": "error", "paper_id": paper_id, "message": error_msg}
        except Exception as e:
            error_msg = f"下载失败: {str(e)}"
            logger.error(f"[MCP] {error_msg}")
            return {"status": "error", "paper_id": paper_id, "message": error_msg}

    def download_arxiv_papers_batch(
        self,
        paper_ids: List[str],
        output_dir: str,
        max_workers: int = 5
    ) -> Dict[str, Any]:
        """
        批量并行下载arXiv论文

        Args:
            paper_ids: 论文ID列表
            output_dir: 输出目录
            max_workers: 最大并行数(默认5，arXiv建议不超过3)

        Returns:
            Dict[str, Any]: 下载结果统计
        """
        import concurrent.futures
        import os

        os.makedirs(output_dir, exist_ok=True)

        results = {
            "success": [],
            "failed": [],
            "cached": [],
            "total": len(paper_ids)
        }

        logger.info(f"[MCP] 开始批量下载 {len(paper_ids)} 篇论文，并行数: {max_workers}")

        def download_single(paper_id: str) -> tuple:
            """下载单篇论文"""
            save_path = os.path.join(output_dir, paper_id, f"{paper_id}.pdf")
            result = self._download_arxiv_paper(paper_id, save_path)
            return paper_id, result

        # 使用线程池并行下载
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_single, pid): pid for pid in paper_ids}

            for future in concurrent.futures.as_completed(futures):
                paper_id, result = future.result()

                if result["status"] == "success":
                    if result.get("cached"):
                        results["cached"].append(paper_id)
                    else:
                        results["success"].append(paper_id)
                else:
                    results["failed"].append({
                        "paper_id": paper_id,
                        "error": result.get("message", "Unknown error")
                    })

                # 进度显示
                done = len(results["success"]) + len(results["cached"]) + len(results["failed"])
                if done % 5 == 0 or done == len(paper_ids):
                    logger.info(f"[MCP] 下载进度: {done}/{len(paper_ids)} "
                               f"(成功: {len(results['success'])}, "
                               f"缓存: {len(results['cached'])}, "
                               f"失败: {len(results['failed'])})")

        logger.info(f"[MCP] 批量下载完成: 成功 {len(results['success'])}, "
                   f"缓存 {len(results['cached'])}, 失败 {len(results['failed'])}")

        return results

    def _download_doi_paper(self, doi: str, save_path: str = None) -> Dict:
        """通过DOI下载论文（占位符实现）"""
        logger.info(f"[MCP] 通过DOI下载论文: {doi}")
        # TODO: 调用实际的MCP工具
        return {"status": "success", "doi": doi, "path": save_path or f"./data/papers/{doi.replace('/', '_')}.pdf"}

    def _extract_pdf_text(self, pdf_path: str) -> Dict:
        """提取PDF文本（占位符实现）"""
        logger.info(f"[MCP] 提取PDF文本: {pdf_path}")
        # TODO: 调用实际的MCP工具
        return {"status": "success", "text": "提取的文本内容...", "path": pdf_path}

    def _extract_pdf_images(self, pdf_path: str, output_dir: str = None) -> Dict:
        """提取PDF图像（占位符实现）"""
        logger.info(f"[MCP] 提取PDF图像: {pdf_path}")
        # TODO: 调用实际的MCP工具
        return {"status": "success", "images": [], "output_dir": output_dir or "./data/images"}

    def _extract_pdf_tables(self, pdf_path: str) -> Dict:
        """提取PDF表格（占位符实现）"""
        logger.info(f"[MCP] 提取PDF表格: {pdf_path}")
        # TODO: 调用实际的MCP工具
        return {"status": "success", "tables": [], "path": pdf_path}

    def _clean_text(self, text: str, remove_special_chars: bool = True, normalize_whitespace: bool = True) -> Dict:
        """清洗文本（占位符实现）"""
        logger.info(f"[MCP] 清洗文本, 长度: {len(text)}")
        # TODO: 调用实际的MCP工具
        cleaned = text
        if normalize_whitespace:
            import re
            cleaned = re.sub(r'\s+', ' ', cleaned)
        return {"status": "success", "cleaned_text": cleaned}

    def _deduplicate_data(self, data_list: list, key_field: str = None) -> Dict:
        """数据去重（占位符实现）"""
        logger.info(f"[MCP] 数据去重, 输入数量: {len(data_list)}")
        # TODO: 调用实际的MCP工具
        return {"status": "success", "unique_count": len(data_list), "duplicates_removed": 0}

    def _validate_data(self, data: dict, schema: dict = None) -> Dict:
        """验证数据（占位符实现）"""
        logger.info(f"[MCP] 验证数据")
        # TODO: 调用实际的MCP工具
        return {"status": "success", "valid": True, "errors": []}

    def _format_attribute_graph(self, nodes: list, edges: list, metadata: dict = None) -> Dict:
        """格式化属性图（占位符实现）"""
        logger.info(f"[MCP] 格式化属性图, 节点数: {len(nodes)}, 边数: {len(edges)}")
        # TODO: 调用实际的MCP工具
        return {
            "status": "success",
            "graph": {
                "nodes": nodes,
                "edges": edges,
                "metadata": metadata or {}
            }
        }

    def _save_dataset(self, dataset: dict, output_path: str, format: str = "json") -> Dict:
        """保存数据集（占位符实现）"""
        logger.info(f"[MCP] 保存数据集到: {output_path}, 格式: {format}")
        # TODO: 调用实际的MCP工具
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            return {"status": "success", "path": output_path}
        except Exception as e:
            return {"status": "error", "message": str(e)}


def create_mcp_tool_manager() -> MCPToolManager:
    """
    创建并初始化MCP工具管理器

    Returns:
        MCPToolManager: 已注册所有工具的管理器实例
    """
    manager = MCPToolManager()
    manager.register_paper_download_tools()
    manager.register_multimodal_tools()
    manager.register_data_cleaning_tools()
    manager.register_dataset_tools()
    return manager
