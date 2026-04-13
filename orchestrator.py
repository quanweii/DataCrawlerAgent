"""
主协调器 (Orchestrator) - 负责协调各分管Agent的执行
仿照XianyuAutoAgent的XianyuReplyBot设计
"""
from typing import Dict, Any, Optional, List
from openai import OpenAI
from loguru import logger
import os
from datetime import datetime

from agents import PaperCollectorAgent, DatasetFormatterAgent, ArxivBatchDownloaderAgent, ReferenceGraphBuilderAgent
from tools import create_mcp_tool_manager


class DataCrawlerOrchestrator:
    """
    数据爬取多Agent系统主协调器

    执行流程（顺序执行）：
    1. PaperCollectorAgent -> 收集论文PDF及多模态数据
    2. DataCleanerAgent -> 清洗整理数据
    3. DatasetFormatterAgent -> 封装成多模态属性图数据集格式
    """

    def __init__(self, context_manager: Optional[Any] = None):
        """
        初始化主协调器

        Args:
            context_manager: 可选的上下文管理器（预留扩展）
        """
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("MODEL_BASE_URL"),
        )

        self.context_manager = context_manager

        # 初始化系统提示词
        self._init_system_prompts()

        # 初始化MCP工具管理器（已注册所有工具）
        self.mcp_tool_manager = create_mcp_tool_manager()
        self._register_custom_tools()

        # 初始化各分管Agent
        self._init_agents()

        # 执行状态
        self.execution_history: List[Dict] = []
        self.current_status: str = "idle"

        logger.info("DataCrawlerOrchestrator 初始化完成")

    def _init_system_prompts(self):
        """初始化各Agent的系统提示词"""
        prompt_dir = "prompts"

        def load_prompt_content(name: str) -> str:
            """加载提示词文件"""
            target_path = os.path.join(prompt_dir, f"{name}.txt")
            if os.path.exists(target_path):
                file_path = target_path
            else:
                file_path = os.path.join(prompt_dir, f"{name}_example.txt")

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                logger.debug(f"已加载 {name} 提示词，路径: {file_path}")
                return content

        try:
            self.paper_collector_prompt = load_prompt_content("paper_collector_prompt")
            self.data_cleaner_prompt = load_prompt_content("data_cleaner_prompt")
            self.dataset_formatter_prompt = load_prompt_content("dataset_formatter_prompt")
            logger.info("成功加载所有提示词")
        except Exception as e:
            logger.warning(f"加载提示词时出错: {e}，使用默认提示词")
            self.paper_collector_prompt = self._default_paper_collector_prompt()
            self.data_cleaner_prompt = self._default_data_cleaner_prompt()
            self.dataset_formatter_prompt = self._default_dataset_formatter_prompt()

    def _register_custom_tools(self):
        """注册自定义MCP工具（如需要）"""
        # 这里可以注册用户自定义的MCP工具
        # self.mcp_tool_manager.register_tool(name, func, description, schema)
        pass

    def _init_agents(self):
        """初始化各分管Agent"""
        self.agents = {
            'arxiv_batch_downloader': ArxivBatchDownloaderAgent(
                client=self.client,
                system_prompt=self._default_arxiv_batch_downloader_prompt(),
                mcp_tool_manager=self.mcp_tool_manager
            ),
            'paper_collector': PaperCollectorAgent(
                client=self.client,
                system_prompt=self.paper_collector_prompt,
                mcp_tool_manager=self.mcp_tool_manager
            ),
            'reference_graph_builder': ReferenceGraphBuilderAgent(
                client=self.client,
                system_prompt=self._default_reference_graph_builder_prompt(),
                mcp_tool_manager=self.mcp_tool_manager
            ),
            'dataset_formatter': DatasetFormatterAgent(
                client=self.client,
                system_prompt=self.dataset_formatter_prompt,
                mcp_tool_manager=self.mcp_tool_manager
            ),
        }
        logger.info("所有分管Agent初始化完成")

    def execute_pipeline(
        self,
        task_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        执行完整的数据处理流水线

        Args:
            task_config: 任务配置，包含：
                - paper_sources: 论文来源列表
                - cleaning_rules: 清洗规则（可选）
                - dataset_name: 数据集名称（可选）
                - output_dirs: 输出目录配置（可选）

        Returns:
            Dict[str, Any]: 完整的执行结果
        """
        logger.info("=" * 50)
        logger.info("开始执行数据处理流水线")
        logger.info("=" * 50)

        self.current_status = "running"
        start_time = datetime.now()

        pipeline_result = {
            "status": "success",
            "start_time": start_time.isoformat(),
            "end_time": None,
            "duration_seconds": 0,
            "stages": {},
            "final_output": {},
            "errors": []
        }

        try:
            # ==================== 阶段1: 论文收集 ====================
            logger.info("\n[阶段 1/3] 论文收集Agent开始执行...")
            stage1_task = {
                "paper_sources": task_config.get("paper_sources", []),
                "output_dir": task_config.get("output_dirs", {}).get("papers", "./data/papers"),
                "extract_multimodal": task_config.get("extract_multimodal", True)
            }

            stage1_result = self.agents['paper_collector'].execute(stage1_task)
            pipeline_result["stages"]["paper_collection"] = stage1_result

            if stage1_result.get("status") != "success":
                raise Exception(f"论文收集阶段失败: {stage1_result.get('errors', [])}")

            logger.info(f"[阶段 1/3] 完成 - 收集了 {len(stage1_result.get('papers', []))} 篇论文")

            # ==================== 阶段2: 数据清洗 ====================
            logger.info("\n[阶段 2/3] 数据清洗Agent开始执行...")
            stage2_task = {
                "cleaning_rules": task_config.get("cleaning_rules", {}),
                "output_dir": task_config.get("output_dirs", {}).get("cleaned", "./data/cleaned")
            }

            stage2_result = self.agents['data_cleaner'].execute(
                task=stage2_task,
                upstream_data=stage1_result
            )
            pipeline_result["stages"]["data_cleaning"] = stage2_result

            if stage2_result.get("status") != "success":
                raise Exception(f"数据清洗阶段失败: {stage2_result.get('errors', [])}")

            logger.info(f"[阶段 2/3] 完成 - 清洗后数据量: {stage2_result['stats'].get('total_output', 0)}")

            # ==================== 阶段3: 数据集封装 ====================
            logger.info("\n[阶段 3/3] 数据集封装Agent开始执行...")
            stage3_task = {
                "dataset_name": task_config.get("dataset_name", f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                "output_format": task_config.get("output_format", "json"),
                "output_dir": task_config.get("output_dirs", {}).get("datasets", "./data/datasets"),
                "include_metadata": task_config.get("include_metadata", True)
            }

            stage3_result = self.agents['dataset_formatter'].execute(
                task=stage3_task,
                upstream_data=stage2_result
            )
            pipeline_result["stages"]["dataset_formatting"] = stage3_result

            if stage3_result.get("status") != "success":
                raise Exception(f"数据集封装阶段失败: {stage3_result.get('message', '未知错误')}")

            logger.info(f"[阶段 3/3] 完成 - 数据集已保存到: {stage3_result.get('output_path', '')}")

            # 记录最终输出
            pipeline_result["final_output"] = {
                "dataset_path": stage3_result.get("output_path", ""),
                "node_count": stage3_result.get("stats", {}).get("node_count", 0),
                "edge_count": stage3_result.get("stats", {}).get("edge_count", 0),
                "modality_distribution": stage3_result.get("stats", {}).get("modality_distribution", {})
            }

        except Exception as e:
            pipeline_result["status"] = "failed"
            pipeline_result["errors"].append(str(e))
            logger.error(f"流水线执行失败: {e}")

        finally:
            end_time = datetime.now()
            pipeline_result["end_time"] = end_time.isoformat()
            pipeline_result["duration_seconds"] = (end_time - start_time).total_seconds()
            self.current_status = "completed"
            self.execution_history.append(pipeline_result)

        logger.info("\n" + "=" * 50)
        logger.info(f"流水线执行完成 - 状态: {pipeline_result['status']}")
        logger.info(f"总耗时: {pipeline_result['duration_seconds']:.2f} 秒")
        logger.info("=" * 50)

        return pipeline_result

    def execute_batch_pipeline(
        self,
        task_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        批量下载模式的流水线 - 简化版

        流程:
        1. ArXiv筛选 → 筛选指定分类的论文
        2. 论文收集 → 下载PDF + 多模态提取 → 生成节点
        3. 引用图构建 → 提取References → 生成边
        4. 数据集封装 → 整合节点和边

        Args:
            task_config: 任务配置

        Returns:
            Dict[str, Any]: 完整的执行结果
        """
        logger.info("=" * 50)
        logger.info("开始执行批量下载流水线")
        logger.info("=" * 50)

        self.current_status = "running"
        start_time = datetime.now()

        pipeline_result = {
            "status": "success",
            "start_time": start_time.isoformat(),
            "end_time": None,
            "duration_seconds": 0,
            "stages": {},
            "final_output": {},
            "errors": []
        }

        try:
            # ==================== 阶段0: ArXiv批量筛选 ====================
            logger.info("\n[阶段 0/3] ArXiv批量下载Agent开始执行...")
            stage0_task = {
                "metadata_path": task_config.get("metadata_path", "/home/QuanWei/arxiv-metadata-oai-snapshot.json"),
                "category": task_config.get("category", "cs.AI"),
                "limit": task_config.get("limit", 20),
                "output_dir": task_config.get("output_dirs", {}).get("papers", "./data/papers")
            }

            stage0_result = self.agents['arxiv_batch_downloader'].execute(stage0_task)
            pipeline_result["stages"]["arxiv_batch_download"] = stage0_result

            if stage0_result.get("status") != "success":
                raise Exception(f"ArXiv批量筛选阶段失败: {stage0_result.get('errors', [])}")

            logger.info(f"[阶段 0/3] 完成 - 筛选到 {stage0_result['total_papers_found']} 篇论文")

            # 使用筛选结果作为后续阶段的输入
            paper_sources = stage0_result.get("paper_sources", [])
            if not paper_sources:
                raise Exception("没有找到符合条件的论文")

            # ==================== 阶段1: 论文收集 + 多模态提取 (生成节点) ====================
            logger.info("\n[阶段 1/3] 论文收集Agent开始执行 (生成节点)...")
            stage1_task = {
                "paper_sources": paper_sources,
                "output_dir": task_config.get("output_dirs", {}).get("papers", "./data/papers"),
                "extract_multimodal": task_config.get("extract_multimodal", True),
                "device": task_config.get("device", "cuda:1")
            }

            stage1_result = self.agents['paper_collector'].execute(stage1_task)
            pipeline_result["stages"]["paper_collection"] = stage1_result

            if stage1_result.get("status") != "success":
                raise Exception(f"论文收集阶段失败: {stage1_result.get('errors', [])}")

            logger.info(f"[阶段 1/3] 完成 - 收集了 {len(stage1_result.get('papers', []))} 篇论文 (节点)")

            # ==================== 阶段2: 引用图构建 (生成边) ====================
            logger.info("\n[阶段 2/3] 引用图构建Agent开始执行 (生成边)...")
            stage2_task = {
                "metadata_path": task_config.get("metadata_path", "/home/QuanWei/arxiv-metadata-cs.json"),
                "output_dir": task_config.get("output_dirs", {}).get("graphs", "./data/graphs")
            }

            stage2_result = self.agents['reference_graph_builder'].execute(
                task=stage2_task,
                upstream_data=stage1_result
            )
            pipeline_result["stages"]["reference_graph_building"] = stage2_result

            if stage2_result.get("status") != "success":
                logger.warning(f"引用图构建阶段部分失败: {stage2_result.get('message', '未知错误')}")
            else:
                logger.info(f"[阶段 2/3] 完成 - 构建了 {stage2_result.get('edge_count', 0)} 条引用边")

            # ==================== 阶段3: 数据集封装 ====================
            logger.info("\n[阶段 3/3] 数据集封装Agent开始执行...")
            stage3_task = {
                "dataset_name": task_config.get("dataset_name", f"multimodal_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                "output_format": task_config.get("output_format", "json"),
                "output_dir": task_config.get("output_dirs", {}).get("datasets", "./data/datasets"),
                "include_metadata": task_config.get("include_metadata", True),
                "edges_file": stage2_result.get("edges_file", "") if stage2_result.get("status") == "success" else None
            }

            stage3_result = self.agents['dataset_formatter'].execute(
                task=stage3_task,
                upstream_data=stage1_result
            )
            pipeline_result["stages"]["dataset_formatting"] = stage3_result

            if stage3_result.get("status") != "success":
                raise Exception(f"数据集封装阶段失败: {stage3_result.get('message', '未知错误')}")

            logger.info(f"[阶段 3/3] 完成 - 数据集已保存到: {stage3_result.get('output_path', '')}")

            # 记录最终输出
            pipeline_result["final_output"] = {
                "dataset_path": stage3_result.get("output_path", ""),
                "nodes_path": stage3_result.get("nodes_path", ""),
                "edges_path": stage2_result.get("edges_file", "") if stage2_result.get("status") == "success" else "",
                "node_count": len(stage1_result.get('papers', [])),
                "edge_count": stage2_result.get("edge_count", 0) if stage2_result.get("status") == "success" else 0,
                "modality_distribution": stage3_result.get("stats", {}).get("modality_distribution", {})
            }

        except Exception as e:
            pipeline_result["status"] = "failed"
            pipeline_result["errors"].append(str(e))
            logger.error(f"流水线执行失败: {e}")

        finally:
            end_time = datetime.now()
            pipeline_result["end_time"] = end_time.isoformat()
            pipeline_result["duration_seconds"] = (end_time - start_time).total_seconds()
            self.current_status = "completed"
            self.execution_history.append(pipeline_result)

        logger.info("\n" + "=" * 50)
        logger.info(f"批量下载流水线执行完成 - 状态: {pipeline_result['status']}")
        logger.info(f"节点数: {pipeline_result['final_output'].get('node_count', 0)}")
        logger.info(f"边数: {pipeline_result['final_output'].get('edge_count', 0)}")
        logger.info(f"总耗时: {pipeline_result['duration_seconds']:.2f} 秒")
        logger.info("=" * 50)

        return pipeline_result

    def execute_single_stage(
        self,
        stage_name: str,
        task: Dict[str, Any],
        upstream_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        执行单个阶段的任务（用于调试或单独执行）

        Args:
            stage_name: 阶段名称 ('paper_collector', 'data_cleaner', 'dataset_formatter')
            task: 任务参数
            upstream_data: 上游数据（如果需要）

        Returns:
            Dict[str, Any]: 执行结果
        """
        if stage_name not in self.agents:
            return {"status": "error", "message": f"未知的阶段: {stage_name}"}

        agent = self.agents[stage_name]
        return agent.execute(task, upstream_data)

    def get_agent(self, agent_name: str):
        """获取指定的Agent实例"""
        return self.agents.get(agent_name)

    def get_execution_history(self) -> List[Dict]:
        """获取执行历史"""
        return self.execution_history

    def get_current_status(self) -> str:
        """获取当前状态"""
        return self.current_status

    def reload_prompts(self):
        """重新加载所有提示词"""
        logger.info("正在重新加载提示词...")
        self._init_system_prompts()
        self._init_agents()
        logger.info("提示词重新加载完成")

    # ==================== 默认提示词 ====================

    def _default_paper_collector_prompt(self) -> str:
        return """你是一个专业的论文数据收集助手。你的任务是：
1. 根据提供的论文来源信息（如arXiv ID、DOI等）下载论文PDF
2. 从PDF中提取多模态数据，包括文本、图像和表格
3. 确保数据收集的完整性和准确性
4. 对收集的数据进行初步分类和标注"""

    def _default_data_cleaner_prompt(self) -> str:
        return """你是一个数据清洗专家。你的任务是：
1. 对收集的数据进行质量检查
2. 移除重复、无效或低质量的数据
3. 标准化数据格式
4. 生成数据清洗报告，说明清洗过程和结果"""

    def _default_dataset_formatter_prompt(self) -> str:
        return """你是一个数据集格式化专家。你的任务是：
1. 将清洗后的数据转换为多模态属性图格式
2. 构建节点和边的关系
3. 生成完整的元数据
4. 确保数据集符合标准格式要求
5. 生成数据集使用说明"""

    def _default_arxiv_batch_downloader_prompt(self) -> str:
        return """你是一个ArXiv论文批量下载专家。你的任务是：
1. 从ArXiv元数据文件中筛选指定分类的论文
2. 生成批量下载任务列表
3. 确保下载的论文符合研究需求
4. 提供筛选结果的摘要报告"""

    def _default_reference_graph_builder_prompt(self) -> str:
        return """你是一个引用关系图构建专家。你的任务是：
1. 从论文markdown文件中提取References部分
2. 解析参考文献，提取论文标题
3. 在arxiv元数据数据库中搜索匹配的论文
4. 构建引用关系有向图，记录节点和边的信息
5. 输出CSV格式的边列表"""


def create_orchestrator() -> DataCrawlerOrchestrator:
    """
    创建并返回Orchestrator实例

    Returns:
        DataCrawlerOrchestrator: 主协调器实例
    """
    return DataCrawlerOrchestrator()
