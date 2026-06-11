"""
主协调器 (Orchestrator) — 多智能体系统入口
内部委托给 OrchestratorAgent 进行动态编排
保持 DataCrawlerOrchestrator 类名和公开API不变
"""
import json
from typing import Dict, Any, Optional, List
from openai import OpenAI
from loguru import logger
import os
from datetime import datetime

from agents import (
    PaperCollectorAgent, DatasetFormatterAgent,
    ArxivBatchDownloaderAgent, ReferenceGraphBuilderAgent, DataCleanerAgent
)
from agents.orchestrator_agent import OrchestratorAgent
from tools import create_mcp_tool_manager


class DataCrawlerOrchestrator:
    """
    数据爬取多Agent系统主协调器

    内部委托给 OrchestratorAgent 进行动态编排：
    - LLM分析任务 → 动态决定执行顺序
    - 每个Agent运行ReAct循环（观察→思考→行动→观察）
    - Blackboard + MessageBus 实现Agent间通信
    - 运行时LLM决策重试和错误恢复

    公开API与旧版完全兼容。
    """

    def __init__(self, context_manager: Optional[Any] = None):
        self.client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("MODEL_BASE_URL"),
        )
        self.context_manager = context_manager
        self._init_system_prompts()
        self.mcp_tool_manager = create_mcp_tool_manager()
        self._register_custom_tools()
        self._init_agents()
        self.execution_history: List[Dict] = []
        self.current_status: str = "idle"
        logger.info("DataCrawlerOrchestrator 初始化完成 (多智能体动态编排模式)")

    def _init_system_prompts(self):
        """初始化各Agent的系统提示词"""
        prompt_dir = "prompts"

        def load_prompt_content(name: str) -> str:
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
            'data_cleaner': DataCleanerAgent(
                client=self.client,
                system_prompt=self.data_cleaner_prompt,
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

    # ==================== 动态编排入口 ====================

    def execute_pipeline(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        多智能体动态编排流水线

        使用 OrchestratorAgent 进行:
        1. LLM分析任务 → 动态规划执行顺序
        2. 每个Agent运行ReAct循环
        3. Blackboard共享状态 + MessageBus通信
        4. 运行时LLM决策重试和错误恢复
        """
        logger.info("=" * 50)
        logger.info("开始多智能体动态编排流水线")
        logger.info("=" * 50)

        self.current_status = "running"
        start_time = datetime.now()

        # 创建 OrchestratorAgent
        orchestrator = OrchestratorAgent(
            client=self.client,
            agents=self.agents,
            mcp_tool_manager=self.mcp_tool_manager
        )

        try:
            result = orchestrator.run_pipeline(task_config)
        except Exception as e:
            result = {
                "status": "failed",
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": (datetime.now() - start_time).total_seconds(),
                "stages": {},
                "final_output": {},
                "errors": [str(e)]
            }

        # 生成LLM最终报告
        report = self._llm_generate_final_report(result)
        result["llm_report"] = report

        logger.info(f"\n{report}")

        self.current_status = "completed"
        self.execution_history.append(result)
        return result

    def execute_batch_pipeline(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """批量模式入口 — 与单论文模式统一走动态编排"""
        return self.execute_pipeline(task_config)

    # ==================== 单阶段执行（调试用）====================

    def execute_single_stage(self, stage_name: str, task: Dict[str, Any],
                              upstream_data: Optional[Dict] = None) -> Dict[str, Any]:
        if stage_name not in self.agents:
            return {"status": "error", "message": f"未知的阶段: {stage_name}"}
        agent = self.agents[stage_name]
        return agent.execute(task, upstream_data)

    # ==================== 辅助方法 ====================

    def get_agent(self, agent_name: str):
        return self.agents.get(agent_name)

    def get_execution_history(self) -> List[Dict]:
        return self.execution_history

    def get_current_status(self) -> str:
        return self.current_status

    def reload_prompts(self):
        logger.info("正在重新加载所有提示词...")
        self._init_system_prompts()
        self._init_agents()
        logger.info("提示词重新加载完成")

    # ==================== LLM报告生成 ====================

    def _llm_generate_final_report(self, pipeline_result: Dict) -> str:
        final = pipeline_result.get("final_output", {})
        stages = pipeline_result.get("stages", {})
        errors = pipeline_result.get("errors", [])

        stage_summaries = {}
        for name, stage in stages.items():
            stage_summaries[name] = {
                "status": stage.get("status"),
                "iterations": stage.get("iterations", "N/A"),
                "actions_count": len(stage.get("action_history", [])),
            }

        prompt = f"""你是多智能体系统的协调器。请生成执行报告。

最终输出: {json.dumps(final, ensure_ascii=False, indent=2)}
各阶段状态: {json.dumps(stage_summaries, ensure_ascii=False, indent=2)}
错误: {errors}

请生成简洁的中文执行报告，包括：
1. 任务概述
2. 各Agent执行情况（含ReAct迭代次数）
3. 最终数据集概览
4. 发现的问题和建议"""

        try:
            response = self.client.chat.completions.create(
                model=os.getenv("MODEL_NAME", "mimo-v2-flash"),
                messages=[
                    {"role": "system", "content": "你是协调器。生成简洁的中文执行报告。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"[Orchestrator] LLM报告生成失败: {e}")
            return "报告生成失败"

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
    return DataCrawlerOrchestrator()
