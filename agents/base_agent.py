"""
Agent基类 - 提供统一的Agent接口和基础功能
支持LLM驱动的任务分析、执行规划和结果评估
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import os
import re
import json
from openai import OpenAI
from loguru import logger


class BaseAgent(ABC):
    """Agent基类，所有专业Agent都继承此类"""

    def __init__(
        self,
        client: OpenAI,
        system_prompt: str,
        mcp_tool_manager: Optional[Any] = None,
        agent_name: str = "BaseAgent",
        blackboard: Optional[Any] = None,
        message_bus: Optional[Any] = None
    ):
        """
        初始化Agent

        Args:
            client: OpenAI客户端实例
            system_prompt: 系统提示词
            mcp_tool_manager: MCP工具管理器实例
            agent_name: Agent名称，用于日志记录
            blackboard: 共享状态存储（多智能体通信用）
            message_bus: 消息总线（多智能体通信用）
        """
        self.client = client
        self.system_prompt = system_prompt
        self.mcp_tool_manager = mcp_tool_manager
        self.agent_name = agent_name
        self.blackboard = blackboard
        self.message_bus = message_bus
        self.execution_history: List[Dict] = []  # 执行历史记录

    def generate(
        self,
        user_input: str,
        context: Optional[str] = None,
        extra_context: Optional[Dict[str, Any]] = None,
        temperature: float = 0.4,
        max_tokens: int = 2000
    ) -> str:
        """
        生成回复的模板方法

        Args:
            user_input: 用户输入/任务描述
            context: 上下文信息
            extra_context: 额外的上下文信息（如上游Agent的输出）
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            str: 生成的回复
        """
        messages = self._build_messages(user_input, context, extra_context)
        response = self._call_llm(messages, temperature, max_tokens)

        # 记录执行历史
        self.execution_history.append({
            "input": user_input,
            "output": response,
            "context": context
        })

        return response

    def _build_messages(
        self,
        user_input: str,
        context: Optional[str] = None,
        extra_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict]:
        """
        构建消息链

        Args:
            user_input: 用户输入
            context: 上下文信息
            extra_context: 额外上下文

        Returns:
            List[Dict]: 消息列表
        """
        system_content = self.system_prompt

        # 添加上下文信息
        if context:
            system_content += f"\n\n【上下文信息】\n{context}"

        # 添加额外上下文（如上游Agent输出）
        if extra_context:
            extra_info = self._format_extra_context(extra_context)
            system_content += f"\n\n【上游数据】\n{extra_info}"

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_input}
        ]

    def _format_extra_context(self, extra_context: Dict[str, Any]) -> str:
        """格式化额外上下文信息"""
        formatted = []
        for key, value in extra_context.items():
            if isinstance(value, dict):
                formatted.append(f"{key}: {value}")
            elif isinstance(value, list):
                formatted.append(f"{key}: {len(value)}项数据")
            else:
                formatted.append(f"{key}: {value}")
        return "\n".join(formatted)

    def _call_llm(
        self,
        messages: List[Dict],
        temperature: float = 0.4,
        max_tokens: int = 2000
    ) -> str:
        """
        调用大语言模型

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            str: 模型回复
        """
        try:
            response = self.client.chat.completions.create(
                model=os.getenv("MODEL_NAME", "qwen-max"),
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.8
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"[{self.agent_name}] LLM调用失败: {e}")
            return ""

    def call_mcp_tool(self, tool_name: str, **kwargs) -> Any:
        """
        调用MCP工具

        Args:
            tool_name: 工具名称
            **kwargs: 工具参数

        Returns:
            Any: 工具执行结果
        """
        if not self.mcp_tool_manager:
            logger.warning(f"[{self.agent_name}] MCP工具管理器未初始化")
            return None

        try:
            logger.info(f"[{self.agent_name}] 调用MCP工具: {tool_name}")
            result = self.mcp_tool_manager.call_tool(tool_name, **kwargs)
            return result
        except Exception as e:
            logger.error(f"[{self.agent_name}] MCP工具调用失败: {e}")
            raise

    @abstractmethod
    def execute(self, task: Dict[str, Any], upstream_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        执行Agent的具体任务（抽象方法，子类必须实现）

        Args:
            task: 任务描述和参数
            upstream_data: 上游Agent传递的数据

        Returns:
            Dict[str, Any]: 执行结果
        """
        pass

    def get_execution_history(self) -> List[Dict]:
        """获取执行历史"""
        return self.execution_history

    def clear_history(self):
        """清空执行历史"""
        self.execution_history = []

    def reason_about_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM驱动的任务分析：在执行前用LLM理解任务、识别风险、制定策略。
        这是让Agent具备"智能"的核心方法。
        """
        task_str = json.dumps(task, ensure_ascii=False, default=str)[:1500]

        prompt = f"""你是 {self.agent_name}。

角色描述: {self.system_prompt[:400]}

当前任务:
{task_str}

请分析这个任务，以JSON格式输出你的理解和计划:
{{
    "understanding": "你对任务目标的理解（一句话）",
    "key_points": ["需要关注的要点1", "要点2"],
    "risks": ["可能遇到的问题"],
    "strategy": "你的执行策略"
}}"""

        try:
            response = self._call_llm(
                messages=[
                    {"role": "system", "content": f"你是{self.agent_name}。分析任务并严格以JSON格式输出计划。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=600
            )
            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                result = json.loads(match.group())
                logger.info(f"[{self.agent_name}] 任务分析: {result.get('understanding', '')}")
                if result.get('risks'):
                    logger.info(f"[{self.agent_name}] 识别风险: {result['risks']}")
                return result
        except Exception as e:
            logger.warning(f"[{self.agent_name}] 任务分析失败: {e}")

        return {"understanding": "分析失败，使用默认策略", "key_points": [], "risks": [], "strategy": "按标准流程执行"}

    def evaluate_own_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM驱动的结果评估：执行后用LLM评估质量并给下游Agent提供建议。
        """
        summary = {"status": results.get("status"), "errors": results.get("errors", [])}

        if "papers" in results:
            summary["paper_count"] = len(results.get("papers", []))
            summary["extracted_count"] = len(results.get("extracted_folders", []))
        if "cleaned_papers" in results:
            summary.update(results.get("stats", {}))
        if "edge_count" in results:
            summary["edge_count"] = results.get("edge_count", 0)
            summary["papers_processed"] = results.get("papers_processed", 0)
        if "stats" in results and "node_count" in results.get("stats", {}):
            summary["node_count"] = results["stats"]["node_count"]

        prompt = f"""你是 {self.agent_name}。请评估你的执行结果。

结果摘要: {json.dumps(summary, ensure_ascii=False, indent=2)}

以JSON格式输出评估:
{{
    "quality": "好/一般/差",
    "issues": ["发现的问题"],
    "suggestions": "对下游Agent的建议",
    "confidence": 0.8
}}"""

        try:
            response = self._call_llm(
                messages=[
                    {"role": "system", "content": f"你是{self.agent_name}。评估结果，严格以JSON格式输出。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=400
            )
            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                return json.loads(match.group())
        except Exception as e:
            logger.warning(f"[{self.agent_name}] 结果评估失败: {e}")

        return {"quality": "未知", "issues": [], "suggestions": "", "confidence": 0.5}
