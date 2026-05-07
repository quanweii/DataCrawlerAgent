"""
Agent基类 - 提供统一的Agent接口和基础功能
仿照XianyuAutoAgent架构设计
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import os
from openai import OpenAI
from loguru import logger


class BaseAgent(ABC):
    """Agent基类，所有专业Agent都继承此类"""

    def __init__(
        self,
        client: OpenAI,
        system_prompt: str,
        mcp_tool_manager: Optional[Any] = None,
        agent_name: str = "BaseAgent"
    ):
        """
        初始化Agent

        Args:
            client: OpenAI客户端实例
            system_prompt: 系统提示词
            mcp_tool_manager: MCP工具管理器实例
            agent_name: Agent名称，用于日志记录
        """
        self.client = client
        self.system_prompt = system_prompt
        self.mcp_tool_manager = mcp_tool_manager
        self.agent_name = agent_name
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
