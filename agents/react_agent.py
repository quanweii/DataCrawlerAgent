"""
ReAct Agent 基类 — 实现 Observe → Think → Act 循环
所有专业Agent继承此类获得迭代决策能力，LLM输出直接驱动行动
"""
import re
import json
from enum import Enum
from typing import Dict, Any, List, Optional, Callable
from loguru import logger

from .base_agent import BaseAgent
from .messages import Blackboard, MessageBus, Message


class AgentState(Enum):
    IDLE = "idle"
    OBSERVING = "observing"
    THINKING = "thinking"
    ACTING = "acting"
    COMPLETED = "completed"
    FAILED = "failed"


class ReActAgent(BaseAgent):
    """
    ReAct循环Agent: Observe → Think → Act → Observe → ...

    子类需要实现:
        get_available_actions() — 声明可用的行动列表
        is_done(observation)    — 判断是否完成
        observe()               — 收集当前状态
        act(action_plan)        — 执行行动

    think() 由基类提供，使用LLM决定下一步行动。
    run_react_loop() 是主入口，替代旧的一次性execute()模式。
    """

    max_iterations: int = 10

    def __init__(self, client, system_prompt, mcp_tool_manager=None,
                 agent_name="ReActAgent", blackboard=None, message_bus=None):
        super().__init__(
            client=client,
            system_prompt=system_prompt,
            mcp_tool_manager=mcp_tool_manager,
            agent_name=agent_name
        )
        self.blackboard = blackboard
        self.message_bus = message_bus
        self._state = AgentState.IDLE
        self._action_history: List[Dict] = []
        self._iteration_count = 0
        self._accumulated_results: Dict[str, Any] = {}

    # ==================== 子类必须实现 ====================

    def get_available_actions(self) -> List[str]:
        """返回可用的行动名称列表"""
        raise NotImplementedError

    def is_done(self, observation: Dict) -> bool:
        """根据观察判断任务是否完成"""
        raise NotImplementedError

    # ==================== 核心ReAct循环 ====================

    def observe(self, task: Dict, blackboard: Blackboard,
                messages: List[Message]) -> Dict[str, Any]:
        """
        收集当前状态供think()使用
        子类应该override此方法添加领域特定观察
        """
        observation = {
            "task_summary": {k: str(v)[:200] for k, v in task.items()},
            "messages_count": len(messages),
            "action_history_count": len(self._action_history),
            "accumulated_results": dict(self._accumulated_results),
        }

        # 读取消息中的关键信息
        for msg in messages:
            if msg.msg_type == "quality_issue":
                observation.setdefault("quality_issues", []).append(msg.content)
            elif msg.msg_type == "reprocess_request":
                observation.setdefault("reprocess_requests", []).append(msg.content)

        return observation

    def think(self, observation: Dict) -> Dict[str, Any]:
        """
        LLM决策下一步行动
        返回: {"action": str, "params": dict, "reasoning": str}
        """
        actions = self.get_available_actions()
        history_summary = self._format_action_history()

        # 提取阶段提示（子类 observe() 可能设置）
        phase_hint = observation.get("phase_hint", "")
        progress = observation.get("progress_summary", "")

        prompt = f"""你是 {self.agent_name}。根据当前观察决定下一步行动。

可用行动:
{json.dumps(actions, ensure_ascii=False)}

当前进度: {progress}
{f"**重要提示**: {phase_hint}" if phase_hint else ""}

当前观察:
{json.dumps(observation, ensure_ascii=False, default=str)[:2000]}

已执行的行动:
{history_summary}

请以JSON格式输出你的决策（只输出JSON）:
{{
    "action": "action_name",
    "params": {{}},
    "reasoning": "为什么选择这个行动"
}}"""

        try:
            response = self._call_llm(
                messages=[
                    {"role": "system",
                     "content": f"你是{self.agent_name}。分析观察并选择最佳行动。严格以JSON格式输出。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=300
            )
            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                plan = json.loads(match.group())
                if plan.get("action") in actions:
                    logger.info(
                        f"[ReAct:{self.agent_name}] Think: "
                        f"action={plan['action']}, reason={plan.get('reasoning', '')[:100]}"
                    )
                    return plan
                else:
                    logger.warning(
                        f"[ReAct:{self.agent_name}] LLM选择了无效行动 '{plan.get('action')}', "
                        f"回退到默认"
                    )
        except Exception as e:
            logger.warning(f"[ReAct:{self.agent_name}] Think失败: {e}")

        return self._default_action(observation)

    def act(self, action_plan: Dict) -> Dict[str, Any]:
        """
        执行think()决定的行动
        子类应该override此方法，根据action_plan['action']分发到具体行动方法
        """
        action_name = action_plan.get("action", "unknown")
        params = action_plan.get("params", {})
        logger.info(f"[ReAct:{self.agent_name}] Act: {action_name}, params={params}")

        return {"status": "error", "message": f"未知行动: {action_name}"}

    def run_react_loop(self, task: Dict, blackboard: Blackboard,
                       message_bus: MessageBus) -> Dict[str, Any]:
        """
        核心ReAct循环: observe → think → act → observe → ...
        替代旧的一次性execute()模式
        """
        self._state = AgentState.IDLE
        self._action_history = []
        self._iteration_count = 0
        self._accumulated_results = {}

        logger.info(f"[ReAct:{self.agent_name}] 开始ReAct循环 (max={self.max_iterations})")

        try:
            while self._iteration_count < self.max_iterations:
                self._iteration_count += 1

                # 1. Observe
                self._state = AgentState.OBSERVING
                messages = message_bus.receive(self.agent_name) if message_bus else []
                observation = self.observe(task, blackboard, messages)

                logger.debug(
                    f"[ReAct:{self.agent_name}] 迭代 {self._iteration_count}: "
                    f"observe完成"
                )

                # 2. 检查是否完成
                if self.is_done(observation):
                    self._state = AgentState.COMPLETED
                    logger.info(
                        f"[ReAct:{self.agent_name}] 任务完成 "
                        f"(共{self._iteration_count}轮迭代)"
                    )
                    break

                # 3. Think
                self._state = AgentState.THINKING
                action_plan = self.think(observation)

                if action_plan.get("action") == "done":
                    self._state = AgentState.COMPLETED
                    logger.info(f"[ReAct:{self.agent_name}] LLM判断任务完成")
                    break

                # 4. Act
                self._state = AgentState.ACTING
                action_result = self.act(action_plan)

                # 记录行动历史
                self._action_history.append({
                    "iteration": self._iteration_count,
                    "action": action_plan.get("action"),
                    "params": action_plan.get("params", {}),
                    "reasoning": action_plan.get("reasoning", ""),
                    "result_status": action_result.get("status", "unknown")
                })

                # 累积结果
                if action_result.get("status") == "success":
                    self._accumulate_result(action_result)

                # 写入blackboard
                if blackboard:
                    namespace = self._get_blackboard_namespace()
                    blackboard.write(
                        namespace,
                        f"iteration_{self._iteration_count}",
                        {
                            "action": action_plan.get("action"),
                            "status": action_result.get("status"),
                            "summary": str(action_result.get("message", ""))[:200]
                        }
                    )

                # 处理消息
                self._process_action_messages(action_result, message_bus)

            else:
                logger.warning(
                    f"[ReAct:{self.agent_name}] 达到最大迭代次数 {self.max_iterations}"
                )

        except Exception as e:
            self._state = AgentState.FAILED
            logger.error(f"[ReAct:{self.agent_name}] 循环异常: {e}")

        # 构建最终结果
        result = self._build_final_result(task)
        return result

    # ==================== 辅助方法 ====================

    def _accumulate_result(self, action_result: Dict):
        """累积行动结果，列表字段去重"""
        for key, value in action_result.items():
            if key == "status":
                continue
            if key in self._accumulated_results:
                existing = self._accumulated_results[key]
                if isinstance(existing, list) and isinstance(value, list):
                    # 去重合并：按 arxiv_id / id / path 去重
                    def _item_identity(item):
                        if not isinstance(item, dict):
                            return id(item)
                        return (item.get("arxiv_id")
                                or item.get("id")
                                or item.get("path")
                                or id(item))
                    existing_ids = {_item_identity(item) for item in existing}
                    for item in value:
                        if _item_identity(item) not in existing_ids:
                            existing.append(item)
                elif isinstance(existing, dict) and isinstance(value, dict):
                    existing.update(value)
                else:
                    self._accumulated_results[key] = value
            else:
                self._accumulated_results[key] = value

    def _process_action_messages(self, action_result: Dict,
                                  message_bus: Optional[MessageBus]):
        """处理行动结果中需要发送的消息"""
        messages_to_send = action_result.get("_messages", [])
        for msg_data in messages_to_send:
            if message_bus:
                msg = Message(
                    sender=self.agent_name,
                    recipient=msg_data.get("recipient", "broadcast"),
                    msg_type=msg_data.get("msg_type", "status"),
                    content=msg_data.get("content", {}),
                    priority=msg_data.get("priority", 0)
                )
                message_bus.send(msg)

    def _build_final_result(self, task: Dict) -> Dict:
        """构建最终执行结果"""
        result = {
            "status": "success" if self._state == AgentState.COMPLETED else "partial",
            "agent_name": self.agent_name,
            "iterations": self._iteration_count,
            "action_history": self._action_history,
            **self._accumulated_results
        }

        if self._state == AgentState.FAILED:
            result["status"] = "error"

        return result

    def _format_action_history(self) -> str:
        """格式化行动历史供think()使用"""
        if not self._action_history:
            return "（尚未执行任何行动）"

        lines = []
        for h in self._action_history[-5:]:  # 只展示最近5个
            lines.append(
                f"  #{h['iteration']}: {h['action']} -> {h['result_status']} "
                f"({h.get('reasoning', '')[:60]})"
            )
        return "\n".join(lines)

    def _get_blackboard_namespace(self) -> str:
        """获取此Agent在blackboard中的命名空间"""
        name_map = {
            # ArXiv 领域
            "ArxivBatchDownloaderAgent": "downloader",
            "PaperCollectorAgent": "collector",
            "DataCleanerAgent": "cleaner",
            "ReferenceGraphBuilderAgent": "graph",
            "DatasetFormatterAgent": "formatter",
            # Reddit 领域
            "SubredditCollectorAgent": "subreddit_collector",
            "PostCollectorAgent": "post_collector",
            "SocialGraphBuilderAgent": "social_graph",
            # Amazon 领域
            "ProductCollectorAgent": "product_collector",
            "ReviewCollectorAgent": "review_collector",
            "ProductGraphBuilderAgent": "product_graph",
            # Steam 领域
            "SteamGameCollectorAgent": "steam_collector",
            "SteamGameGraphBuilderAgent": "steam_graph",
            # TMDb 领域
            "TmdbMovieCollectorAgent": "tmdb_collector",
            "TmdbMovieGraphBuilderAgent": "tmdb_graph",
            # WikiArt 领域
            "WikiartArtworkCollectorAgent": "wikiart_collector",
            "WikiartArtworkGraphBuilderAgent": "wikiart_graph",
            # 对抗性校验 (多智能体闭环)
            "GraphVerifierAgent": "verifier",
        }
        return name_map.get(self.agent_name, self.agent_name.lower())

    def _default_action(self, observation: Dict) -> Dict[str, Any]:
        """
        LLM决策失败时的默认行动
        子类可override以提供合理的默认行为
        """
        actions = self.get_available_actions()
        if actions:
            return {
                "action": actions[0],
                "params": {},
                "reasoning": "LLM决策失败，执行第一个可用行动作为默认"
            }
        return {"action": "done", "params": {}, "reasoning": "无可用行动"}

    def get_state(self) -> AgentState:
        """获取当前Agent状态"""
        return self._state

    def get_action_history(self) -> List[Dict]:
        """获取行动历史"""
        return self._action_history
