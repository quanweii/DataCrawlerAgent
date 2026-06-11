"""
动态编排Agent — LLM驱动的多智能体调度器
替代硬编码流水线，由LLM动态决定Agent执行顺序和重试策略
"""
import re
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from openai import OpenAI
from loguru import logger
import os
from datetime import datetime

from .messages import Blackboard, MessageBus


@dataclass
class PlanStep:
    """单个计划步骤"""
    agent_name: str
    purpose: str
    depends_on: List[str] = field(default_factory=list)
    max_retries: int = 1
    task_overrides: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending | running | completed | failed | skipped


class OrchestratorAgent:
    """
    动态编排Agent — 多智能体系统的调度核心

    工作流程:
    1. analyze_task() — LLM分析任务，生成动态执行计划
    2. run_pipeline() — 按计划执行，每个Agent运行ReAct循环
    3. decide_next_agent() — 运行时LLM决策下一步
    4. should_retry() — LLM评估失败是否值得重试
    """

    def __init__(self, client: OpenAI, agents: Dict[str, Any],
                 mcp_tool_manager=None):
        self.client = client
        self.agents = agents
        self.mcp_tool_manager = mcp_tool_manager
        self.blackboard = Blackboard()
        self.message_bus = MessageBus()

        # 为所有Agent注册消息总线
        for name in agents:
            self.message_bus.register_agent(name)

        self.max_retries_per_agent = 2

    def analyze_task(self, task_config: Dict) -> List[PlanStep]:
        """LLM分析任务并生成动态执行计划"""

        agents_desc = {
            "arxiv_batch_downloader": "从ArXiv元数据文件中按分类筛选论文，生成下载列表",
            "paper_collector": "下载论文PDF并用pdf-extract-kit提取文本/图片/公式等多模态数据",
            "data_cleaner": "用规则管道清洗Markdown：去重、修复断行、规范化公式和引用格式",
            "reference_graph_builder": "从论文中提取References，在arxiv元数据中匹配并构建引用边",
            "dataset_formatter": "将节点和边封装为多模态属性图数据集JSON"
        }

        task_summary = {k: v for k, v in task_config.items()
                        if k not in ['cleaning_rules']}
        task_str = json.dumps(task_summary, ensure_ascii=False, default=str)[:2000]

        prompt = f"""你是多智能体系统的动态编排器。分析任务并制定执行计划。

可用Agent:
{json.dumps(agents_desc, ensure_ascii=False, indent=2)}

重要规则:
- 如果任务已经提供了paper_sources（论文源列表），则不需要arxiv_batch_downloader，直接从paper_collector开始
- 只有当任务需要从metadata_path按分类筛选论文时，才需要arxiv_batch_downloader
- task_overrides必须留空{{}}，编排器会自动构建每个Agent的任务参数，不要在overrides中填写paper_sources等字段

任务配置:
{task_str}

请以JSON格式输出执行计划:
{{
    "task_type": "batch或single",
    "analysis": "简要分析",
    "plan": [
        {{
            "agent": "agent_name",
            "purpose": "目的",
            "depends_on": ["上游agent名"],
            "max_retries": 1,
            "task_overrides": {{}}
        }}
    ],
    "notes": "注意事项"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=os.getenv("MODEL_NAME", "mimo-v2-flash"),
                messages=[
                    {"role": "system",
                     "content": "你是多智能体编排器。分析任务并严格以JSON格式输出执行计划。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=800
            )
            text = response.choices[0].message.content
            logger.debug(f"[OrchestratorAgent] LLM原始输出:\n{text}")
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                json_str = match.group()
                plan_json = self._robust_json_parse(json_str)
                if not plan_json:
                    return self._default_plan(task_config)
                logger.info(
                    f"[OrchestratorAgent] LLM规划完成: "
                    f"{plan_json.get('analysis', '')}"
                )
                return self._parse_plan(plan_json)
        except Exception as e:
            logger.warning(f"[OrchestratorAgent] LLM规划失败: {e}")

        # 回退到默认计划
        return self._default_plan(task_config)

    @staticmethod
    def _robust_json_parse(json_str: str) -> Optional[Dict]:
        """多策略JSON解析，处理LLM常见的格式问题"""
        # 策略1: 直接解析
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # 策略2: 综合修复（单引号 + 注释 + 尾逗号 + unquoted key）
        import re as _re
        fixed = json_str
        # 去除 // 注释
        fixed = _re.sub(r'//[^\n]*', '', fixed)
        # 单引号替换为双引号（值中的引号）
        fixed = fixed.replace("'", '"')
        # 去除尾逗号 (, } 或 , ])
        fixed = _re.sub(r',\s*([}\]])', r'\1', fixed)
        # 引号包裹未加引号的key
        fixed = _re.sub(r'(?<=[{,])\s*(\w+)\s*:', r' "\1":', fixed)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        # 策略3: 提取第一个完整 {} 块后重试策略2
        depth = 0
        start = json_str.index('{')
        for i in range(start, len(json_str)):
            if json_str[i] == '{':
                depth += 1
            elif json_str[i] == '}':
                depth -= 1
                if depth == 0:
                    candidate = json_str[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        # 对提取的块再做一次综合修复
                        fixed2 = candidate
                        fixed2 = _re.sub(r'//[^\n]*', '', fixed2)
                        fixed2 = fixed2.replace("'", '"')
                        fixed2 = _re.sub(r',\s*([}\]])', r'\1', fixed2)
                        fixed2 = _re.sub(r'(?<=[{,])\s*(\w+)\s*:', r' "\1":', fixed2)
                        try:
                            return json.loads(fixed2)
                        except json.JSONDecodeError:
                            break

        logger.warning(f"[OrchestratorAgent] JSON解析全部失败: {json_str[:200]}")
        return None

    def _parse_plan(self, plan_json: Dict) -> List[PlanStep]:
        """解析LLM返回的JSON计划"""
        steps = []
        for s in plan_json.get("plan", []):
            agent_name = s.get("agent", "")
            if agent_name in self.agents:
                steps.append(PlanStep(
                    agent_name=agent_name,
                    purpose=s.get("purpose", ""),
                    depends_on=s.get("depends_on", []) or [],
                    max_retries=s.get("max_retries", 1),
                    task_overrides=s.get("task_overrides", {})
                ))
        return steps

    def _default_plan(self, task_config: Dict) -> List[PlanStep]:
        """默认执行计划（LLM规划失败时使用）"""
        has_paper_sources = bool(task_config.get("paper_sources"))
        has_category = bool(task_config.get("category"))
        has_metadata = bool(task_config.get("metadata_path"))

        if has_paper_sources:
            # 已有paper_sources，直接从collector开始
            # metadata_path仅用于graph builder的引用匹配，不触发batch模式
            return [
                PlanStep("paper_collector", "下载和提取论文",
                         max_retries=1),
                PlanStep("data_cleaner", "清洗数据",
                         depends_on=["paper_collector"],
                         max_retries=1),
                PlanStep("reference_graph_builder", "构建引用图",
                         depends_on=["paper_collector"],
                         max_retries=1),
                PlanStep("dataset_formatter", "封装数据集",
                         depends_on=["reference_graph_builder"],
                         max_retries=1),
            ]
        elif has_category:
            # 按分类筛选论文，需要batch downloader
            return [
                PlanStep("arxiv_batch_downloader", "筛选论文",
                         max_retries=1),
                PlanStep("paper_collector", "下载和提取论文",
                         depends_on=["arxiv_batch_downloader"],
                         max_retries=1),
                PlanStep("reference_graph_builder", "构建引用图",
                         depends_on=["paper_collector"],
                         max_retries=1),
                PlanStep("dataset_formatter", "封装数据集",
                         depends_on=["reference_graph_builder"],
                         max_retries=1),
            ]
        else:
            return [
                PlanStep("paper_collector", "下载和提取论文",
                         max_retries=1),
                PlanStep("data_cleaner", "清洗数据",
                         depends_on=["paper_collector"],
                         max_retries=1),
                PlanStep("dataset_formatter", "封装数据集",
                         depends_on=["data_cleaner"],
                         max_retries=1),
            ]

    def decide_next_agent(self, plan: List[PlanStep],
                          completed: List[str]) -> Optional[PlanStep]:
        """
        决定下一个要运行的Agent
        优先用LLM决策，失败时回退到依赖分析
        """
        # 找出所有依赖已满足的pending步骤
        runnable = []
        for step in plan:
            if step.status != "pending":
                continue
            deps_satisfied = all(d in completed for d in step.depends_on)
            if deps_satisfied:
                runnable.append(step)

        if not runnable:
            return None

        if len(runnable) == 1:
            return runnable[0]

        # 多个可运行时，用LLM选择
        blackboard_state = self.blackboard.snapshot()

        prompt = f"""你是编排器。以下Agent可以运行，选择最合适的一个。

可运行Agent: {json.dumps([{"agent": s.agent_name, "purpose": s.purpose} for s in runnable], ensure_ascii=False)}
已完成: {completed}
Blackboard状态: {json.dumps({k: {kk: str(vv)[:100] for kk, vv in v.items()} for k, v in blackboard_state.items()}, ensure_ascii=False)[:1000]}

以JSON输出: {{"choice": "agent_name", "reason": "原因"}}"""

        try:
            response = self.client.chat.completions.create(
                model=os.getenv("MODEL_NAME", "mimo-v2-flash"),
                messages=[
                    {"role": "system", "content": "你是编排器，选择下一个Agent。严格JSON输出。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=200
            )
            text = response.choices[0].message.content
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                decision = json.loads(match.group())
                chosen = decision.get("choice", "")
                for step in runnable:
                    if step.agent_name == chosen:
                        logger.info(
                            f"[OrchestratorAgent] LLM选择: {chosen}, "
                            f"原因: {decision.get('reason', '')[:80]}"
                        )
                        return step
        except Exception as e:
            logger.warning(f"[OrchestratorAgent] LLM选择失败: {e}")

        return runnable[0]

    def should_retry(self, step: PlanStep, result: Dict) -> bool:
        """LLM评估是否值得重试"""
        if step.max_retries <= 0:
            return False

        summary = {
            "agent": step.agent_name,
            "status": result.get("status"),
            "errors": result.get("errors", [])[:5],
            "iterations": result.get("iterations", 0),
            "action_history_count": len(result.get("action_history", []))
        }

        prompt = f"""Agent执行失败，是否值得重试？

{json.dumps(summary, ensure_ascii=False, indent=2)}

以JSON输出: {{"retry": true/false, "reason": "原因"}}"""

        try:
            response = self.client.chat.completions.create(
                model=os.getenv("MODEL_NAME", "mimo-v2-flash"),
                messages=[
                    {"role": "system", "content": "评估是否重试。严格JSON输出。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=150
            )
            text = response.choices[0].message.content
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                decision = json.loads(match.group())
                logger.info(
                    f"[OrchestratorAgent] 重试决策: "
                    f"retry={decision.get('retry')}, "
                    f"原因: {decision.get('reason', '')[:80]}"
                )
                return bool(decision.get("retry", False))
        except Exception as e:
            logger.warning(f"[OrchestratorAgent] 重试评估失败: {e}")

        return False

    def run_pipeline(self, task_config: Dict) -> Dict[str, Any]:
        """
        动态调度主循环
        """
        logger.info("=" * 50)
        logger.info("[OrchestratorAgent] 开始动态编排")
        logger.info("=" * 50)

        start_time = datetime.now()
        self.blackboard.write("global", "task_config", task_config)
        self.blackboard.write("global", "pipeline_status", "running")

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
            # 1. LLM规划
            plan = self.analyze_task(task_config)
            logger.info(
                f"[OrchestratorAgent] 执行计划: "
                f"{[s.agent_name for s in plan]}"
            )

            completed = []
            failed = []

            # 2. 动态调度循环
            while True:
                step = self.decide_next_agent(plan, completed)
                if step is None:
                    break

                step.status = "running"
                agent = self.agents.get(step.agent_name)
                if not agent:
                    logger.warning(f"[OrchestratorAgent] 未知Agent: {step.agent_name}")
                    step.status = "skipped"
                    continue

                # 注入blackboard和message_bus到Agent
                agent.blackboard = self.blackboard
                agent.message_bus = self.message_bus

                # 构建Agent任务
                agent_task = self._build_agent_task(
                    step.agent_name, task_config, step.task_overrides
                )

                logger.info(
                    f"\n{'='*50}\n"
                    f"[OrchestratorAgent] 调度 {step.agent_name}: {step.purpose}\n"
                    f"{'='*50}"
                )

                # 3. 运行Agent（ReAct循环）
                result = agent.execute(
                    task=agent_task,
                    upstream_data=self._get_upstream_data(completed)
                )

                pipeline_result["stages"][step.agent_name] = result

                # 4. 评估结果
                if result.get("status") in ("success", "partial"):
                    step.status = "completed"
                    completed.append(step.agent_name)
                    logger.info(
                        f"[OrchestratorAgent] {step.agent_name} 完成 "
                        f"(迭代: {result.get('iterations', 'N/A')})"
                    )
                else:
                    # 检查是否重试
                    if self.should_retry(step, result):
                        step.max_retries -= 1
                        step.status = "pending"  # 重置为pending
                        logger.info(
                            f"[OrchestratorAgent] 重试 {step.agent_name}"
                        )
                    else:
                        step.status = "failed"
                        failed.append(step.agent_name)
                        pipeline_result["errors"].append(
                            f"{step.agent_name}: {result.get('message', '执行失败')}"
                        )

                        # LLM决定是否继续
                        if self._should_continue(failed, completed):
                            logger.warning(
                                f"[OrchestratorAgent] {step.agent_name} 失败，继续执行"
                            )
                        else:
                            logger.error(
                                f"[OrchestratorAgent] {step.agent_name} 失败，终止流水线"
                            )
                            break

            # 5. 构建最终输出
            self._build_final_output(pipeline_result, task_config)

        except Exception as e:
            pipeline_result["status"] = "failed"
            pipeline_result["errors"].append(str(e))
            logger.error(f"[OrchestratorAgent] 流水线异常: {e}")

        finally:
            end_time = datetime.now()
            pipeline_result["end_time"] = end_time.isoformat()
            pipeline_result["duration_seconds"] = (end_time - start_time).total_seconds()
            self.blackboard.write("global", "pipeline_status", pipeline_result["status"])

        return pipeline_result

    def _build_agent_task(self, agent_name: str, task_config: Dict,
                          overrides: Dict) -> Dict:
        """根据Agent类型构建任务参数"""
        base = dict(task_config)

        # 调试：记录overrides中可能导致问题的字段
        if overrides:
            logger.info(
                f"[OrchestratorAgent] {agent_name} task_overrides: "
                f"{json.dumps(overrides, ensure_ascii=False, default=str)[:500]}"
            )

        if agent_name == "arxiv_batch_downloader":
            task = {
                "metadata_path": base.get("metadata_path"),
                "category": base.get("category", "cs.AI"),
                "limit": base.get("limit", 20),
                "output_dir": base.get("output_dirs", {}).get("papers", "./data/papers"),
                "sort_by_date": base.get("sort_by_date", True),
                "paper_sources": base.get("paper_sources", [])
            }
        elif agent_name == "paper_collector":
            # 从blackboard读取paper_sources
            paper_sources = self.blackboard.read("downloader", "paper_sources", [])
            if not paper_sources:
                paper_sources = base.get("paper_sources", [])
            task = {
                "paper_sources": paper_sources,
                "output_dir": base.get("output_dirs", {}).get("papers", "./data/papers"),
                "extract_multimodal": base.get("extract_multimodal", True),
                "device": base.get("device", "cuda:0"),
                "parallel_extraction": base.get("parallel_extraction",
                                                 base.get("parallel", {}).get("enabled", True)),
                "num_gpus": base.get("parallel", {}).get("num_gpus", 8),
                "batch_size": base.get("parallel", {}).get("batch_size", 20),
                "batch_download": True
            }
        elif agent_name == "data_cleaner":
            papers_dir = base.get("output_dirs", {}).get("papers", "./data/papers")
            task = {
                "parallel": base.get("parallel", {}).get("enabled", False),
                "num_workers": base.get("parallel", {}).get("num_workers", 8),
                "output_dir": base.get("output_dirs", {}).get("cleaned", "./data/cleaned"),
                "data_dir": papers_dir
            }
        elif agent_name == "reference_graph_builder":
            task = {
                "metadata_path": base.get("metadata_path"),
                "output_dir": base.get("output_dirs", {}).get("graphs", "./data/graphs"),
                "parallel": base.get("parallel", {}).get("enabled", True),
                "num_workers": base.get("parallel", {}).get("num_workers", 8)
            }
        elif agent_name == "dataset_formatter":
            # 优先使用 config 中的 edges_file，否则从 blackboard 读取
            edges_file = base.get("edges_file", "")
            if not edges_file:
                edges_file = self.blackboard.read("graph", "edges_file", "")
            task = {
                "dataset_name": base.get("dataset_name",
                                          f"multimodal_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                "output_format": base.get("output_format", "json"),
                "output_dir": base.get("output_dirs", {}).get("datasets", "./data/datasets"),
                "include_metadata": base.get("include_metadata", True),
                "edges_file": edges_file
            }
        else:
            task = base

        # LLM的task_overrides不应包含由编排器自动管理的字段
        protected_keys = {"paper_sources", "output_dir", "metadata_path"}
        clean_overrides = {k: v for k, v in overrides.items()
                           if k not in protected_keys}
        if len(clean_overrides) < len(overrides):
            removed = set(overrides.keys()) - set(clean_overrides.keys())
            logger.info(
                f"[OrchestratorAgent] 从task_overrides中移除自动管理字段: {removed}"
            )
        task.update(clean_overrides)

        return task

    def _get_upstream_data(self, completed: List[str]) -> Optional[Dict]:
        """从blackboard构建上游数据"""
        if not completed:
            return None

        upstream = {}
        for agent_name in completed:
            ns_data = self.blackboard.read_namespace(
                self._agent_namespace(agent_name)
            )
            if ns_data:
                upstream[agent_name] = ns_data

        # 特殊：paper_collector的上游数据
        if "paper_collector" in completed:
            upstream["papers"] = self.blackboard.read("collector", "papers", [])
            upstream["extracted_folders"] = self.blackboard.read(
                "collector", "extracted_folders", []
            )

        return upstream

    def _should_continue(self, failed: List[str], completed: List[str]) -> bool:
        """LLM决定是否在Agent失败后继续"""
        # 简单规则：关键Agent失败则终止
        critical_agents = {"arxiv_batch_downloader", "paper_collector"}
        if any(f in critical_agents for f in failed):
            return False

        # 非关键Agent（如reference_graph_builder）失败可以继续
        return True

    def _build_final_output(self, pipeline_result: Dict, task_config: Dict):
        """构建最终输出"""
        stages = pipeline_result.get("stages", {})

        collector_result = stages.get("paper_collector", {})
        graph_result = stages.get("reference_graph_builder", {})
        formatter_result = stages.get("dataset_formatter", {})

        pipeline_result["final_output"] = {
            "dataset_path": formatter_result.get("output_path", ""),
            "nodes_path": formatter_result.get("output_path", ""),
            "edges_path": graph_result.get("edges_file", ""),
            "node_count": len(collector_result.get('papers', [])),
            "edge_count": graph_result.get("edge_count", 0),
            "modality_distribution": formatter_result.get("stats", {}).get(
                "modality_distribution", {}
            )
        }

    def _agent_namespace(self, agent_name: str) -> str:
        name_map = {
            "arxiv_batch_downloader": "downloader",
            "paper_collector": "collector",
            "data_cleaner": "cleaner",
            "reference_graph_builder": "graph",
            "dataset_formatter": "formatter",
        }
        return name_map.get(agent_name, agent_name)
