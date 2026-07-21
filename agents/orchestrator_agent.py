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


@dataclass
class DomainConfig:
    """领域特定的 Pipeline 配置"""
    domain: str
    agents_desc: Dict[str, str]           # agent_name -> 描述
    default_plan_factory: str             # 工厂方法名
    critical_agents: set                  # 关键 Agent（失败则终止）
    task_builders: Dict[str, str]         # agent_name -> _build_task_xxx 方法名
    namespace_map: Dict[str, str]         # agent_name -> blackboard namespace
    graph_mode: str = "heterogeneous"     # 建图模式: homogeneous / heterogeneous


# ==================== 领域配置注册表 ====================

DOMAIN_CONFIGS = {
    "arxiv": DomainConfig(
        domain="arxiv",
        agents_desc={
            "arxiv_batch_downloader": "从ArXiv元数据文件中按分类筛选论文，生成下载列表",
            "paper_collector": "下载论文PDF并用pdf-extract-kit提取文本/图片/公式等多模态数据",
            "data_cleaner": "用规则管道清洗Markdown：去重、修复断行、规范化公式和引用格式",
            "reference_graph_builder": "从论文中提取References，在arxiv元数据中匹配并构建引用边",
            "dataset_formatter": "将节点和边封装为多模态属性图数据集JSON",
        },
        default_plan_factory="_default_plan_arxiv",
        critical_agents={"arxiv_batch_downloader", "paper_collector"},
        task_builders={
            "arxiv_batch_downloader": "_build_task_arxiv_downloader",
            "paper_collector": "_build_task_paper_collector",
            "data_cleaner": "_build_task_data_cleaner",
            "reference_graph_builder": "_build_task_ref_graph",
            "dataset_formatter": "_build_task_dataset_formatter",
        },
        namespace_map={
            "arxiv_batch_downloader": "downloader",
            "paper_collector": "collector",
            "data_cleaner": "cleaner",
            "reference_graph_builder": "graph",
            "dataset_formatter": "formatter",
        },
    ),
    "reddit": DomainConfig(
        domain="reddit",
        agents_desc={
            "subreddit_collector": "发现/验证目标subreddit，采集帖子元数据（标题、得分、图片URL等）",
            "post_collector": "采集帖子评论树（深度限制BFS）、下载帖子图片、收集用户元数据",
            "data_cleaner": "清洗Reddit文本数据：去除标记残留、规范化URL、处理[deleted]占位符",
            "social_graph_builder": "构建User-Post-Comment-Subreddit异构图（author_of/reply_to/belongs_to/crosspost边）",
            "dataset_formatter": "将异构节点和边封装为多模态属性图数据集JSON",
        },
        default_plan_factory="_default_plan_reddit",
        critical_agents={"subreddit_collector", "post_collector"},
        task_builders={
            "subreddit_collector": "_build_task_subreddit_collector",
            "post_collector": "_build_task_post_collector",
            "data_cleaner": "_build_task_data_cleaner",
            "social_graph_builder": "_build_task_social_graph",
            "dataset_formatter": "_build_task_dataset_formatter",
        },
        namespace_map={
            "subreddit_collector": "subreddit_collector",
            "post_collector": "post_collector",
            "data_cleaner": "cleaner",
            "social_graph_builder": "social_graph",
            "dataset_formatter": "formatter",
        },
    ),
    "amazon": DomainConfig(
        domain="amazon",
        agents_desc={
            "product_collector": "加载McAuley Amazon产品元数据、下载产品图片、提取共买/共看关系",
            "review_collector": "加载/过滤/采样产品评论，构建用户-产品交互数据",
            "data_cleaner": "清洗产品描述和评论文本：去除HTML标签、规范化格式",
            "product_graph_builder": "构建Product-User-Category-Brand异构图（review/also_bought/also_viewed/belongs_to/made_by边）",
            "dataset_formatter": "将异构节点和边封装为多模态属性图数据集JSON",
        },
        default_plan_factory="_default_plan_amazon",
        critical_agents={"product_collector"},
        task_builders={
            "product_collector": "_build_task_product_collector",
            "review_collector": "_build_task_review_collector",
            "data_cleaner": "_build_task_data_cleaner",
            "product_graph_builder": "_build_task_product_graph",
            "dataset_formatter": "_build_task_dataset_formatter",
        },
        namespace_map={
            "product_collector": "product_collector",
            "review_collector": "review_collector",
            "data_cleaner": "cleaner",
            "product_graph_builder": "product_graph",
            "dataset_formatter": "formatter",
        },
    ),
    "steam": DomainConfig(
        domain="steam",
        agents_desc={
            "steam_game_collector": "加载/爬取Steam游戏元数据(genres/developers/publishers/categories)，确保cs_lo提取目录就绪",
            "steam_game_graph_builder": "构建Steam游戏同构图：小类过滤+密度优化co_attribute边(shared_genre/same_developer/same_publisher/shared_category)，产出带label/label_index的节点和边",
            "dataset_formatter": "将同构节点和边封装为多模态属性图数据集JSON(text+image双模态)",
            "graph_verifier": "对抗性校验图质量：检测标签泄露(LP-ceiling)、定义性同质(per-relation homophily)、节点覆盖——发现缺陷发reprocess_request给上游",
        },
        default_plan_factory="_default_plan_steam",
        critical_agents={"steam_game_collector"},
        task_builders={
            "steam_game_collector": "_build_task_steam_collector",
            "steam_game_graph_builder": "_build_task_steam_graph",
            "dataset_formatter": "_build_task_dataset_formatter",
            "graph_verifier": "_build_task_verifier",
        },
        namespace_map={
            "steam_game_collector": "steam_collector",
            "steam_game_graph_builder": "steam_graph",
            "dataset_formatter": "formatter",
            "graph_verifier": "verifier",
        },
        graph_mode="homogeneous",
    ),
    "tmdb": DomainConfig(
        domain="tmdb",
        agents_desc={
            "tmdb_movie_collector": "加载/爬取TMDb电影元数据(genres/directors/actors)，确保cs_lo提取目录就绪",
            "tmdb_movie_graph_builder": "构建TMDb电影同构图：小类过滤+密度优化co_attribute边(shared_genre/same_director/shared_actor)，产出带label/label_index的节点和边",
            "dataset_formatter": "将同构节点和边封装为多模态属性图数据集JSON(text+image双模态)",
            "graph_verifier": "对抗性校验图质量：检测标签泄露(LP-ceiling)、定义性同质(per-relation homophily)、节点覆盖——发现缺陷发reprocess_request给上游",
        },
        default_plan_factory="_default_plan_tmdb",
        critical_agents={"tmdb_movie_collector"},
        task_builders={
            "tmdb_movie_collector": "_build_task_tmdb_collector",
            "tmdb_movie_graph_builder": "_build_task_tmdb_graph",
            "dataset_formatter": "_build_task_dataset_formatter",
            "graph_verifier": "_build_task_verifier",
        },
        namespace_map={
            "tmdb_movie_collector": "tmdb_collector",
            "tmdb_movie_graph_builder": "tmdb_graph",
            "dataset_formatter": "formatter",
            "graph_verifier": "verifier",
        },
        graph_mode="homogeneous",
    ),
    "wikiart": DomainConfig(
        domain="wikiart",
        agents_desc={
            "wikiart_artwork_collector": "加载/爬取WikiArt画作元数据(style/artist/title)，确保cs_lo提取目录就绪",
            "wikiart_artwork_graph_builder": "构建WikiArt画作同构图：密度优化co_attribute边(same_artist/shared_style)，产出带label/label_index的节点和边",
            "dataset_formatter": "将同构节点和边封装为多模态属性图数据集JSON(text+image双模态)",
            "graph_verifier": "对抗性校验图质量：检测标签泄露(LP-ceiling)、定义性同质(per-relation homophily)——WikiArt的shared_style边homophily≈1.0是经典泄漏，必须标记并触发reprocess",
        },
        default_plan_factory="_default_plan_wikiart",
        critical_agents={"wikiart_artwork_collector"},
        task_builders={
            "wikiart_artwork_collector": "_build_task_wikiart_collector",
            "wikiart_artwork_graph_builder": "_build_task_wikiart_graph",
            "dataset_formatter": "_build_task_dataset_formatter",
            "graph_verifier": "_build_task_verifier",
        },
        namespace_map={
            "wikiart_artwork_collector": "wikiart_collector",
            "wikiart_artwork_graph_builder": "wikiart_graph",
            "dataset_formatter": "formatter",
            "graph_verifier": "verifier",
        },
        graph_mode="homogeneous",
    ),
}


def register_domain_config(name: str, config: DomainConfig):
    """动态注册域配置 (供 YAML 驱动域使用)。线程安全。"""
    DOMAIN_CONFIGS[name] = config


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

        domain = task_config.get("domain", "arxiv")
        domain_cfg = DOMAIN_CONFIGS.get(domain, DOMAIN_CONFIGS["arxiv"])

        agents_desc = domain_cfg.agents_desc

        task_summary = {k: v for k, v in task_config.items()
                        if k not in ['cleaning_rules']}
        task_str = json.dumps(task_summary, ensure_ascii=False, default=str)[:2000]

        prompt = f"""你是多智能体系统的动态编排器。分析任务并制定执行计划。

当前领域: {domain}
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
        """默认执行计划 -- 按 DomainConfig.default_plan_factory 分发。
        对于硬编码域: 使用预定义的 factory 方法名。
        对于 YAML 动态域: 使用 _default_plan_co_attribute。
        """
        domain = task_config.get("domain", "arxiv")
        domain_cfg = DOMAIN_CONFIGS.get(domain)
        if domain_cfg:
            factory_name = domain_cfg.default_plan_factory
            factory = getattr(self, factory_name, None)
            if factory:
                return factory(task_config)
        # 未知域 / 无 DomainConfig: 回退到 arxiv 计划
        return self._default_plan_arxiv(task_config)

    def _default_plan_arxiv(self, task_config: Dict) -> List[PlanStep]:
        """ArXiv 默认执行计划"""
        # 统一模式 (A′): 仅跑 formatter, 把 reference_graph_builder 已构造的引用网络
        # (graph_data.pt) 统一成 newdomain MAG。不爬取、不需要 OAI 快照。
        if task_config.get("arxiv_unify"):
            return [
                PlanStep("dataset_formatter", "统一封装 arxiv MAG",
                         max_retries=1),
            ]
        has_paper_sources = bool(task_config.get("paper_sources"))
        has_category = bool(task_config.get("category"))
        has_metadata = bool(task_config.get("metadata_path"))

        if has_paper_sources:
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

    def _default_plan_reddit(self, task_config: Dict) -> List[PlanStep]:
        """Reddit 默认执行计划"""
        return [
            PlanStep("subreddit_collector", "发现subreddit并采集帖子",
                     max_retries=1),
            PlanStep("post_collector", "采集评论和下载图片",
                     depends_on=["subreddit_collector"],
                     max_retries=1),
            PlanStep("data_cleaner", "清洗文本数据",
                     depends_on=["post_collector"],
                     max_retries=1),
            PlanStep("social_graph_builder", "构建社交图",
                     depends_on=["post_collector"],
                     max_retries=1),
            PlanStep("dataset_formatter", "封装数据集",
                     depends_on=["social_graph_builder"],
                     max_retries=1),
        ]

    def _default_plan_amazon(self, task_config: Dict) -> List[PlanStep]:
        """Amazon 默认执行计划"""
        return [
            PlanStep("product_collector", "加载产品元数据和图片",
                     max_retries=1),
            PlanStep("review_collector", "加载产品评论",
                     depends_on=["product_collector"],
                     max_retries=1),
            PlanStep("data_cleaner", "清洗文本数据",
                     depends_on=["review_collector"],
                     max_retries=1),
            PlanStep("product_graph_builder", "构建产品图",
                     depends_on=["product_collector", "review_collector"],
                     max_retries=1),
            PlanStep("dataset_formatter", "封装数据集",
                     depends_on=["product_graph_builder"],
                     max_retries=1),
        ]

    def _default_plan_steam(self, task_config: Dict) -> List[PlanStep]:
        """Steam 默认执行计划 (无 cleaner: cs_lo 文本已优质)"""
        return self._default_plan_co_attribute(task_config)

    def _default_plan_tmdb(self, task_config: Dict) -> List[PlanStep]:
        """TMDb 默认执行计划 (无 cleaner)"""
        return self._default_plan_co_attribute(task_config)

    def _default_plan_wikiart(self, task_config: Dict) -> List[PlanStep]:
        """WikiArt 默认执行计划 (无 cleaner)"""
        return self._default_plan_co_attribute(task_config)

    def _default_plan_co_attribute(self, task_config: Dict) -> List[PlanStep]:
        """共享 co_attribute 域默认计划 (steam/tmdb/wikiart + YAML 动态域)。

        从 DomainConfig.agents_desc 动态发现 agent 名称, 而非硬编码。
        计划: collector -> graph_builder -> formatter -> verifier
        """
        domain = task_config.get("domain", "")
        domain_cfg = DOMAIN_CONFIGS.get(domain)
        if not domain_cfg:
            return self._default_plan_arxiv(task_config)

        agent_names = list(domain_cfg.agents_desc.keys())
        collector = next((a for a in agent_names if "collector" in a), None)
        builder = next((a for a in agent_names if "graph" in a and "builder" in a), None)

        steps = []
        if collector:
            steps.append(PlanStep(collector, f"加载{domain}元数据", max_retries=1))
        if builder:
            depends = [collector] if collector else []
            steps.append(PlanStep(builder, f"构建{domain}同构图",
                                  depends_on=depends, max_retries=1))
        depends_on = [builder] if builder else ([collector] if collector else [])
        steps.append(PlanStep("dataset_formatter", "封装数据集",
                              depends_on=depends_on, max_retries=1))
        steps.append(PlanStep("graph_verifier", "对抗性校验图质量",
                              depends_on=["dataset_formatter"], max_retries=1))
        return steps

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

                    # ---- 多智能体反馈回路 (multi-agent quality loop) ----
                    # 接住 MessageBus 上的 reprocess_request, 让已完成 agent 重新执行。
                    # 这是 "multi-agent" 承重的关节: Verifier→Builder 的 reprocess
                    # 回路使图构建从单次流水线变成对抗性质量闭环。
                    self._reprocess_count = getattr(self, '_reprocess_count', {})
                    MAX_REPROCESS_ROUNDS = 2
                    for src_agent in list(self.message_bus._queues.keys()):
                        msgs = self.message_bus.receive(src_agent)
                        for msg in msgs:
                            if msg.msg_type != "reprocess_request" or msg.priority < 1:
                                continue
                            target = msg.recipient
                            if target not in completed and target not in [
                                s.agent_name for s in plan if s.status == "pending"
                            ]:
                                continue
                            cnt = self._reprocess_count.get(target, 0)
                            if cnt >= MAX_REPROCESS_ROUNDS:
                                logger.warning(
                                    f"[OrchestratorAgent] {target} 已达 reprocess 上限 "
                                    f"({cnt}/{MAX_REPROCESS_ROUNDS}), 跳过"
                                )
                                continue
                            self._reprocess_count[target] = cnt + 1
                            reprocess_step = PlanStep(
                                agent_name=target,
                                purpose=f"[Reprocess #{cnt+1}] {msg.content.get('reason', str(msg.content)[:80])}",
                                depends_on=[src_agent],
                                max_retries=1,
                                task_overrides={"reprocess_context": msg.content},
                            )
                            plan.append(reprocess_step)
                            # 级联: 标记 target 的所有下游 completed→pending
                            def _downstream(agent_name, all_steps, visited=None):
                                if visited is None:
                                    visited = set()
                                if agent_name in visited:
                                    return set()
                                visited.add(agent_name)
                                ds = set()
                                for s in all_steps:
                                    if agent_name in s.depends_on:
                                        ds.add(s.agent_name)
                                        ds |= _downstream(s.agent_name, all_steps, visited)
                                return ds
                            cascade = _downstream(target, plan)
                            for s in plan:
                                if s.agent_name in cascade and s.status == "completed":
                                    s.status = "pending"
                                    # 从 completed 列表移除 (下游需要重新在 target 之后运行)
                                    if s.agent_name in completed:
                                        completed.remove(s.agent_name)
                                    logger.info(
                                        f"[OrchestratorAgent] ⟲ Cascade: {s.agent_name} "
                                        f"→ pending (upstream {target} reprocessed)"
                                    )
                            logger.info(
                                f"[OrchestratorAgent] ⟲ Feedback loop: "
                                f"{src_agent} → reprocess {target}"
                                f"(round {cnt+1}/{MAX_REPROCESS_ROUNDS})"
                                f" + cascade {sorted(cascade)}: "
                                f"{msg.content.get('reason', '')[:100]}"
                            )
                    # ---- 反馈回路结束 ----
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
        """根据Agent类型构建任务参数 — 支持多领域"""
        base = dict(task_config)
        domain = base.get("domain", "arxiv")

        # 调试：记录overrides中可能导致问题的字段
        if overrides:
            logger.info(
                f"[OrchestratorAgent] {agent_name} task_overrides: "
                f"{json.dumps(overrides, ensure_ascii=False, default=str)[:500]}"
            )

        # 按领域 + agent 分发
        task = {}
        if domain == "arxiv":
            task = self._build_task_arxiv(agent_name, base)
        elif domain == "reddit":
            task = self._build_task_reddit(agent_name, base)
        elif domain == "amazon":
            task = self._build_task_amazon(agent_name, base)
        elif domain == "steam":
            task = self._build_task_steam(agent_name, base)
        elif domain == "tmdb":
            task = self._build_task_tmdb(agent_name, base)
        elif domain == "wikiart":
            task = self._build_task_wikiart(agent_name, base)
        else:
            # 动态域 (YAML 驱动) — 从 DomainConfig.task_builders 分发
            dc = DOMAIN_CONFIGS.get(domain)
            if dc and agent_name in dc.task_builders:
                builder_name = dc.task_builders[agent_name]
                common = {
                    "domain": domain,
                    "graph_mode": dc.graph_mode,
                    "data_source": base.get("data_source", "file"),
                    "load_from_files": base.get("load_from_files", True),
                }
                base_with_domain = {**base, **common}
                # 分发: co_attribute 方法需要 agent_name，共享方法不需要
                if builder_name in ("_build_task_co_attribute_collector",
                                     "_build_task_co_attribute_graph"):
                    task = getattr(self, builder_name)(agent_name, base_with_domain)
                else:
                    task = getattr(self, builder_name)(base_with_domain)
            else:
                task = base

        # LLM的task_overrides不应包含由编排器自动管理的字段
        protected_keys = {"paper_sources", "output_dir", "metadata_path",
                          "posts", "products", "reviews"}
        clean_overrides = {k: v for k, v in overrides.items()
                           if k not in protected_keys}
        if len(clean_overrides) < len(overrides):
            removed = set(overrides.keys()) - set(clean_overrides.keys())
            logger.info(
                f"[OrchestratorAgent] 从task_overrides中移除自动管理字段: {removed}"
            )
        task.update(clean_overrides)

        return task

    # ==================== ArXiv 任务构建 ====================

    def _build_task_arxiv(self, agent_name: str, base: Dict) -> Dict:
        if agent_name == "arxiv_batch_downloader":
            return self._build_task_arxiv_downloader(base)
        elif agent_name == "paper_collector":
            return self._build_task_paper_collector(base)
        elif agent_name == "data_cleaner":
            return self._build_task_data_cleaner(base)
        elif agent_name == "reference_graph_builder":
            return self._build_task_ref_graph(base)
        elif agent_name == "dataset_formatter":
            return self._build_task_dataset_formatter(base)
        return base

    def _build_task_arxiv_downloader(self, base: Dict) -> Dict:
        return {
            "metadata_path": base.get("metadata_path"),
            "category": base.get("category", "cs.AI"),
            "limit": base.get("limit", 20),
            "output_dir": base.get("output_dirs", {}).get("papers", "./data/papers"),
            "sort_by_date": base.get("sort_by_date", True),
            "paper_sources": base.get("paper_sources", [])
        }

    def _build_task_paper_collector(self, base: Dict) -> Dict:
        paper_sources = self.blackboard.read("downloader", "paper_sources", [])
        if not paper_sources:
            paper_sources = base.get("paper_sources", [])
        return {
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

    def _build_task_data_cleaner(self, base: Dict) -> Dict:
        domain = base.get("domain", "arxiv")
        if domain == "arxiv":
            data_dir = base.get("output_dirs", {}).get("papers", "./data/papers")
        elif domain == "reddit":
            data_dir = base.get("output_dirs", {}).get("raw", "./data/reddit_raw")
        elif domain == "amazon":
            data_dir = base.get("output_dirs", {}).get("raw", "./data/amazon_raw")
        elif domain in ("steam", "tmdb", "wikiart"):
            data_dir = base.get("output_dirs", {}).get("raw", f"./data/{domain}_raw")
        else:
            data_dir = base.get("output_dirs", {}).get("papers", "./data/papers")

        return {
            "domain": domain,
            "parallel": base.get("parallel", {}).get("enabled", False),
            "num_workers": base.get("parallel", {}).get("num_workers", 8),
            "output_dir": base.get("output_dirs", {}).get("cleaned", "./data/cleaned"),
            "data_dir": data_dir
        }

    def _build_task_ref_graph(self, base: Dict) -> Dict:
        return {
            "metadata_path": base.get("metadata_path"),
            "output_dir": base.get("output_dirs", {}).get("graphs", "./data/graphs"),
            "parallel": base.get("parallel", {}).get("enabled", True),
            "num_workers": base.get("parallel", {}).get("num_workers", 8)
        }

    def _build_task_dataset_formatter(self, base: Dict) -> Dict:
        domain = base.get("domain", "arxiv")
        edges_file = base.get("edges_file", "")

        # 按领域从不同 blackboard 命名空间读取
        if domain == "arxiv" and not edges_file:
            edges_file = self.blackboard.read("graph", "edges_file", "")
        elif domain in ("reddit", "amazon", "steam", "tmdb", "wikiart"):
            edges_file = ""  # 非arxiv域从各自 graph 命名空间读取

        return {
            "domain": domain,
            "graph_mode": base.get("graph_mode", "heterogeneous"),
            "dataset_name": base.get("dataset_name",
                                      f"multimodal_graph_{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            "output_format": base.get("output_format", "json"),
            "output_dir": base.get("output_dirs", {}).get("datasets", "./data/datasets"),
            "include_metadata": base.get("include_metadata", True),
            "edges_file": edges_file,
            "arxiv_unify": base.get("arxiv_unify", False),
            "graph_data_pt": base.get("graph_data_pt"),
        }

    def _build_task_verifier(self, base: Dict) -> Dict:
        """Verifier 任务: 从 blackboard 读图, 对抗性校验, 发现缺陷→发 reprocess。"""
        domain = base.get("domain", "")
        builder_map = {
            "steam": "steam_game_graph_builder",
            "tmdb": "tmdb_movie_graph_builder",
            "wikiart": "wikiart_artwork_graph_builder",
            "arxiv": "reference_graph_builder",
        }
        return {
            "domain": domain,
            "graph_builder_agent": builder_map.get(domain, f"{domain}_graph_builder"),
            "output_dirs": base.get("output_dirs", {}),
        }

    # ==================== Reddit 任务构建 ====================

    def _build_task_reddit(self, agent_name: str, base: Dict) -> Dict:
        common = {
            "domain": "reddit",
            "graph_mode": base.get("graph_mode", "heterogeneous"),
            "data_source": base.get("data_source", "api"),
            "load_from_files": base.get("load_from_files", False),
        }
        base_with_domain = {**base, **common}

        if agent_name == "subreddit_collector":
            return {
                **common,
                "subreddits": base.get("subreddits", []),
                "posts_per_subreddit": base.get("posts_per_subreddit", 100),
                "sort": base.get("sort", "hot"),
                "time_filter": base.get("time_filter", "year"),
                "output_dirs": base.get("output_dirs", {}),
            }
        elif agent_name == "post_collector":
            return {
                **common,
                "comments_per_post": base.get("comments_per_post", 25),
                "comment_depth": base.get("comment_depth", 3),
                "max_user_profiles": base.get("max_user_profiles", 500),
                "output_dirs": base.get("output_dirs", {}),
            }
        elif agent_name == "data_cleaner":
            return self._build_task_data_cleaner(base_with_domain)
        elif agent_name == "social_graph_builder":
            return {
                **common,
                "output_dirs": base.get("output_dirs", {}),
            }
        elif agent_name == "dataset_formatter":
            return self._build_task_dataset_formatter(base_with_domain)
        return base

    # ==================== Amazon 任务构建 ====================

    def _build_task_amazon(self, agent_name: str, base: Dict) -> Dict:
        common = {"domain": "amazon", "graph_mode": base.get("graph_mode", "heterogeneous")}
        base_with_domain = {**base, **common}

        if agent_name == "product_collector":
            return {
                **common,
                "categories": base.get("categories", []),
                "products_per_category": base.get("products_per_category", 5000),
                "require_images": base.get("require_images", True),
                "require_description": base.get("require_description", False),
                "max_products": base.get("max_products", None),
                "max_images_per_product": base.get("max_images_per_product", 3),
                "metadata_path": base.get("metadata_path", None),
                "output_dirs": base.get("output_dirs", {}),
            }
        elif agent_name == "review_collector":
            return {
                **common,
                "categories": base.get("categories", []),
                "reviews_per_product": base.get("reviews_per_product", 10),
                "output_dirs": base.get("output_dirs", {}),
            }
        elif agent_name == "data_cleaner":
            return self._build_task_data_cleaner(base_with_domain)
        elif agent_name == "product_graph_builder":
            return {
                **common,
                "output_dirs": base.get("output_dirs", {}),
            }
        elif agent_name == "dataset_formatter":
            return self._build_task_dataset_formatter(base_with_domain)
        return base

    # ==================== Steam 任务构建 ====================

    def _build_task_steam(self, agent_name: str, base: Dict) -> Dict:
        common = {
            "domain": "steam",
            "graph_mode": base.get("graph_mode", "homogeneous"),
            "data_source": base.get("data_source", "file"),
            "load_from_files": base.get("load_from_files", True),
        }
        base_with_domain = {**base, **common}

        if agent_name == "steam_game_collector":
            return {
                **common,
                "max_games": base.get("max_games", 10000),
                "output_dirs": base.get("output_dirs", {}),
            }
        elif agent_name == "steam_game_graph_builder":
            return {
                **common,
                "graph_opts": base.get("graph_opts", {}),
                "output_dirs": base.get("output_dirs", {}),
            }
        elif agent_name == "dataset_formatter":
            return self._build_task_dataset_formatter(base_with_domain)
        elif agent_name == "graph_verifier":
            return self._build_task_verifier(base_with_domain)
        return base

    # ==================== TMDb 任务构建 ====================

    def _build_task_tmdb(self, agent_name: str, base: Dict) -> Dict:
        common = {
            "domain": "tmdb",
            "graph_mode": base.get("graph_mode", "homogeneous"),
            "data_source": base.get("data_source", "file"),
            "load_from_files": base.get("load_from_files", True),
        }
        base_with_domain = {**base, **common}

        if agent_name == "tmdb_movie_collector":
            return {
                **common,
                "max_movies": base.get("max_movies", 13000),
                "output_dirs": base.get("output_dirs", {}),
            }
        elif agent_name == "tmdb_movie_graph_builder":
            return {
                **common,
                "graph_opts": base.get("graph_opts", {}),
                "output_dirs": base.get("output_dirs", {}),
            }
        elif agent_name == "dataset_formatter":
            return self._build_task_dataset_formatter(base_with_domain)
        elif agent_name == "graph_verifier":
            return self._build_task_verifier(base_with_domain)
        return base

    # ==================== WikiArt 任务构建 ====================

    def _build_task_wikiart(self, agent_name: str, base: Dict) -> Dict:
        common = {
            "domain": "wikiart",
            "graph_mode": base.get("graph_mode", "homogeneous"),
            "data_source": base.get("data_source", "file"),
            "load_from_files": base.get("load_from_files", True),
        }
        base_with_domain = {**base, **common}

        if agent_name == "wikiart_artwork_collector":
            return {
                **common,
                "max_artworks": base.get("max_artworks", 22000),
                "output_dirs": base.get("output_dirs", {}),
            }
        elif agent_name == "wikiart_artwork_graph_builder":
            return {
                **common,
                "graph_opts": base.get("graph_opts", {}),
                "output_dirs": base.get("output_dirs", {}),
            }
        elif agent_name == "dataset_formatter":
            return self._build_task_dataset_formatter(base_with_domain)
        elif agent_name == "graph_verifier":
            return self._build_task_verifier(base_with_domain)
        return base

    def _build_task_co_attribute_collector(self, agent_name: str, base: Dict) -> Dict:
        """通用 co_attribute collector 任务构建 (YAML 动态域 + steam/tmdb/wikiart)。"""
        domain = base.get("domain", "")
        return {
            "domain": domain,
            "graph_mode": base.get("graph_mode", "homogeneous"),
            "data_source": base.get("data_source", "file"),
            "load_from_files": base.get("load_from_files", True),
            "max_items": base.get("max_items", 10000),
            "data_file": base.get("data_file", ""),
            "output_dirs": base.get("output_dirs", {}),
        }

    def _build_task_co_attribute_graph(self, agent_name: str, base: Dict) -> Dict:
        """通用 co_attribute graph builder 任务构建 (YAML 动态域 + steam/tmdb/wikiart)。"""
        domain = base.get("domain", "")
        return {
            "domain": domain,
            "graph_mode": base.get("graph_mode", "homogeneous"),
            "data_source": base.get("data_source", "file"),
            "load_from_files": base.get("load_from_files", True),
            "graph_opts": base.get("graph_opts", {}),
            "output_dirs": base.get("output_dirs", {}),
        }

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
        """判断Agent失败后是否继续 — 按领域获取关键Agent列表"""
        domain = self.blackboard.read("global", "task_config", {}).get("domain", "arxiv")
        domain_cfg = DOMAIN_CONFIGS.get(domain, DOMAIN_CONFIGS["arxiv"])
        critical = domain_cfg.critical_agents

        if any(f in critical for f in failed):
            return False
        return True

    def _build_final_output(self, pipeline_result: Dict, task_config: Dict):
        """构建最终输出 — 通过 DomainConfig 动态发现 agent 名称。"""
        stages = pipeline_result.get("stages", {})
        domain = task_config.get("domain", "arxiv")
        domain_cfg = DOMAIN_CONFIGS.get(domain, DOMAIN_CONFIGS.get("arxiv"))

        # 从 DomainConfig 动态发现 collector/builder 名称
        agent_names = list(domain_cfg.agents_desc.keys()) if domain_cfg else []
        collector = next((a for a in agent_names if "collector" in a), None)
        builder = next((a for a in agent_names if "graph" in a and "builder" in a), None)

        collector_result = stages.get(collector, {}) if collector else {}
        graph_result = stages.get(builder, {}) if builder else {}
        formatter_result = stages.get("dataset_formatter", {})

        fmt_stats = formatter_result.get("stats", {})
        node_count = (fmt_stats.get("node_count")
                      if fmt_stats else len(collector_result.get("papers", [])))

        pipeline_result["final_output"] = {
            "domain": domain,
            "dataset_path": formatter_result.get("output_path", ""),
            "nodes_path": formatter_result.get("output_path", ""),
            "edges_path": graph_result.get("edges_file", "") or graph_result.get("edge_file", ""),
            "node_count": node_count,
            "edge_count": graph_result.get("edge_count", 0) or graph_result.get("total_edges", 0),
            "modality_distribution": fmt_stats.get(
                "modality_distribution", {}
            )
        }

    def _agent_namespace(self, agent_name: str) -> str:
        """获取 agent 的 blackboard 命名空间 — 支持多领域"""
        domain = self.blackboard.read("global", "task_config", {}).get("domain", "arxiv")
        domain_cfg = DOMAIN_CONFIGS.get(domain, DOMAIN_CONFIGS["arxiv"])
        return domain_cfg.namespace_map.get(agent_name, agent_name)
