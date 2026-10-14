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
    ArxivBatchDownloaderAgent, ReferenceGraphBuilderAgent, DataCleanerAgent,
    # Reddit
    SubredditCollectorAgent, PostCollectorAgent, SocialGraphBuilderAgent,
    # Amazon
    ProductCollectorAgent, ReviewCollectorAgent, ProductGraphBuilderAgent,
    # Steam
    SteamGameCollectorAgent, SteamGameGraphBuilderAgent,
    # TMDb
    TmdbMovieCollectorAgent, TmdbMovieGraphBuilderAgent,
    # WikiArt
    WikiartArtworkCollectorAgent, WikiartArtworkGraphBuilderAgent,
    # 对抗性校验 (多智能体闭环关节)
    GraphVerifierAgent, SchemaDiscoveryAgent,
)
from agents.orchestrator_agent import OrchestratorAgent
from tools import create_mcp_tool_manager


class DataCrawlerOrchestrator:
    """
    数据爬取多Agent系统主协调器

    支持三个领域：
    - arxiv: 学术论文多模态属性图
    - reddit: 社交媒体异构图
    - amazon: 电商产品推荐异构图

    内部委托给 OrchestratorAgent 进行动态编排：
    - LLM分析任务 → 动态决定执行顺序
    - 每个Agent运行ReAct循环（观察→思考→行动→观察）
    - Blackboard + MessageBus 实现Agent间通信
    - 运行时LLM决策重试和错误恢复

    公开API与旧版完全兼容。
    """

    def __init__(self, context_manager: Optional[Any] = None, domain: str = "arxiv"):
        self.client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("MODEL_BASE_URL"),
        )
        self.context_manager = context_manager
        self.domain = domain
        self._init_system_prompts()
        self.mcp_tool_manager = create_mcp_tool_manager()
        self._register_custom_tools()
        self._init_agents()
        self.execution_history: List[Dict] = []
        self.current_status: str = "idle"
        logger.info(f"DataCrawlerOrchestrator 初始化完成 (领域={domain}, 多智能体动态编排模式)")

    def _init_system_prompts(self):
        """初始化各Agent的系统提示词"""
        prompt_dir = "prompts"

        def load_prompt_content(name: str) -> str:
            target_path = os.path.join(prompt_dir, f"{name}.txt")
            if os.path.exists(target_path):
                file_path = target_path
            else:
                file_path = os.path.join(prompt_dir, f"{name}_example.txt")
                if not os.path.exists(file_path):
                    return None

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                logger.debug(f"已加载 {name} 提示词，路径: {file_path}")
                return content

        # ArXiv 提示词
        self.paper_collector_prompt = load_prompt_content("paper_collector_prompt") or self._default_paper_collector_prompt()
        self.data_cleaner_prompt = load_prompt_content("data_cleaner_prompt") or self._default_data_cleaner_prompt()
        self.dataset_formatter_prompt = load_prompt_content("dataset_formatter_prompt") or self._default_dataset_formatter_prompt()

        # Reddit 提示词
        self.subreddit_collector_prompt = load_prompt_content("subreddit_collector_prompt") or self._default_subreddit_collector_prompt()
        self.post_collector_prompt = load_prompt_content("post_collector_prompt") or self._default_post_collector_prompt()
        self.social_graph_builder_prompt = load_prompt_content("social_graph_builder_prompt") or self._default_social_graph_builder_prompt()

        # Amazon 提示词
        self.product_collector_prompt = load_prompt_content("product_collector_prompt") or self._default_product_collector_prompt()
        self.review_collector_prompt = load_prompt_content("review_collector_prompt") or self._default_review_collector_prompt()
        self.product_graph_builder_prompt = load_prompt_content("product_graph_builder_prompt") or self._default_product_graph_builder_prompt()

        # Steam 提示词
        self.steam_game_collector_prompt = load_prompt_content("steam_game_collector_prompt") or self._default_steam_game_collector_prompt()
        self.steam_game_graph_builder_prompt = load_prompt_content("steam_game_graph_builder_prompt") or self._default_steam_game_graph_builder_prompt()

        # TMDb 提示词
        self.tmdb_movie_collector_prompt = load_prompt_content("tmdb_movie_collector_prompt") or self._default_tmdb_movie_collector_prompt()
        self.tmdb_movie_graph_builder_prompt = load_prompt_content("tmdb_movie_graph_builder_prompt") or self._default_tmdb_movie_graph_builder_prompt()

        # WikiArt 提示词
        self.wikiart_artwork_collector_prompt = load_prompt_content("wikiart_artwork_collector_prompt") or self._default_wikiart_artwork_collector_prompt()
        self.wikiart_artwork_graph_builder_prompt = load_prompt_content("wikiart_artwork_graph_builder_prompt") or self._default_wikiart_artwork_graph_builder_prompt()
        # 对抗性校验 (多智能体闭环关节)
        self.graph_verifier_prompt = load_prompt_content("graph_verifier_prompt") or self._default_graph_verifier_prompt()

        logger.info(f"成功加载所有提示词 (领域={self.domain})")

    def _register_custom_tools(self):
        pass

    def _init_agents(self):
        """初始化各分管Agent — 按领域动态选择"""
        # 所有领域共享的 Agent
        shared_agents = {
            'schema_discovery': SchemaDiscoveryAgent(
                client=self.client,
                system_prompt=("Profile fields, confirm or propose a graph schema, then "
                               "validate identifier, label and relation provenance deterministically."),
                mcp_tool_manager=self.mcp_tool_manager,
                agent_name="schema_discovery",
            ),
            'data_cleaner': DataCleanerAgent(
                client=self.client,
                system_prompt=self.data_cleaner_prompt,
                mcp_tool_manager=self.mcp_tool_manager
            ),
            'dataset_formatter': DatasetFormatterAgent(
                client=self.client,
                system_prompt=self.dataset_formatter_prompt,
                mcp_tool_manager=self.mcp_tool_manager
            ),
            # 对抗性校验 Agent — 所有域共享, 主动发现图缺陷并触发 Builder reprocess 回路
            'graph_verifier': GraphVerifierAgent(
                client=self.client,
                system_prompt=self.graph_verifier_prompt,
                mcp_tool_manager=self.mcp_tool_manager,
                agent_name="GraphVerifierAgent",
            ),
        }

        # 领域特定 Agent
        if self.domain == "reddit":
            try:
                from tools.reddit_tools import RedditToolManager
                reddit_tools = RedditToolManager()
            except Exception:
                reddit_tools = None
                logger.warning("Reddit 工具初始化失败，将在运行时尝试")

            domain_agents = {
                'subreddit_collector': SubredditCollectorAgent(
                    client=self.client,
                    system_prompt=self.subreddit_collector_prompt,
                    mcp_tool_manager=self.mcp_tool_manager,
                    reddit_tools=reddit_tools,
                ),
                'post_collector': PostCollectorAgent(
                    client=self.client,
                    system_prompt=self.post_collector_prompt,
                    mcp_tool_manager=self.mcp_tool_manager,
                    reddit_tools=reddit_tools,
                ),
                'social_graph_builder': SocialGraphBuilderAgent(
                    client=self.client,
                    system_prompt=self.social_graph_builder_prompt,
                    mcp_tool_manager=self.mcp_tool_manager,
                ),
            }

        elif self.domain == "amazon":
            try:
                from tools.amazon_tools import AmazonToolManager
                amazon_tools = AmazonToolManager()
            except Exception:
                amazon_tools = None
                logger.warning("Amazon 工具初始化失败，将在运行时尝试")

            domain_agents = {
                'product_collector': ProductCollectorAgent(
                    client=self.client,
                    system_prompt=self.product_collector_prompt,
                    mcp_tool_manager=self.mcp_tool_manager,
                    amazon_tools=amazon_tools,
                ),
                'review_collector': ReviewCollectorAgent(
                    client=self.client,
                    system_prompt=self.review_collector_prompt,
                    mcp_tool_manager=self.mcp_tool_manager,
                    amazon_tools=amazon_tools,
                ),
                'product_graph_builder': ProductGraphBuilderAgent(
                    client=self.client,
                    system_prompt=self.product_graph_builder_prompt,
                    mcp_tool_manager=self.mcp_tool_manager,
                ),
            }

        elif self.domain == "steam":
            try:
                from tools.steam_tools import SteamToolManager
                steam_tools = SteamToolManager()
            except Exception:
                steam_tools = None
                logger.warning("Steam 工具初始化失败，将在运行时尝试")

            domain_agents = {
                'steam_game_collector': SteamGameCollectorAgent(
                    client=self.client,
                    system_prompt=self.steam_game_collector_prompt,
                    mcp_tool_manager=self.mcp_tool_manager,
                    steam_tools=steam_tools,
                ),
                'steam_game_graph_builder': SteamGameGraphBuilderAgent(
                    client=self.client,
                    system_prompt=self.steam_game_graph_builder_prompt,
                    mcp_tool_manager=self.mcp_tool_manager,
                ),
            }

        elif self.domain == "tmdb":
            try:
                from tools.tmdb_tools import TmdbToolManager
                tmdb_tools = TmdbToolManager()
            except Exception:
                tmdb_tools = None
                logger.warning("TMDb 工具初始化失败，将在运行时尝试")

            domain_agents = {
                'tmdb_movie_collector': TmdbMovieCollectorAgent(
                    client=self.client,
                    system_prompt=self.tmdb_movie_collector_prompt,
                    mcp_tool_manager=self.mcp_tool_manager,
                    tmdb_tools=tmdb_tools,
                ),
                'tmdb_movie_graph_builder': TmdbMovieGraphBuilderAgent(
                    client=self.client,
                    system_prompt=self.tmdb_movie_graph_builder_prompt,
                    mcp_tool_manager=self.mcp_tool_manager,
                ),
            }

        elif self.domain == "wikiart":
            try:
                from tools.wikiart_tools import WikiartToolManager
                wikiart_tools = WikiartToolManager()
            except Exception:
                wikiart_tools = None
                logger.warning("WikiArt 工具初始化失败，将在运行时尝试")

            domain_agents = {
                'wikiart_artwork_collector': WikiartArtworkCollectorAgent(
                    client=self.client,
                    system_prompt=self.wikiart_artwork_collector_prompt,
                    mcp_tool_manager=self.mcp_tool_manager,
                    wikiart_tools=wikiart_tools,
                ),
                'wikiart_artwork_graph_builder': WikiartArtworkGraphBuilderAgent(
                    client=self.client,
                    system_prompt=self.wikiart_artwork_graph_builder_prompt,
                    mcp_tool_manager=self.mcp_tool_manager,
                ),
            }

        else:  # arxiv (默认)
            domain_agents = {
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
            }

        self.agents = {**domain_agents, **shared_agents}
        logger.info(
            f"所有分管Agent初始化完成: {list(self.agents.keys())} "
            f"(领域={self.domain})"
        )

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

    # ==================== Reddit 默认提示词 ====================

    def _default_subreddit_collector_prompt(self) -> str:
        return """你是一个Reddit subreddit数据采集专家。你的任务是：
1. 发现和验证目标subreddit
2. 采集帖子元数据（标题、得分、图片URL、评论数等）
3. 确保采集足够数量和多样性的帖子
4. 验证数据完整性，报告质量问题"""

    def _default_post_collector_prompt(self) -> str:
        return """你是一个Reddit帖子深度采集专家。你的任务是：
1. 采集帖子评论树（BFS深度限制）
2. 下载帖子图片
3. 收集用户元数据（karma等）
4. 确保评论覆盖度和图片下载成功率"""

    def _default_social_graph_builder_prompt(self) -> str:
        return """你是一个社交图构建专家。你的任务是：
1. 从帖子、评论数据中提取所有实体（用户、帖子、评论、subreddit）
2. 构建 author_of 边（用户→帖子、用户→评论）
3. 构建 reply_to 边（评论→帖子、评论→评论）
4. 构建 belongs_to 边（帖子→subreddit）
5. 构建 crosspost 边（帖子→帖子）
6. 输出各类型边的CSV文件"""

    # ==================== Amazon 默认提示词 ====================

    def _default_product_collector_prompt(self) -> str:
        return """你是一个Amazon产品数据采集专家。你的任务是：
1. 加载McAuley Amazon产品元数据（JSON Lines格式）
2. 按品类过滤产品，确保有图片和描述
3. 下载产品图片
4. 提取共买(also_bought)和共看(also_viewed)关系
5. 提取类目树和品牌信息
6. 验证数据完整性"""

    def _default_review_collector_prompt(self) -> str:
        return """你是一个Amazon评论采集专家。你的任务是：
1. 加载目标产品的评论数据
2. 过滤低质量评论（过短、无意义等）
3. 限制每产品评论数量，按帮助度排序
4. 提取唯一用户ID
5. 验证评论覆盖度"""

    def _default_product_graph_builder_prompt(self) -> str:
        return """你是一个产品图构建专家。你的任务是：
1. 从产品元数据和评论中提取所有实体（产品、用户、类目、品牌）
2. 构建 review 边（用户→产品，含评分）
3. 构建 also_bought 边（产品→产品，共买关系）
4. 构建 also_viewed 边（产品→产品，共看关系）
5. 构建 belongs_to 边（产品→类目）
6. 构建 made_by 边（产品→品牌）
7. 输出各类型边的CSV文件"""

    # ==================== Steam 默认提示词 ====================

    def _default_steam_game_collector_prompt(self) -> str:
        return """你是一个Steam游戏元数据采集专家。你的任务是：
1. 加载/爬取Steam游戏元数据（genres/developers/publishers/categories）
2. 确保每个游戏的cs_lo提取目录（markdown+图片）就绪
3. 验证游戏数据质量（标签覆盖率、目录完整性）
4. 将采集结果写入blackboard供下游图构建Agent使用"""

    def _default_steam_game_graph_builder_prompt(self) -> str:
        return """你是一个Steam游戏同构图构建专家。你的任务是：
1. 从collector读取游戏items
2. 构建Steam游戏同构图：用首个genre作为节点标签，过滤<20样本的小类
3. 构建密度优化的co_attribute边（shared_genre/same_developer/same_publisher/shared_category），
   跨关系去重，控制边密度使平均度在合理区间
4. 计算同质性homophily，输出带label/label_index的节点和边bundle
5. 将图结果写入blackboard供formatter封装"""

    # ==================== TMDb 默认提示词 ====================

    def _default_tmdb_movie_collector_prompt(self) -> str:
        return """你是一个TMDb电影元数据采集专家。你的任务是：
1. 加载/爬取TMDb电影元数据（genres/directors/actors）
2. 确保每部电影的cs_lo提取目录（markdown+海报图片）就绪
3. 验证数据质量（流派标签覆盖率、目录完整性）
4. 将采集结果写入blackboard供下游图构建Agent使用"""

    def _default_tmdb_movie_graph_builder_prompt(self) -> str:
        return """你是一个TMDb电影同构图构建专家。你的任务是：
1. 从collector读取电影items
2. 构建TMDb电影同构图：用首个genre作为节点标签，过滤<20样本的小类
3. 构建密度优化的co_attribute边（shared_genre/same_director/shared_actor），
   跨关系去重，控制边密度
4. 计算同质性homophily（TMDb预期为异质~0.24），输出带label/label_index的节点和边bundle
5. 将图结果写入blackboard供formatter封装"""

    # ==================== WikiArt 默认提示词 ====================

    def _default_wikiart_artwork_collector_prompt(self) -> str:
        return """你是一个WikiArt画作元数据采集专家。你的任务是：
1. 加载/爬取WikiArt画作元数据（style/artist/title）
2. 确保每幅画作的cs_lo提取目录（markdown文本+高清图片）就绪
3. 验证数据质量（风格标签覆盖率、目录完整性）
4. 将采集结果写入blackboard供下游图构建Agent使用"""

    def _default_wikiart_artwork_graph_builder_prompt(self) -> str:
        return """你是一个WikiArt画作同构图构建专家。你的任务是：
1. 从collector读取画作items
2. 构建WikiArt画作同构图：用style作为节点标签
3. 构建密度优化的co_attribute边（same_artist/shared_style），跨关系去重
4. 计算同质性homophily（WikiArt预期为高同质~0.89）
   注意：shared_style边定义性同质，链路预测必须用硬负采样（见docs/wikiart_lp_protocol.md）
5. 输出带label/label_index的节点和边bundle，写入blackboard供formatter封装"""

    def _default_graph_verifier_prompt(self) -> str:
        return """你是一个对抗性图质量校验专家。你的角色是Builder的对偶:
Builder构建图以"满足spec"为目标;你以"找到任何违反spec的缺陷"为目标。

你必须逐项检查:
1. 加载图数据(从blackboard)
2. 测量 topology-only predictability: 计算 LP probe（full topology vs
   split-disjoint）。高分只说明图结构具有标签预测力，不能单独判为泄露。
3. 生成四个协议视图：transductive、split-disjoint、temporal、
   group-inductive；raw MAG 保持不可变。
4. 只有 provenance 证实的 target-derived 字段/关系、重复实体、未来信息，
   或明确违反所声明 protocol 时，才能判为 integrity defect 并触发重建。
5. 测量逐关系同质性，但只有关系 lineage 明确来自 target 字段时才标记为
   target-derived defect；高同质本身不是充分证据。
6. 检查节点覆盖: 是否有大量孤立节点(>30%) — 说明collector或edge builder不完整。
7. 汇总所有已验证问题。若有问题，发reprocess_request给图构建agent
   (携带exclude_relations列表)，触发重建回路。

你是多智能体质量闭环的关键一环：负责区分真实完整性缺陷、协议依赖的信息暴露
和正常结构信号，并产出可复现 protocol views 与 integrity card。
对wikiart: shared_style 来自 target 字段 style，所以应由 provenance 判定并触发重建；
same_artist 即使高同质，也只能作为 topology signal 记录。"""

    # ==================== YAML 驱动域支持 ====================

    def _default_generic_collector_prompt(self) -> str:
        return """你是一个数据元数据采集专家。你的任务是:
1. 加载领域元数据(标签、属性等)
2. 确保每个条目的cs_lo提取目录(markdown+图片)就绪
3. 验证数据质量(标签覆盖率、目录完整性)
4. 将采集结果写入blackboard供下游图构建Agent使用"""

    def _default_generic_builder_prompt(self) -> str:
        return """你是一个同构图构建专家。你的任务是:
1. 从collector读取items
2. 构建同构图:用label作为节点标签，过滤小类
3. 构建密度优化的co_attribute边，跨关系去重，控制边密度
4. 计算同质性homophily，输出带label/label_index的节点和边
5. 将图结果写入blackboard供formatter封装"""

    def _init_agents_from_yaml(self, yc):
        """从 DomainYamlConfig 动态初始化域 agent 池。

        对已内置 Collector 的域 (steam/tmdb/wikiart): 使用原 Collector,
        GraphBuilder 使用 build_graph_from_config。
        对全新域: 使用 DomainCollectorAgent + DomainGraphBuilderAgent。
        """
        from agents.domain_graph_builder import DomainGraphBuilderAgent
        from agents.schema_discovery import SchemaDiscoveryAgent
        from crawlers.config_driven import build_graph_from_config
        from domains.domain_schema import yaml_to_graph_config

        domain = yc.domain
        graph_config = yaml_to_graph_config(yc)

        # --- Collector ---
        collector_name = yc.agents.collector_agent_name or f"{domain}_collector"
        collector_prompt = (yc.collector_prompt or
                           self._load_prompt_for(collector_name) or
                           self._default_generic_collector_prompt())

        # 优先用内置 Collector (向后兼容), 否则用通用 DomainCollectorAgent
        builtin_collectors = {
            "steam": ("SteamGameCollectorAgent", "steam_tools", "SteamToolManager"),
            "tmdb": ("TmdbMovieCollectorAgent", "tmdb_tools", "TmdbToolManager"),
            "wikiart": ("WikiartArtworkCollectorAgent", "wikiart_tools", "WikiartToolManager"),
        }
        if domain in builtin_collectors:
            cls_name, tool_mod, tool_cls = builtin_collectors[domain]
            from agents import (SteamGameCollectorAgent, TmdbMovieCollectorAgent,
                              WikiartArtworkCollectorAgent)
            cls_map = {"SteamGameCollectorAgent": SteamGameCollectorAgent,
                      "TmdbMovieCollectorAgent": TmdbMovieCollectorAgent,
                      "WikiartArtworkCollectorAgent": WikiartArtworkCollectorAgent}
            collector_cls = cls_map[cls_name]
            # Load tool manager
            tool_manager = None
            try:
                import importlib
                mod = importlib.import_module(f"tools.{tool_mod}")
                tool_manager = getattr(mod, tool_cls)()
            except Exception as e:
                logger.warning(f"Tool manager {tool_cls} not available: {e}")
            collector = collector_cls(
                client=self.client, system_prompt=collector_prompt,
                mcp_tool_manager=self.mcp_tool_manager)
            if tool_manager:
                setattr(collector, f"{domain}_tools", tool_manager)
        else:
            from agents.domain_collector import DomainCollectorAgent
            collector_config = {
                "domain": domain,
                "agent_name": collector_name,
                "id_field": yc.data.id_field,
                "label_field": yc.data.label_field,
                "data_source": yc.data.data_source,
                "data_file": yc.data.data_file,
                "data_dir": yc.output_dirs.raw,
                "max_items": yc.data.max_items,
                "collector_ns": f"{domain}_collector",
                "cslo_dir_key": "items",
            }
            if yc.agents.tool_manager_class:
                collector_config["tool_manager_class"] = yc.agents.tool_manager_class
                collector_config["max_items_param"] = f"max_{domain}s"
            collector = DomainCollectorAgent(
                client=self.client, system_prompt=collector_prompt,
                mcp_tool_manager=self.mcp_tool_manager,
                agent_name=collector_name, config=collector_config)

        # --- Graph Builder ---
        builder_name = yc.agents.builder_agent_name or f"{domain}_graph_builder"
        builder_prompt = (yc.builder_prompt or
                         self._load_prompt_for(builder_name) or
                         self._default_generic_builder_prompt())
        def yaml_build(items, cs_lo_dir, opts=None):
            local_opts = dict(opts or {})
            runtime_schema = local_opts.pop("_schema_config", graph_config)
            return build_graph_from_config(
                items, cs_lo_dir, runtime_schema, local_opts)

        builder = DomainGraphBuilderAgent(
            client=self.client, system_prompt=builder_prompt,
            mcp_tool_manager=self.mcp_tool_manager,
            domain=domain,
            # Wrap build_graph_from_config so the 3-arg signature (items, cs_lo_dir, opts)
            # used by DomainGraphBuilderAgent.build_action is compatible with the 4-arg
            # signature (items, cs_lo_dir, config, opts) required by build_graph_from_config.
            build_fn=yaml_build,
            config={
                "collector_ns": f"{domain}_collector",
                "graph_ns": f"{domain}_graph",
                "default_data_dir": yc.output_dirs.items,
                "graphs_subdir": "graphs",
                "degree_range": (5, 80),
            })

        # Store graph_config on builder for build_fn to access
        builder._graph_config = graph_config

        schema_agent = SchemaDiscoveryAgent(
            client=self.client,
            system_prompt=("Profile fields, confirm or propose a graph schema, and publish "
                           "it only after deterministic provenance validation."),
            mcp_tool_manager=self.mcp_tool_manager,
            agent_name="schema_discovery")

        return {
            collector_name: collector,
            "schema_discovery": schema_agent,
            builder_name: builder,
        }

    def _load_prompt_for(self, agent_name: str) -> str:
        """尝试从 prompts/ 加载 agent 的 prompt 文件。"""
        from pathlib import Path
        prompt_path = Path(__file__).resolve().parent / "prompts" / f"{agent_name}_prompt.txt"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return ""


def load_orchestrator_from_yaml(yaml_path: str) -> DataCrawlerOrchestrator:
    """从 YAML 域配置创建 orchestrator (模块级工厂函数)。

    Usage:
        orch = load_orchestrator_from_yaml("domains/spotify/domain.yaml")
        result = orch.execute_pipeline(task_config)
    """
    from domains.domain_schema import load_domain_yaml, yaml_to_domain_config
    from agents.orchestrator_agent import register_domain_config

    yc = load_domain_yaml(yaml_path)

    # 注册 DomainConfig (供 OrchestratorAgent 调度)
    domain_cfg = yaml_to_domain_config(yc)
    register_domain_config(yc.domain, domain_cfg)

    # 创建 orchestrator
    orch = DataCrawlerOrchestrator(domain=yc.domain)
    orch._domain_yaml_config = yc

    # 用 YAML 驱动的 agent 替换硬编码 agent
    yaml_agents = orch._init_agents_from_yaml(yc)
    orch.agents = {
        **yaml_agents,
        "data_cleaner": orch.agents.get("data_cleaner"),
        "schema_discovery": yaml_agents.get("schema_discovery"),
        "dataset_formatter": orch.agents.get("dataset_formatter"),
        "graph_verifier": orch.agents.get("graph_verifier"),
    }

    logger.info(f"[YAML domain] {yc.domain}: {len(yc.relations)} relations, "
                f"collector={'builtin' if yc.domain in ('steam','tmdb','wikiart') else 'generic'}")
    return orch


def create_orchestrator() -> DataCrawlerOrchestrator:
    return DataCrawlerOrchestrator()
