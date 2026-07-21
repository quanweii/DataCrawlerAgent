"""
TmdbMovieGraphBuilderAgent — TMDb 电影同构图构建 Agent。

域专属薄子类: 所有 ReAct 骨架在 DomainGraphBuilderAgent 中, 这里仅配置 domain+build_fn+config。
(无 homophily 警告: TMDb 同质性 ~0.24, 异质 → 链路预测无同质泄露问题)
"""
from .domain_graph_builder import DomainGraphBuilderAgent
from crawlers import build_tmdb_graph


class TmdbMovieGraphBuilderAgent(DomainGraphBuilderAgent):
    """TMDb 域图构建 Agent — 同构 5-action ReAct, 参数化配置。"""

    def __init__(self, client, system_prompt, mcp_tool_manager=None,
                 agent_name="TmdbMovieGraphBuilderAgent", blackboard=None,
                 message_bus=None):
        super().__init__(
            client=client, system_prompt=system_prompt,
            mcp_tool_manager=mcp_tool_manager, agent_name=agent_name,
            blackboard=blackboard, message_bus=message_bus,
            domain="tmdb",
            build_fn=build_tmdb_graph,
            config={
                "collector_ns": "tmdb_collector",
                "graph_ns": "tmdb_graph",
                "default_data_dir": "./data/tmdb_movies",
                "graphs_subdir": "graphs",
                "degree_range": (5, 80),
                "homophily_warn": None,  # TMDb 异质, 无同质泄露风险
            },
        )
