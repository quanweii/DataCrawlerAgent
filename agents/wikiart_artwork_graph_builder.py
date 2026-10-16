"""
WikiartArtworkGraphBuilderAgent — WikiArt 画作同构图构建 Agent。

域专属薄子类: 所有 ReAct 骨架在 DomainGraphBuilderAgent 中, 这里仅配置 domain+build_fn+config。
shared_style 来源于 target style 字段；raw MAG 可记录该关系，但 benchmark
derived views 必须由 Verifier 按 provenance 排除。
"""
from .domain_graph_builder import DomainGraphBuilderAgent
from crawlers import build_wikiart_graph


class WikiartArtworkGraphBuilderAgent(DomainGraphBuilderAgent):
    """WikiArt 域图构建 Agent — 同构 5-action ReAct, 参数化配置。"""

    def __init__(self, client, system_prompt, mcp_tool_manager=None,
                 agent_name="WikiartArtworkGraphBuilderAgent", blackboard=None,
                 message_bus=None):
        super().__init__(
            client=client, system_prompt=system_prompt,
            mcp_tool_manager=mcp_tool_manager, agent_name=agent_name,
            blackboard=blackboard, message_bus=message_bus,
            domain="wikiart",
            build_fn=build_wikiart_graph,
            config={
                "collector_ns": "wikiart_collector",
                "graph_ns": "wikiart_graph",
                "default_data_dir": "./data/wikiart_artworks_v2",
                "graphs_subdir": "graphs",
                "degree_range": (5, 80),
                "homophily_warn": 0.95,
                "homophily_msg": "高同质仅作诊断；关系去留由字段 provenance 决定。",
            },
        )
