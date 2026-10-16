"""
SteamGameGraphBuilderAgent — Steam 游戏同构图构建 Agent。

域专属薄子类: 所有 ReAct 骨架在 DomainGraphBuilderAgent 中, 这里仅配置 domain+build_fn+config。
"""
from .domain_graph_builder import DomainGraphBuilderAgent
from crawlers import build_steam_graph


class SteamGameGraphBuilderAgent(DomainGraphBuilderAgent):
    """Steam 域图构建 Agent — 同构 5-action ReAct, 参数化配置。"""

    def __init__(self, client, system_prompt, mcp_tool_manager=None,
                 agent_name="SteamGameGraphBuilderAgent", blackboard=None,
                 message_bus=None):
        super().__init__(
            client=client, system_prompt=system_prompt,
            mcp_tool_manager=mcp_tool_manager, agent_name=agent_name,
            blackboard=blackboard, message_bus=message_bus,
            domain="steam",
            build_fn=build_steam_graph,
            config={
                "collector_ns": "steam_collector",
                "graph_ns": "steam_graph",
                "default_data_dir": "./data/steam_games",
                "graphs_subdir": "graphs",
                "degree_range": (5, 80),
                "homophily_warn": 0.95,
                "homophily_msg": "",
            },
        )
