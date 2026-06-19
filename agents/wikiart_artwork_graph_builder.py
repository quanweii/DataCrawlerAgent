"""
WikiartArtworkGraphBuilderAgent — WikiArt 画作同构图构建 Agent (仿 SteamGameGraphBuilderAgent)

读 collector 的 items, 调 crawlers.wikiart.build_wikiart_graph (same_artist/shared_style,
输入已过滤), 产出完整 bundle 写入 wikiart_graph。
注意: WikiArt 同质性高 (~0.89), shared_style 边定义性同质, 链路预测需硬负采样。
"""
import os
import csv
from typing import Dict, Any, List
from loguru import logger

from .react_agent import ReActAgent
from .messages import Blackboard, MessageBus, Message


class WikiartArtworkGraphBuilderAgent(ReActAgent):
    max_iterations = 10

    def __init__(self, client, system_prompt, mcp_tool_manager=None,
                 agent_name="WikiartArtworkGraphBuilderAgent", blackboard=None,
                 message_bus=None):
        super().__init__(
            client=client, system_prompt=system_prompt,
            mcp_tool_manager=mcp_tool_manager, agent_name=agent_name,
            blackboard=blackboard, message_bus=message_bus
        )
        self._current_task = {}

    def get_available_actions(self) -> List[str]:
        return ["load_items", "build_graph", "save_results", "validate_graph", "done"]

    def _default_action(self, observation: Dict) -> Dict[str, Any]:
        results = self._accumulated_results
        if not results.get("items"):
            return {"action": "load_items", "params": {}, "reasoning": "默认: 加载 items"}
        if not results.get("nodes"):
            return {"action": "build_graph", "params": {}, "reasoning": "默认: 构建图"}
        if not results.get("saved"):
            return {"action": "save_results", "params": {}, "reasoning": "默认: 保存结果"}
        return {"action": "validate_graph", "params": {}, "reasoning": "默认: 验证图"}

    def observe(self, task, blackboard, messages):
        observation = super().observe(task, blackboard, messages)
        res = self._accumulated_results
        nodes = res.get("nodes", [])
        edges = res.get("edges", [])
        stats = res.get("stats", {})
        items = res.get("items", [])
        observation.update({"node_count": len(nodes), "edge_count": len(edges),
                            "homophily": stats.get("homophily"),
                            "progress_summary": f"items {len(items)}, 节点 {len(nodes)}, 边 {len(edges)}"})
        # phase_hint 引导 LLM 决策下一步 (同 PaperCollectorAgent 模式, 非写死控制流)
        if not items:
            observation["phase_hint"] = "尚未加载 items, 下一步选择 load_items。"
        elif not nodes:
            observation["phase_hint"] = (
                f"已加载 {len(items)} 个 items, 下一步必须 build_graph 构建同构关系图。"
            )
        elif not res.get("saved"):
            observation["phase_hint"] = (
                f"图已构建 (节点 {len(nodes)} / 边 {len(edges)}), 下一步必须 save_results "
                f"写入 blackboard 与 CSV — 不保存图会丢失!"
            )
        else:
            observation["phase_hint"] = "图已构建并保存, 可选 validate_graph 校验后 done 结束。"
        return observation

    def is_done(self, observation: Dict) -> bool:
        has_saved = any(h["action"] == "save_results" and h["result_status"] == "success"
                        for h in self._action_history)
        if observation.get("edge_count", 0) > 0 and has_saved:
            return True
        return len(self._action_history) >= 6

    def act(self, action_plan: Dict) -> Dict[str, Any]:
        action = action_plan.get("action", "")
        params = action_plan.get("params", {})
        m = {"load_items": self.action_load_items, "build_graph": self.action_build_graph,
             "save_results": self.action_save_results, "validate_graph": self.action_validate_graph}
        h = m.get(action)
        if h:
            return h(params)
        elif action == "done":
            return {"status": "success", "message": "图构建完成"}
        return {"status": "error", "message": f"未知行动: {action}"}

    def action_load_items(self, params: Dict) -> Dict:
        items = []
        cs_lo_dir = "./data/wikiart_artworks_v2"
        if self.blackboard:
            items = self.blackboard.read("wikiart_collector", "items", [])
            cs_lo_dir = self.blackboard.read("wikiart_collector", "cs_lo_dir", cs_lo_dir)
        if not items:
            items = self._current_task.get("items", [])
        self._accumulate_result({"items": items, "cs_lo_dir": cs_lo_dir})
        return {"status": "success", "item_count": len(items), "cs_lo_dir": cs_lo_dir}

    def action_build_graph(self, params: Dict) -> Dict:
        from crawlers import build_wikiart_graph
        items = self._accumulated_results.get("items", [])
        cs_lo_dir = self._accumulated_results.get("cs_lo_dir", "./data/wikiart_artworks_v2")
        opts = self._current_task.get("graph_opts", {})
        nodes, edges, stats = build_wikiart_graph(items, cs_lo_dir, opts)
        self._accumulate_result({"nodes": nodes, "edges": edges, "stats": stats})
        logger.info(f"[WikiartGraphBuilder] 节点 {len(nodes)} 边 {len(edges)} "
                    f"同质 {stats.get('homophily')} 类 {stats.get('num_labels')}")
        return {"status": "success", "node_count": len(nodes), "edge_count": len(edges),
                "homophily": stats.get("homophily")}

    def action_save_results(self, params: Dict) -> Dict:
        nodes = self._accumulated_results.get("nodes", [])
        edges = self._accumulated_results.get("edges", [])
        stats = self._accumulated_results.get("stats", {})
        label_map = {n["attributes"]["label"]: n["attributes"]["label_index"] for n in nodes}
        graphs_dir = self._current_task.get("output_dirs", {}).get("graphs", "./data/wikiart_graphs")
        os.makedirs(graphs_dir, exist_ok=True)
        edge_file = os.path.join(graphs_dir, "co_attribute_edges.csv")
        with open(edge_file, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["source_id", "target_id", "edge_type", "relation"])
            for e in edges:
                w.writerow([e["source"], e["target"], e["type"], e["attributes"].get("relation", "")])
        if self.blackboard:
            self.blackboard.write("wikiart_graph", "nodes", nodes)
            self.blackboard.write("wikiart_graph", "edges", edges)
            self.blackboard.write("wikiart_graph", "stats", stats)
            self.blackboard.write("wikiart_graph", "label_map", label_map)
            self.blackboard.write("wikiart_graph", "edge_file", edge_file)
            self.blackboard.write("wikiart_graph", "total_edges", len(edges))
        self._accumulate_result({"saved": True, "edge_file": edge_file})
        return {"status": "success", "edge_file": edge_file,
                "node_count": len(nodes), "edge_count": len(edges)}

    def action_validate_graph(self, params: Dict) -> Dict:
        stats = self._accumulated_results.get("stats", {})
        deg = stats.get("avg_degree", 0)
        issues = []
        if stats.get("homophily", 0) > 0.95:
            issues.append("同质性过高, shared_style边定义性同质 — 链路预测必须用硬负采样(见 docs/wikiart_lp_protocol.md)")
        if not (5 <= deg <= 80):
            issues.append(f"平均度 {deg} 超出预期区间 [5,80]")
        return {"status": "warning" if issues else "success", "issues": issues,
                "avg_degree": deg, "homophily": stats.get("homophily")}

    def execute(self, task: Dict, upstream_data=None) -> Dict:
        """ReAct 入口 — 经 LLM 驱动的 observe→think→act 循环构建图 (同 arxiv agent)。
        think() 每轮用 LLM 决策下一步; observe() 的 phase_hint 引导决策;
        LLM 失败时 _default_action 兜底 (load→build→save→validate), 保证不丢图。"""
        self._current_task = task
        self._upstream_data = upstream_data
        if not self.blackboard:
            self.blackboard = Blackboard()
        if not self.message_bus:
            self.message_bus = MessageBus()
        return self.run_react_loop(task, self.blackboard, self.message_bus)
