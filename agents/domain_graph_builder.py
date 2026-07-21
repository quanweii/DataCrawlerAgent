"""
DomainGraphBuilderAgent — 参数化图构建 Agent。

消除 steam/tmdb/wikiart 3 个重复 wrapper (之前各 ~155 LOC, 同构的 5-action ReAct 骨架,
仅 domain/build_fn/namespace 不同)。现在域专属只交付 ~5 行子类配置。

域配置 (config dict):
  collector_ns:   str  — blackboard 上 collector 的命名空间 (e.g. "steam_collector")
  graph_ns:       str  — blackboard 上 graph 输出的命名空间 (e.g. "steam_graph")
  default_data_dir: str
  graphs_subdir:  str  — task output_dirs 里的键 (e.g. "graphs")
  degree_range:   (min, max) — validate 的 avg_degree 合理区间 (default (5, 80))
  homophily_warn: float|None — 同质性超此阈值 issue 警告 (default None, 不检查);
                  e.g. wikiart 设 0.95
  homophily_msg:  str|None — 超阈值的额外消息
"""

import os
import csv
from typing import Dict, Any, List, Optional, Tuple, Callable
from loguru import logger

from .react_agent import ReActAgent
from .messages import Blackboard, MessageBus, Message


class DomainGraphBuilderAgent(ReActAgent):
    max_iterations = 10

    def __init__(self, client, system_prompt, mcp_tool_manager=None,
                 agent_name="DomainGraphBuilderAgent", blackboard=None,
                 message_bus=None,
                 domain: str = "unknown",
                 build_fn: Optional[Callable] = None,
                 config: Optional[Dict[str, Any]] = None):
        super().__init__(
            client=client, system_prompt=system_prompt,
            mcp_tool_manager=mcp_tool_manager, agent_name=agent_name,
            blackboard=blackboard, message_bus=message_bus
        )
        self._current_task = {}
        self.domain = domain
        self.build_fn = build_fn
        self.cfg = config or {}
        # defaults
        self.cfg.setdefault("collector_ns", f"{domain}_collector")
        self.cfg.setdefault("graph_ns", f"{domain}_graph")
        self.cfg.setdefault("default_data_dir", f"./data/{domain}_data")
        self.cfg.setdefault("graphs_subdir", "graphs")
        self.cfg.setdefault("degree_range", (5, 80))
        self.cfg.setdefault("homophily_warn", None)
        self.cfg.setdefault("homophily_msg", "")

    # ==================== ReAct 接口 (域无关, 全部共享) ====================

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

    def observe(self, task: Dict, blackboard: Blackboard,
                messages: List[Message]) -> Dict[str, Any]:
        observation = super().observe(task, blackboard, messages)
        res = self._accumulated_results
        nodes = res.get("nodes", [])
        edges = res.get("edges", [])
        stats = res.get("stats", {})
        items = res.get("items", [])
        observation.update({
            "node_count": len(nodes),
            "edge_count": len(edges),
            "homophily": stats.get("homophily"),
            "progress_summary": f"items {len(items)}, 节点 {len(nodes)}, 边 {len(edges)}",
        })
        # phase_hint 引导 LLM 决策下一步 (非写死控制流)
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
        if len(self._action_history) >= 6:
            return True
        return False

    def act(self, action_plan: Dict) -> Dict[str, Any]:
        action = action_plan.get("action", "")
        params = action_plan.get("params", {})
        action_map = {
            "load_items": self.action_load_items,
            "build_graph": self.action_build_graph,
            "save_results": self.action_save_results,
            "validate_graph": self.action_validate_graph,
        }
        handler = action_map.get(action)
        if handler:
            return handler(params)
        elif action == "done":
            return {"status": "success", "message": "图构建完成"}
        return {"status": "error", "message": f"未知行动: {action}"}

    # ==================== 行动方法 (由 domain + config 参数化) ====================

    def action_load_items(self, params: Dict) -> Dict:
        ns = self.cfg["collector_ns"]
        default_dir = self.cfg["default_data_dir"]
        items = []
        data_dir = default_dir
        if self.blackboard:
            items = self.blackboard.read(ns, "items", [])
            data_dir = self.blackboard.read(ns, "data_dir", default_dir)
        if not items:
            items = self._current_task.get("items", [])
        self._accumulate_result({"items": items, "data_dir": data_dir})
        return {"status": "success", "item_count": len(items), "data_dir": data_dir}

    def action_build_graph(self, params: Dict) -> Dict:
        if self.build_fn is None:
            return {"status": "error", "message": "build_fn 未配置 — 必须在子类 __init__ 传入"}
        items = self._accumulated_results.get("items", [])
        data_dir = self._accumulated_results.get("data_dir", self.cfg["default_data_dir"])
        opts = self._current_task.get("graph_opts", {})

        # ---- 多智能体 reprocess 回路 ----
        # Verifier→Builder reprocess: 携带 exclude_relations, 重建时过滤泄露边。
        reprocess = self._current_task.get("reprocess_context", {})
        exclude = reprocess.get("exclude_relations", [])
        if exclude:
            logger.warning(
                f"[{self.domain.capitalize()}GraphBuilder] ⟲ REPROCESS: "
                f"exclude_relations={exclude} "
                f"(reason: {reprocess.get('reason', '')[:120]})"
            )

        nodes, edges, stats = self.build_fn(items, data_dir, opts)

        # 应用 exclude_relations (后过滤, 不修改 crawler SoT)
        if exclude:
            before = len(edges)
            edges = [e for e in edges
                     if e.get("attributes", {}).get("relation") not in exclude]
            if len(edges) < before:
                from crawlers.common import compute_homophily, make_stats
                try:
                    new_stats = make_stats(nodes, edges)
                    stats.update(new_stats)
                except Exception:
                    stats["edge_count"] = len(edges)
            logger.info(
                f"[{self.domain.capitalize()}GraphBuilder] exclude_relations={exclude}: "
                f"{before}→{len(edges)} edges (removed {before-len(edges)})"
            )

        self._accumulate_result({"nodes": nodes, "edges": edges, "stats": stats})
        logger.info(f"[{self.domain.capitalize()}GraphBuilder] 节点 {len(nodes)} 边 {len(edges)} "
                    f"同质 {stats.get('homophily')} 类 {stats.get('num_labels')}")
        return {"status": "success", "node_count": len(nodes), "edge_count": len(edges),
                "homophily": stats.get("homophily")}

    def action_save_results(self, params: Dict) -> Dict:
        ns = self.cfg["graph_ns"]
        nodes = self._accumulated_results.get("nodes", [])
        edges = self._accumulated_results.get("edges", [])
        stats = self._accumulated_results.get("stats", {})
        label_map = {n["attributes"]["label"]: n["attributes"]["label_index"]
                     for n in nodes}

        graphs_subdir = self.cfg["graphs_subdir"]
        graphs_dir = self._current_task.get("output_dirs", {}).get(
            graphs_subdir, f"./data/{self.domain}_{graphs_subdir}")
        os.makedirs(graphs_dir, exist_ok=True)
        edge_file = os.path.join(graphs_dir, "co_attribute_edges.csv")
        with open(edge_file, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["source_id", "target_id", "edge_type", "relation"])
            for e in edges:
                w.writerow([e["source"], e["target"], e["type"],
                            e["attributes"].get("relation", "")])

        if self.blackboard:
            self.blackboard.write(ns, "nodes", nodes)
            self.blackboard.write(ns, "edges", edges)
            self.blackboard.write(ns, "stats", stats)
            self.blackboard.write(ns, "label_map", label_map)
            self.blackboard.write(ns, "edge_file", edge_file)
            self.blackboard.write(ns, "total_edges", len(edges))
        self._accumulate_result({"saved": True, "edge_file": edge_file})
        return {"status": "success", "edge_file": edge_file,
                "node_count": len(nodes), "edge_count": len(edges)}

    def action_validate_graph(self, params: Dict) -> Dict:
        stats = self._accumulated_results.get("stats", {})
        deg = stats.get("avg_degree", 0)
        issues = []
        d_min, d_max = self.cfg["degree_range"]
        if not (d_min <= deg <= d_max):
            issues.append(f"平均度 {deg} 超出预期区间 [{d_min},{d_max}]")
        homo_thr = self.cfg.get("homophily_warn")
        if homo_thr is not None and stats.get("homophily", 0) > homo_thr:
            msg = self.cfg.get("homophily_msg", "")
            issues.append(f"同质性过高(>{homo_thr}), 注意链路预测需硬负采样。{msg}")
        return {"status": "warning" if issues else "success", "issues": issues,
                "avg_degree": deg, "homophily": stats.get("homophily")}

    # ==================== execute 入口 ====================

    def execute(self, task: Dict, upstream_data=None) -> Dict:
        """ReAct 入口 — LLM 驱动的 observe→think→act 循环构建图。
        think() 每轮用 LLM 决策下一步; observe() 的 phase_hint 引导决策;
        LLM 失败时 _default_action 兜底 (load→build→save→validate), 保证不丢图。"""
        self._current_task = task
        self._upstream_data = upstream_data
        if not self.blackboard:
            self.blackboard = Blackboard()
        if not self.message_bus:
            self.message_bus = MessageBus()
        return self.run_react_loop(task, self.blackboard, self.message_bus)
