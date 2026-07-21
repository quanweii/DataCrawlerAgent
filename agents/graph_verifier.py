"""
GraphVerifierAgent — 对抗性图质量校验 Agent。

这是多智能体闭环的"对抗性"半场。Builder 构建图以满足 spec; Verifier 主动搜寻
任何违反 spec 的缺陷——包括且不限于:
  - 标签泄露(某类边的高同质性把 train 标签漏到 test)
  - LP-ceiling 异常高(远超多数类基线 = 结构泄露)
  - 节点覆盖缺失(collector 遗漏)
  - 边密度异常(太少/太多)

发现缺陷后 → 用 MessageBus 发 reprocess_request 给 Builder → 编排器回路触发重建。
这是 "multi-agent" 承重的关节: 一个 agent 的错误能被另一个 agent 发现并纠正。
"""

import re
import json
from typing import Dict, Any, List
from collections import defaultdict

import torch
from loguru import logger

from .react_agent import ReActAgent
from .messages import Blackboard, MessageBus, Message


class GraphVerifierAgent(ReActAgent):
    """对抗性图质量校验 —— Builder 的对偶。

    在编排器计划中应排在 formatter 之后; 若发现缺陷则触发 Builder reprocess 回路。
    """

    max_iterations = 8

    def __init__(self, client, system_prompt, mcp_tool_manager=None,
                 agent_name="GraphVerifierAgent", blackboard=None, message_bus=None):
        super().__init__(
            client=client, system_prompt=system_prompt,
            mcp_tool_manager=mcp_tool_manager, agent_name=agent_name,
            blackboard=blackboard, message_bus=message_bus
        )
        self._current_task = {}

    # ==================== ReAct 接口 ====================

    def get_available_actions(self) -> List[str]:
        return ["load_graph", "check_leakage", "check_homophily",
                "check_coverage", "report_issues", "done"]

    def _default_action(self, observation: Dict) -> Dict[str, Any]:
        results = self._accumulated_results
        if not results.get("nodes"):
            return {"action": "load_graph", "params": {},
                    "reasoning": "默认: 加载图数据"}
        if not results.get("leakage_checked"):
            return {"action": "check_leakage", "params": {},
                    "reasoning": "默认: 检测标签泄露"}
        if not results.get("homophily_checked"):
            return {"action": "check_homophily", "params": {},
                    "reasoning": "默认: 检测每类边的同质性"}
        if not results.get("coverage_checked"):
            return {"action": "check_coverage", "params": {},
                    "reasoning": "默认: 检查节点覆盖"}
        return {"action": "report_issues", "params": {},
                "reasoning": "默认: 汇总问题并发送 reprocess"}

    def observe(self, task, blackboard, messages):
        observation = super().observe(task, blackboard, messages)
        res = self._accumulated_results
        observation.update({
            "node_count": len(res.get("nodes", [])),
            "edge_count": len(res.get("edges", [])),
            "leakage_issues": res.get("leakage_issues", []),
            "homo_issues": res.get("homo_issues", []),
            "coverage_issues": res.get("coverage_issues", []),
        })
        return observation

    def is_done(self, observation: Dict) -> bool:
        has_reported = any(
            h["action"] == "report_issues" and h["result_status"] == "success"
            for h in self._action_history)
        if has_reported:
            return True
        return len(self._action_history) >= 6

    def act(self, action_plan: Dict) -> Dict[str, Any]:
        action = action_plan.get("action", "")
        params = action_plan.get("params", {})
        m = {
            "load_graph": self.action_load_graph,
            "check_leakage": self.action_check_leakage,
            "check_homophily": self.action_check_homophily,
            "check_coverage": self.action_check_coverage,
            "report_issues": self.action_report_issues,
        }
        h = m.get(action)
        if h:
            return h(params)
        elif action == "done":
            return {"status": "success", "message": "校验完成"}
        return {"status": "error", "message": f"未知行动: {action}"}

    # ==================== 行动方法 ====================

    def action_load_graph(self, params: Dict) -> Dict:
        """从 blackboard 加载已构建的图数据。"""
        domain = self._current_task.get("domain", "")
        ns_map = {
            "steam": "steam_graph", "tmdb": "tmdb_graph",
            "wikiart": "wikiart_graph", "arxiv": "graph",
        }
        ns = ns_map.get(domain, f"{domain}_graph")
        nodes = self.blackboard.read(ns, "nodes", [])
        edges = self.blackboard.read(ns, "edges", [])
        stats = self.blackboard.read(ns, "stats", {})

        if not nodes:
            # 回退: 从 task config 读 graph 路径, 直接加载
            gpath = self._current_task.get("output_path")
            if gpath:
                import json as _json
                from pathlib import Path
                if Path(gpath).exists():
                    g = _json.load(open(gpath))
                    nodes = g.get("nodes", [])
                    edges = g.get("edges", [])

        self._accumulate_result({
            "nodes": nodes, "edges": edges, "stats": stats,
            "domain": domain,
        })
        return {"status": "success",
                "node_count": len(nodes), "edge_count": len(edges)}

    def action_check_leakage(self, params: Dict) -> Dict:
        """LP-ceiling 双报: 原始图 vs 剪枝图。泄露 = LP-ceiling ≫ 多数类基线。"""
        nodes = self._accumulated_results.get("nodes", [])
        edges = self._accumulated_results.get("edges", [])
        issues = []
        lp_info = {}

        if len(nodes) < 10 or len(edges) < 10:
            self._accumulate_result({"leakage_checked": True, "leakage_issues": [],
                                     "lp_info": {"skipped": "too small"}})
            return {"status": "success", "message": "图太小,跳过 LP-ceiling"}

        try:
            from benchmark.data.splits import (stratified_split,
                prune_cross_split_edges, label_propagation_ceiling)

            id2idx = {n["id"]: i for i, n in enumerate(nodes)}
            y = torch.tensor(
                [n["attributes"].get("label_index",
                 n["attributes"].get("label", "?")) for n in nodes],
                dtype=torch.long)

            # 建 edge_index
            src, dst = [], []
            for e in edges:
                s, t = e["source"], e["target"]
                if s in id2idx and t in id2idx:
                    si, ti = id2idx[s], id2idx[t]
                    src += [si, ti]
                    dst += [ti, si]
            if not src:
                self._accumulate_result({"leakage_checked": True, "leakage_issues": [],
                                         "lp_info": {"skipped": "no valid edges"}})
                return {"status": "success"}
            ei = torch.tensor([src, dst], dtype=torch.long)

            masks = stratified_split(y, (0.6, 0.2, 0.2), seed=1)
            tr, va, te = masks
            lp_orig = label_propagation_ceiling(y, ei, tr, te)
            ei_pruned = prune_cross_split_edges(ei, masks)
            lp_pruned = label_propagation_ceiling(y, ei_pruned, tr, te)

            lp_info = {
                "lp_ceiling_original": lp_orig["acc"],
                "lp_ceiling_pruned": lp_pruned["acc"],
                "majority_baseline": lp_pruned["majority_acc"],
            }

            # 泄露判定: 原始图 LP-ceiling 显著高于多数类 = 结构泄露
            excess = lp_orig["acc"] - lp_orig.get("majority_acc",
                                                   lp_pruned["majority_acc"])
            if excess > 0.10:
                issues.append(
                    f"LP-ceiling {lp_orig['acc']:.3f} >> majority "
                    f"{lp_orig.get('majority_acc', lp_pruned['majority_acc']):.3f} "
                    f"(excess={excess:.3f}) = structural label leakage detected")

            logger.info(
                f"[Verifier] LP-ceiling: orig={lp_orig['acc']:.3f} → "
                f"pruned={lp_pruned['acc']:.3f} (maj={lp_pruned['majority_acc']:.3f}), "
                f"issues={len(issues)}"
            )
        except Exception as e:
            logger.warning(f"[Verifier] LP-ceiling check failed: {e}")
            issues.append(f"LP-ceiling computation error: {e}")

        self._accumulate_result({
            "leakage_checked": True, "leakage_issues": issues,
            "lp_info": lp_info,
        })
        return {"status": "success", "issues_found": len(issues), "lp_info": lp_info}

    def action_check_homophily(self, params: Dict) -> Dict:
        """逐关系计算同质性。homo > 0.95 的关系 = 定义性泄露(标签等价于边类型)。"""
        nodes = self._accumulated_results.get("nodes", [])
        edges = self._accumulated_results.get("edges", [])
        id2lab = {n["id"]: n["attributes"].get("label", "?") for n in nodes}
        rel_counts = defaultdict(lambda: {"same": 0, "total": 0})
        for e in edges:
            rel = e.get("attributes", {}).get("relation", "?")
            rel_counts[rel]["total"] += 1
            if id2lab.get(e["source"]) == id2lab.get(e["target"]):
                rel_counts[rel]["same"] += 1

        per_rel = {}
        issues = []
        exclude_candidates = []
        for rel, c in sorted(rel_counts.items(), key=lambda x: -x[1]["total"]):
            h = c["same"] / max(c["total"], 1)
            per_rel[rel] = {"homophily": round(h, 3), "edges": c["total"]}
            if h > 0.95 and c["total"] >= 50:
                issues.append(
                    f"Relation '{rel}' homophily={h:.3f} ({c['total']} edges) "
                    f"= definitional leakage: edge type ≡ label")
                exclude_candidates.append(rel)

        self._accumulate_result({
            "homophily_checked": True, "homo_issues": issues,
            "per_relation_homophily": per_rel,
            "exclude_candidates": exclude_candidates,
        })
        logger.info(f"[Verifier] per-relation homophily: {per_rel}, "
                    f"exclude_candidates={exclude_candidates}")
        return {"status": "success", "issues_found": len(issues),
                "per_relation": per_rel, "exclude_candidates": exclude_candidates}

    def action_check_coverage(self, params: Dict) -> Dict:
        """检查节点覆盖: 边是否覆盖了所有节点? 孤立节点多少?"""
        nodes = self._accumulated_results.get("nodes", [])
        edges = self._accumulated_results.get("edges", [])
        connected = set()
        for e in edges:
            connected.add(e["source"])
            connected.add(e["target"])
        isolated = [n["id"] for n in nodes if n["id"] not in connected]
        issues = []
        if len(nodes) > 100 and len(isolated) > 0.3 * len(nodes):
            issues.append(f"{len(isolated)}/{len(nodes)} nodes isolated "
                          f"({100*len(isolated)/len(nodes):.1f}%) — "
                          f"collector or edge builder may be incomplete")
        self._accumulate_result({
            "coverage_checked": True, "coverage_issues": issues,
            "connected_nodes": len(connected), "isolated_nodes": len(isolated),
        })
        return {"status": "success", "connected": len(connected),
                "isolated": len(isolated), "issues_found": len(issues)}

    def action_report_issues(self, params: Dict) -> Dict:
        """汇总所有问题, 发 reprocess_request 给对应 agent(通过 MessageBus)。"""
        leakage_issues = self._accumulated_results.get("leakage_issues", [])
        homo_issues = self._accumulated_results.get("homo_issues", [])
        coverage_issues = self._accumulated_results.get("coverage_issues", [])
        all_issues = leakage_issues + homo_issues + coverage_issues

        if not all_issues:
            logger.info("[Verifier] ✅ 图质量检查通过 — 无问题")
            return {"status": "success", "issues_found": 0,
                    "message": "Graph passes all quality checks"}

        # 确定需要排除的关系(来自高同质性检测)
        exclude_relations = self._accumulated_results.get("exclude_candidates", [])

        # 确定 reprocess 目标(图构建 agent)
        target = self._current_task.get("graph_builder_agent", "")
        if not target:
            domain = self._current_task.get("domain", "")
            builder_map = {
                "steam": "steam_game_graph_builder",
                "tmdb": "tmdb_movie_graph_builder",
                "wikiart": "wikiart_artwork_graph_builder",
                "arxiv": "reference_graph_builder",
            }
            target = builder_map.get(domain, f"{domain}_graph_builder")

        if self.message_bus and target:
            self.message_bus.send(Message(
                sender=self.agent_name,
                recipient=target,
                msg_type="reprocess_request",
                content={
                    "reason": "; ".join(all_issues),
                    "exclude_relations": exclude_relations,
                    "lp_info": self._accumulated_results.get("lp_info", {}),
                    "per_relation_homophily": self._accumulated_results.get(
                        "per_relation_homophily", {}),
                },
                priority=2,
            ))
            logger.info(
                f"[Verifier] ⚠️ 发现 {len(all_issues)} 个问题, "
                f"已发 reprocess 到 {target}: exclude={exclude_relations}"
            )

        return {"status": "success", "issues_found": len(all_issues),
                "issues": all_issues, "exclude_relations": exclude_relations,
                "reprocess_target": target}

    # ==================== execute 入口 ====================

    def execute(self, task: Dict, upstream_data=None) -> Dict:
        """ReAct 循环入口。"""
        self._current_task = task
        self._upstream_data = upstream_data
        if not self.blackboard:
            self.blackboard = Blackboard()
        if not self.message_bus:
            self.message_bus = MessageBus()
        return self.run_react_loop(task, self.blackboard, self.message_bus)
