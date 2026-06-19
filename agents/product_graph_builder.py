"""
ProductGraphBuilderAgent — Amazon 产品图构建 Agent
从产品元数据和评论中构建 Product-User-Category-Brand 异构图
或 MAGB 兼容的同构图（product 节点 + co-purchasing 边）
输出多类型边 CSV 文件
"""
import os
import csv
import json
from typing import Dict, Any, List, Set
from collections import defaultdict
from loguru import logger

from .react_agent import ReActAgent
from .messages import Blackboard, MessageBus, Message


class ProductGraphBuilderAgent(ReActAgent):
    """
    Amazon 产品图构建 Agent — 支持双模建图

    graph_mode="heterogeneous" (默认):
      - review: user→product (含评分、评论文本长度)
      - also_bought: product→product (共买关系)
      - also_viewed: product→product (共看关系)
      - belongs_to: product→category (类目归属)
      - made_by: product→brand (品牌归属)

    graph_mode="homogeneous" (MAGB兼容):
      - co_purchase: product→product (also_buy + also_view 合并, 无向边)
      - 标签 = 叶品类 (category_path 最后一级)

    可用行动:
      异构模式: collect_entities, build_review_edges, build_copurchase_edges,
               build_coview_edges, build_category_edges, build_brand_edges,
               save_edges, validate_graph
      同构模式: collect_products, build_copurchase_edges_homogeneous,
               save_edges, validate_graph
      通用: done
    """

    max_iterations = 10

    def __init__(self, client, system_prompt, mcp_tool_manager=None,
                 agent_name="ProductGraphBuilderAgent", blackboard=None,
                 message_bus=None):
        super().__init__(
            client=client, system_prompt=system_prompt,
            mcp_tool_manager=mcp_tool_manager, agent_name=agent_name,
            blackboard=blackboard, message_bus=message_bus
        )
        self._current_task = {}
        self.graph_mode = "heterogeneous"
        self._edges: Dict[str, List[Dict]] = defaultdict(list)
        self._entities: Dict[str, Set[str]] = defaultdict(set)
        # 同构模式标签映射
        self._label_map: Dict[str, str] = {}

    def get_available_actions(self) -> List[str]:
        return [
            "collect_entities",
            "collect_products",
            "build_review_edges",
            "build_copurchase_edges",
            "build_coview_edges",
            "build_category_edges",
            "build_brand_edges",
            "build_copurchase_edges_homogeneous",
            "save_edges",
            "validate_graph",
            "done"
        ]

    def observe(self, task: Dict, blackboard: Blackboard,
                messages: List[Message]) -> Dict[str, Any]:
        observation = super().observe(task, blackboard, messages)

        edge_counts = {k: len(v) for k, v in self._edges.items()}
        entity_counts = {k: len(v) for k, v in self._entities.items()}

        observation.update({
            "graph_mode": self.graph_mode,
            "edge_counts": edge_counts,
            "entity_counts": entity_counts,
            "total_edges": sum(edge_counts.values()),
            "progress_summary": (
                f"模式: {self.graph_mode}, "
                f"实体: {entity_counts}, "
                f"边: {edge_counts}"
            ),
        })

        return observation

    def is_done(self, observation: Dict) -> bool:
        total_edges = observation.get("total_edges", 0)
        has_saved = any(
            h["action"] == "save_edges" and h["result_status"] == "success"
            for h in self._action_history
        )
        if total_edges > 0 and has_saved:
            return True
        if len(self._action_history) >= 10:
            return True
        return False

    def _default_action(self, observation: Dict) -> Dict[str, Any]:
        """按阶段推进的默认行动 — 根据 graph_mode 分叉"""
        if self.graph_mode == "homogeneous":
            return self._default_action_homogeneous(observation)
        else:
            return self._default_action_heterogeneous(observation)

    def _default_action_heterogeneous(self, observation: Dict) -> Dict[str, Any]:
        """异构模式的默认行动序列"""
        edge_counts = {k: len(v) for k, v in self._edges.items()}
        entity_counts = {k: len(v) for k, v in self._entities.items()}

        if not entity_counts:
            return {"action": "collect_entities", "params": {},
                    "reasoning": "默认: 先收集实体"}
        elif "review" not in edge_counts:
            return {"action": "build_review_edges", "params": {},
                    "reasoning": "默认: 构建review边"}
        elif "also_bought" not in edge_counts:
            return {"action": "build_copurchase_edges", "params": {},
                    "reasoning": "默认: 构建共买边"}
        elif "also_viewed" not in edge_counts:
            return {"action": "build_coview_edges", "params": {},
                    "reasoning": "默认: 构建共看边"}
        elif "belongs_to" not in edge_counts:
            return {"action": "build_category_edges", "params": {},
                    "reasoning": "默认: 构建类目边"}
        elif "made_by" not in edge_counts:
            return {"action": "build_brand_edges", "params": {},
                    "reasoning": "默认: 构建品牌边"}
        elif not any(h["action"] == "save_edges" and h["result_status"] == "success"
                     for h in self._action_history):
            return {"action": "save_edges", "params": {},
                    "reasoning": "默认: 保存边文件"}
        else:
            return {"action": "validate_graph", "params": {},
                    "reasoning": "默认: 验证图"}

    def _default_action_homogeneous(self, observation: Dict) -> Dict[str, Any]:
        """同构模式的默认行动序列"""
        edge_counts = {k: len(v) for k, v in self._edges.items()}
        has_products = len(self._entities.get("product", set())) > 0

        if not has_products:
            return {"action": "collect_products", "params": {},
                    "reasoning": "默认: 收集产品节点"}
        elif not edge_counts.get("co_purchase") and "co_purchase" not in edge_counts:
            return {"action": "build_copurchase_edges_homogeneous", "params": {},
                    "reasoning": "默认: 构建co-purchasing边"}
        elif not any(h["action"] == "save_edges" and h["result_status"] == "success"
                     for h in self._action_history):
            return {"action": "save_edges", "params": {},
                    "reasoning": "默认: 保存边文件"}
        else:
            return {"action": "validate_graph", "params": {},
                    "reasoning": "默认: 验证图"}

    def act(self, action_plan: Dict) -> Dict[str, Any]:
        action = action_plan.get("action", "")
        params = action_plan.get("params", {})

        action_map = {
            "collect_entities": self.action_collect_entities,
            "collect_products": self.action_collect_products,
            "build_review_edges": self.action_build_review_edges,
            "build_copurchase_edges": self.action_build_copurchase_edges,
            "build_coview_edges": self.action_build_coview_edges,
            "build_category_edges": self.action_build_category_edges,
            "build_brand_edges": self.action_build_brand_edges,
            "build_copurchase_edges_homogeneous": self.action_build_copurchase_edges_homogeneous,
            "save_edges": self.action_save_edges,
            "validate_graph": self.action_validate_graph,
        }

        handler = action_map.get(action)
        if handler:
            return handler(params)
        elif action == "done":
            return {"status": "success", "message": "任务完成"}
        else:
            return {"status": "error", "message": f"未知行动: {action}"}

    # ==================== 数据获取 ====================

    def _get_products(self) -> Dict[str, Dict]:
        """获取产品列表"""
        if self.blackboard:
            products = self.blackboard.read("product_collector", "products", {})
            if products:
                return products
        return self._current_task.get("products", {})

    def _get_reviews(self) -> Dict[str, List[Dict]]:
        """获取评论"""
        if self.blackboard:
            reviews = self.blackboard.read("review_collector", "reviews", {})
            if reviews:
                return reviews
        return self._current_task.get("reviews", {})

    # ==================== 异构模式行动 ====================

    def action_collect_entities(self, params: Dict) -> Dict:
        """收集所有实体（异构模式）"""
        products = self._get_products()
        reviews = self._get_reviews()

        for asin in products:
            self._entities["product"].add(asin)
            brand = products[asin].get("brand", "")
            if brand:
                self._entities["brand"].add(brand)

        # 从评论中收集用户
        for asin, review_list in reviews.items():
            for review in review_list:
                rid = review.get("reviewer_id", "")
                if rid:
                    self._entities["user"].add(rid)

        # 收集类目
        for asin, meta in products.items():
            cat_path = meta.get("category_path", "")
            if cat_path:
                parts = [p.strip() for p in cat_path.split("|") if p.strip()]
                for part in parts:
                    self._entities["category"].add(part)

        entity_counts = {k: len(v) for k, v in self._entities.items()}
        logger.info(f"[ProductGraphBuilder] 实体收集(异构): {entity_counts}")

        return {
            "status": "success",
            "message": f"收集 {sum(entity_counts.values())} 个实体",
            "entity_counts": entity_counts,
        }

    def action_build_review_edges(self, params: Dict) -> Dict:
        """构建 review 边 (user→product)"""
        # 确保 defaultdict 键存在（即使无边也需要标记已构建）
        _ = self._edges["review"]

        reviews = self._get_reviews()
        products = self._get_products()
        product_asins = set(products.keys())

        count = 0
        for asin, review_list in reviews.items():
            if asin not in product_asins:
                continue
            for review in review_list:
                self._edges["review"].append({
                    "source_id": f"user_{review['reviewer_id']}",
                    "source_type": "user",
                    "target_id": f"product_{asin}",
                    "target_type": "product",
                    "edge_type": "review",
                    "attributes": json.dumps({
                        "rating": review.get("rating", 0),
                        "review_text_length": review.get("review_text_length", 0),
                        "helpful_votes": review.get("helpful_votes", 0),
                        "verified": review.get("verified", False),
                        "timestamp": review.get("timestamp", 0),
                    })
                })
                count += 1

        logger.info(f"[ProductGraphBuilder] review 边: {count}")
        return {
            "status": "success",
            "message": f"构建 {count} 条 review 边",
            "edge_count": count,
        }

    def action_build_copurchase_edges(self, params: Dict) -> Dict:
        """构建 also_bought 边 (product→product)"""
        _ = self._edges["also_bought"]
        products = self._get_products()
        product_asins = set(products.keys())

        count = 0
        for asin, meta in products.items():
            for related in meta.get("also_buy", []):
                if related in product_asins:
                    self._edges["also_bought"].append({
                        "source_id": f"product_{asin}",
                        "source_type": "product",
                        "target_id": f"product_{related}",
                        "target_type": "product",
                        "edge_type": "also_bought",
                        "attributes": "{}"
                    })
                    count += 1

        logger.info(f"[ProductGraphBuilder] also_bought 边: {count}")
        return {
            "status": "success",
            "message": f"构建 {count} 条 also_bought 边",
            "edge_count": count,
        }

    def action_build_coview_edges(self, params: Dict) -> Dict:
        """构建 also_viewed 边 (product→product)"""
        _ = self._edges["also_viewed"]
        products = self._get_products()
        product_asins = set(products.keys())

        count = 0
        for asin, meta in products.items():
            for related in meta.get("also_view", []):
                if related in product_asins:
                    self._edges["also_viewed"].append({
                        "source_id": f"product_{asin}",
                        "source_type": "product",
                        "target_id": f"product_{related}",
                        "target_type": "product",
                        "edge_type": "also_viewed",
                        "attributes": "{}"
                    })
                    count += 1

        logger.info(f"[ProductGraphBuilder] also_viewed 边: {count}")
        return {
            "status": "success",
            "message": f"构建 {count} 条 also_viewed 边",
            "edge_count": count,
        }

    def action_build_category_edges(self, params: Dict) -> Dict:
        """构建 belongs_to 边 (product→category)"""
        _ = self._edges["belongs_to"]
        products = self._get_products()

        count = 0
        for asin, meta in products.items():
            cat_path = meta.get("category_path", "")
            if not cat_path:
                continue

            parts = [p.strip() for p in cat_path.split("|") if p.strip()]
            if parts:
                leaf_cat = parts[-1]
                self._edges["belongs_to"].append({
                    "source_id": f"product_{asin}",
                    "source_type": "product",
                    "target_id": f"cat_{leaf_cat}",
                    "target_type": "category",
                    "edge_type": "belongs_to",
                    "attributes": json.dumps({
                        "category_level": len(parts) - 1
                    })
                })
                count += 1

        logger.info(f"[ProductGraphBuilder] belongs_to 边: {count}")
        return {
            "status": "success",
            "message": f"构建 {count} 条 belongs_to 边",
            "edge_count": count,
        }

    def action_build_brand_edges(self, params: Dict) -> Dict:
        """构建 made_by 边 (product→brand)"""
        _ = self._edges["made_by"]
        products = self._get_products()

        count = 0
        for asin, meta in products.items():
            brand = meta.get("brand", "")
            if brand:
                self._edges["made_by"].append({
                    "source_id": f"product_{asin}",
                    "source_type": "product",
                    "target_id": f"brand_{brand}",
                    "target_type": "brand",
                    "edge_type": "made_by",
                    "attributes": "{}"
                })
                count += 1

        logger.info(f"[ProductGraphBuilder] made_by 边: {count}")
        return {
            "status": "success",
            "message": f"构建 {count} 条 made_by 边",
            "edge_count": count,
        }

    # ==================== 同构模式行动 (MAGB兼容) ====================

    def action_collect_products(self, params: Dict) -> Dict:
        """
        收集产品节点 + 构建标签映射（同构模式）

        算法：
        1. 遍历所有产品，收集 ASIN 集合
        2. 标签 = category_path 的最后一级（叶品类）
        """
        products = self._get_products()
        self._label_map = {}

        for asin in products:
            self._entities["product"].add(asin)

            # 标签 = 叶品类
            cat_path = products[asin].get("category_path", "")
            if cat_path:
                parts = [p.strip() for p in cat_path.split("|") if p.strip()]
                if parts:
                    self._label_map[asin] = parts[-1]

        entity_counts = {k: len(v) for k, v in self._entities.items()}
        logger.info(
            f"[ProductGraphBuilder] 同构模式收集: "
            f"{len(self._entities['product'])} 产品, "
            f"{len(self._label_map)} 有标签"
        )

        return {
            "status": "success",
            "message": f"收集 {len(self._entities['product'])} 个产品节点",
            "entity_counts": entity_counts,
            "labeled_products": len(self._label_map),
        }

    def action_build_copurchase_edges_homogeneous(self, params: Dict) -> Dict:
        """
        构建 co-purchasing 边（同构模式, MAGB兼容）

        算法：
        1. 遍历每个产品的 also_buy + also_view 列表
        2. 只保留目标产品集内的边
        3. 用 tuple(sorted([asin_a, asin_b])) 去重（无向）
        4. 存两条有向边 (i→j, j→i) 表示无向
        """
        if not self._entities.get("product"):
            self.action_collect_products(params)

        products = self._get_products()
        product_asins = set(products.keys())

        edge_set = set()
        count = 0

        for asin, meta in products.items():
            # 合并 also_buy 和 also_view
            related_all = list(set(
                meta.get("also_buy", []) + meta.get("also_view", [])
            ))

            for related in related_all:
                if related == asin:
                    continue  # 无自环
                if related not in product_asins:
                    continue

                edge_key = tuple(sorted([asin, related]))
                if edge_key not in edge_set:
                    edge_set.add(edge_key)
                    # 无向边：两条有向边
                    self._edges["co_purchase"].append({
                        "source_id": f"product_{asin}",
                        "source_type": "product",
                        "target_id": f"product_{related}",
                        "target_type": "product",
                        "edge_type": "co_purchase",
                        "attributes": "{}"
                    })
                    self._edges["co_purchase"].append({
                        "source_id": f"product_{related}",
                        "source_type": "product",
                        "target_id": f"product_{asin}",
                        "target_type": "product",
                        "edge_type": "co_purchase",
                        "attributes": "{}"
                    })
                    count += 2

        logger.info(
            f"[ProductGraphBuilder] co_purchase 边: {count} "
            f"({count // 2} 无向边)"
        )
        return {
            "status": "success",
            "message": f"构建 {count} 条 co_purchase 边 ({count // 2} 无向)",
            "edge_count": count,
        }

    # ==================== 共用行动 ====================

    def action_save_edges(self, params: Dict) -> Dict:
        """保存边到 CSV"""
        task = self._current_task
        output_dir = task.get("output_dirs", {}).get("graphs", "./data/amazon_graphs")
        os.makedirs(output_dir, exist_ok=True)

        saved_files = {}
        total_edges = 0

        for edge_type, edges in self._edges.items():
            if not edges:
                continue

            file_path = os.path.join(output_dir, f"{edge_type}_edges.csv")
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["source_id", "source_type", "target_id",
                                "target_type", "edge_type", "attributes"]
                )
                writer.writeheader()
                writer.writerows(edges)

            saved_files[edge_type] = file_path
            total_edges += len(edges)

        # 写入 blackboard
        if self.blackboard:
            self.blackboard.write("product_graph", "edge_files", saved_files)
            self.blackboard.write("product_graph", "total_edges", total_edges)
            self.blackboard.write("product_graph", "edge_counts",
                                  {k: len(v) for k, v in self._edges.items()})
            self.blackboard.write("product_graph", "entity_counts",
                                  {k: len(v) for k, v in self._entities.items()})
            self.blackboard.write("product_graph", "graph_mode", self.graph_mode)
            if self.graph_mode == "homogeneous":
                self.blackboard.write("product_graph", "label_map", self._label_map)

        logger.info(
            f"[ProductGraphBuilder] 保存 {total_edges} 条边到 {len(saved_files)} 个文件"
        )

        return {
            "status": "success",
            "message": f"保存 {total_edges} 条边",
            "edge_files": saved_files,
            "total_edges": total_edges,
            "edge_counts": {k: len(v) for k, v in self._edges.items()},
        }

    def action_validate_graph(self, params: Dict) -> Dict:
        """验证图"""
        entity_counts = {k: len(v) for k, v in self._entities.items()}
        edge_counts = {k: len(v) for k, v in self._edges.items()}

        stats = {
            "graph_mode": self.graph_mode,
            "entity_counts": entity_counts,
            "edge_counts": edge_counts,
            "total_entities": sum(entity_counts.values()),
            "total_edges": sum(edge_counts.values()),
        }

        if self.graph_mode == "homogeneous":
            stats["label_distribution"] = {
                label: sum(1 for l in self._label_map.values() if l == label)
                for label in set(self._label_map.values())
            }

        return {
            "status": "success",
            "message": f"图统计: {stats}",
            "stats": stats,
        }

    def execute(self, task: Dict, upstream_data=None) -> Dict:
        """执行入口"""
        self._current_task = task
        self._accumulated_results = {}
        self._edges = defaultdict(list)
        self._entities = defaultdict(set)
        self._label_map = {}

        # 从 task dict 读取 graph_mode
        self.graph_mode = task.get("graph_mode", "heterogeneous")
        logger.info(
            f"[ProductGraphBuilder] 启动, graph_mode={self.graph_mode}"
        )

        return self.run_react_loop(
            task=task,
            blackboard=self.blackboard or Blackboard(),
            message_bus=self.message_bus or MessageBus()
        )
