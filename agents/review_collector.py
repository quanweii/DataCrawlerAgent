"""
ReviewCollectorAgent — Amazon 评论采集 Agent
加载、过滤、采样产品评论，构建用户-产品交互数据
"""
import os
import json
from typing import Dict, Any, List, Set
from collections import defaultdict
from loguru import logger

from .react_agent import ReActAgent
from .messages import Blackboard, MessageBus, Message


class ReviewCollectorAgent(ReActAgent):
    """
    Amazon 评论采集 Agent

    可用行动:
    - load_reviews: 加载评论数据
    - filter_reviews: 过滤低质量评论
    - limit_per_product: 限制每产品评论数
    - extract_reviewer_ids: 提取唯一用户 ID
    - validate_reviews: 验证评论覆盖度
    - save_results: 保存结果
    - done: 完成
    """

    max_iterations = 8

    def __init__(self, client, system_prompt, mcp_tool_manager=None,
                 agent_name="ReviewCollectorAgent", blackboard=None,
                 message_bus=None, amazon_tools=None):
        super().__init__(
            client=client, system_prompt=system_prompt,
            mcp_tool_manager=mcp_tool_manager, agent_name=agent_name,
            blackboard=blackboard, message_bus=message_bus
        )
        self.amazon_tools = amazon_tools
        self._current_task = {}

    def get_available_actions(self) -> List[str]:
        return [
            "load_reviews",
            "filter_reviews",
            "limit_per_product",
            "extract_reviewer_ids",
            "validate_reviews",
            "save_results",
            "done"
        ]

    def observe(self, task: Dict, blackboard: Blackboard,
                messages: List[Message]) -> Dict[str, Any]:
        observation = super().observe(task, blackboard, messages)

        reviews = self._accumulated_results.get("reviews", {})
        total_reviews = sum(len(r) for r in reviews.values())
        reviewer_count = len(self._accumulated_results.get("reviewer_ids", set()))

        observation.update({
            "product_with_reviews": len(reviews),
            "total_reviews": total_reviews,
            "reviewer_count": reviewer_count,
            "progress_summary": (
                f"{total_reviews} 评论, {reviewer_count} 用户, "
                f"{len(reviews)} 产品"
            ),
        })

        return observation

    def is_done(self, observation: Dict) -> bool:
        total_reviews = observation.get("total_reviews", 0)
        # 只有当 save_results 已执行时才完成
        if total_reviews > 0 and self.blackboard:
            saved_reviews = self.blackboard.read("review_collector", "total_reviews", 0)
            if saved_reviews > 0:
                return True
        if len(self._action_history) >= 8:
            return True
        return False

    def _default_action(self, observation: Dict) -> Dict[str, Any]:
        """按阶段推进的默认行动"""
        results = self._accumulated_results
        reviews = results.get("reviews", {})

        if not reviews:
            return {"action": "load_reviews", "params": {},
                    "reasoning": "默认: 先加载评论"}
        elif not results.get("reviewer_ids"):
            return {"action": "extract_reviewer_ids", "params": {},
                    "reasoning": "默认: 提取用户ID"}
        else:
            return {"action": "save_results", "params": {},
                    "reasoning": "默认: 保存结果"}

    def act(self, action_plan: Dict) -> Dict[str, Any]:
        action = action_plan.get("action", "")
        params = action_plan.get("params", {})

        action_map = {
            "load_reviews": self.action_load_reviews,
            "filter_reviews": self.action_filter_reviews,
            "limit_per_product": self.action_limit_per_product,
            "extract_reviewer_ids": self.action_extract_reviewer_ids,
            "validate_reviews": self.action_validate_reviews,
            "save_results": self.action_save_results,
        }

        handler = action_map.get(action)
        if handler:
            return handler(params)
        elif action == "done":
            return {"status": "success", "message": "任务完成"}
        else:
            return {"status": "error", "message": f"未知行动: {action}"}

    # ==================== 行动实现 ====================

    def action_load_reviews(self, params: Dict) -> Dict:
        """加载评论数据"""
        task = self._current_task
        categories = task.get("categories", [])
        reviews_per_product = task.get("reviews_per_product", 10)

        # 获取目标产品 ASIN 集合
        target_asins = self._get_target_asins()

        all_reviews = {}
        for category in categories:
            # 先尝试带 target_asins 过滤加载
            reviews = self.amazon_tools.load_reviews(
                category=category,
                target_asins=target_asins,
                reviews_per_product=reviews_per_product
            )

            # 如果过滤后为空，不使用过滤重新加载（元数据和5-core评论可能不匹配）
            if not reviews and target_asins:
                logger.warning(
                    f"[ReviewCollector] {category} 评论与产品元数据无交集, "
                    f"尝试无过滤加载"
                )
                reviews = self.amazon_tools.load_reviews(
                    category=category,
                    target_asins=None,
                    reviews_per_product=reviews_per_product
                )

            all_reviews.update(reviews)

        self._accumulate_result({"reviews": all_reviews})

        total = sum(len(r) for r in all_reviews.values())
        logger.info(
            f"[ReviewCollector] 加载 {total} 评论, "
            f"{len(all_reviews)} 产品"
        )

        return {
            "status": "success",
            "message": f"加载 {total} 评论",
            "total_reviews": total,
            "product_count": len(all_reviews),
        }

    def action_filter_reviews(self, params: Dict) -> Dict:
        """过滤低质量评论"""
        reviews = self._accumulated_results.get("reviews", {})
        min_length = params.get("min_length", 10)

        filtered = {}
        removed = 0
        for asin, review_list in reviews.items():
            good_reviews = []
            for review in review_list:
                # 过滤条件
                if len(review.get("review_text", "")) < min_length:
                    removed += 1
                    continue
                good_reviews.append(review)
            if good_reviews:
                filtered[asin] = good_reviews

        self._accumulated_results["reviews"] = filtered
        total = sum(len(r) for r in filtered.values())

        return {
            "status": "success",
            "message": f"过滤: 移除 {removed} 条, 保留 {total} 条",
            "total_reviews": total,
            "removed": removed,
        }

    def action_limit_per_product(self, params: Dict) -> Dict:
        """限制每产品评论数"""
        reviews = self._accumulated_results.get("reviews", {})
        task = self._current_task
        max_per_product = task.get("reviews_per_product", 10)

        limited = {}
        total = 0
        for asin, review_list in reviews.items():
            if len(review_list) > max_per_product:
                limited[asin] = review_list[:max_per_product]
            else:
                limited[asin] = review_list
            total += len(limited[asin])

        self._accumulated_results["reviews"] = limited

        return {
            "status": "success",
            "message": f"限制后: {total} 评论",
            "total_reviews": total,
        }

    def action_extract_reviewer_ids(self, params: Dict) -> Dict:
        """提取唯一用户 ID"""
        reviews = self._accumulated_results.get("reviews", {})

        reviewer_ids = set()
        for review_list in reviews.values():
            for review in review_list:
                rid = review.get("reviewer_id", "")
                if rid:
                    reviewer_ids.add(rid)

        self._accumulate_result({"reviewer_ids": reviewer_ids})

        return {
            "status": "success",
            "message": f"发现 {len(reviewer_ids)} 个唯一用户",
            "reviewer_count": len(reviewer_ids),
        }

    def action_validate_reviews(self, params: Dict) -> Dict:
        """验证评论数据"""
        reviews = self._accumulated_results.get("reviews", {})

        stats = {
            "total_reviews": sum(len(r) for r in reviews.values()),
            "products_with_reviews": len(reviews),
            "avg_reviews_per_product": 0,
            "verified_purchase_count": 0,
            "avg_rating": 0,
        }

        total_reviews = stats["total_reviews"]
        total_rating = 0
        for review_list in reviews.values():
            for review in review_list:
                total_rating += review.get("rating", 0)
                if review.get("verified", False):
                    stats["verified_purchase_count"] += 1

        if total_reviews > 0:
            stats["avg_reviews_per_product"] = total_reviews / max(len(reviews), 1)
            stats["avg_rating"] = total_rating / total_reviews

        return {
            "status": "success",
            "message": f"评论统计: {stats}",
            "stats": stats,
        }

    def action_save_results(self, params: Dict) -> Dict:
        """保存评论数据"""
        task = self._current_task
        reviews = self._accumulated_results.get("reviews", {})
        output_dir = task.get("output_dirs", {}).get("raw", "./data/amazon_raw")
        os.makedirs(output_dir, exist_ok=True)

        reviews_path = os.path.join(output_dir, "reviews.json")
        self.amazon_tools.save_reviews_json(reviews, reviews_path)

        # 写入 blackboard
        if self.blackboard:
            self.blackboard.write("review_collector", "reviews", reviews)
            self.blackboard.write(
                "review_collector", "total_reviews",
                sum(len(r) for r in reviews.values())
            )
            self.blackboard.write(
                "review_collector", "reviewer_ids",
                list(self._accumulated_results.get("reviewer_ids", set()))
            )

        return {
            "status": "success",
            "message": f"保存 {sum(len(r) for r in reviews.values())} 评论",
            "reviews_path": reviews_path,
        }

    # ==================== 辅助方法 ====================

    def _get_target_asins(self) -> Set[str]:
        """获取目标产品 ASIN 集合"""
        if self.blackboard:
            products = self.blackboard.read("product_collector", "products", {})
            if products:
                return set(products.keys())
        return set(self._current_task.get("target_asins", []))

    def execute(self, task: Dict, upstream_data=None) -> Dict:
        """执行入口"""
        self._current_task = task
        self._accumulated_results = {}

        if self.amazon_tools is None:
            data_dir = task.get("output_dirs", {}).get("raw", "./data/amazon_raw")
            try:
                from tools.amazon_tools import AmazonToolManager
                self.amazon_tools = AmazonToolManager(data_dir=data_dir)
            except Exception as e:
                logger.error(f"[ReviewCollector] 初始化 Amazon 工具失败: {e}")
                return {"status": "error", "message": f"Amazon 工具初始化失败: {e}"}

        return self.run_react_loop(
            task=task,
            blackboard=self.blackboard or Blackboard(),
            message_bus=self.message_bus or MessageBus()
        )
