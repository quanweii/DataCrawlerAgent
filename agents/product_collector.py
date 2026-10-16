"""
ProductCollectorAgent — Amazon 产品元数据加载与图片下载 Agent
从 McAuley 公开数据集加载产品元数据，下载产品图片
"""
import os
import json
from typing import Dict, Any, List
from loguru import logger

from .react_agent import ReActAgent
from .messages import Blackboard, MessageBus, Message


class ProductCollectorAgent(ReActAgent):
    """
    Amazon 产品采集 Agent

    可用行动:
    - load_metadata: 加载产品元数据
    - filter_by_category: 按品类过滤
    - download_images: 下载产品图片
    - extract_co_purchase: 提取共买关系
    - extract_categories: 提取类目树
    - extract_brands: 提取品牌信息
    - validate_products: 验证数据完整性
    - save_results: 保存结果
    - done: 完成
    """

    max_iterations = 10

    def __init__(self, client, system_prompt, mcp_tool_manager=None,
                 agent_name="ProductCollectorAgent", blackboard=None,
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
            "load_metadata",
            "filter_by_category",
            "download_images",
            "extract_co_purchase",
            "extract_categories",
            "extract_brands",
            "validate_products",
            "save_results",
            "done"
        ]

    def observe(self, task: Dict, blackboard: Blackboard,
                messages: List[Message]) -> Dict[str, Any]:
        observation = super().observe(task, blackboard, messages)

        products = self._accumulated_results.get("products", {})
        images_downloaded = self._accumulated_results.get("images_downloaded", 0)

        observation.update({
            "product_count": len(products),
            "images_downloaded": images_downloaded,
            "progress_summary": (
                f"{len(products)} 产品, {images_downloaded} 图片已下载"
            ),
        })

        return observation

    def is_done(self, observation: Dict) -> bool:
        product_count = observation.get("product_count", 0)
        # 只有当 save_results 已执行时才完成
        if product_count > 0 and self.blackboard:
            saved = self.blackboard.read("product_collector", "product_count", 0)
            if saved > 0:
                return True
        if len(self._action_history) >= 8:
            return True
        return False

    def _default_action(self, observation: Dict) -> Dict[str, Any]:
        """按阶段推进的默认行动"""
        results = self._accumulated_results
        products = results.get("products", {})

        if not products:
            return {"action": "load_metadata", "params": {},
                    "reasoning": "默认: 先加载元数据"}
        elif not results.get("images_downloaded") and not results.get("image_stats"):
            return {"action": "download_images", "params": {},
                    "reasoning": "默认: 下载图片"}
        elif not results.get("also_bought_edges"):
            return {"action": "extract_co_purchase", "params": {},
                    "reasoning": "默认: 提取共买关系"}
        elif not results.get("categories"):
            return {"action": "extract_categories", "params": {},
                    "reasoning": "默认: 提取类目"}
        elif not results.get("brands"):
            return {"action": "extract_brands", "params": {},
                    "reasoning": "默认: 提取品牌"}
        else:
            return {"action": "save_results", "params": {},
                    "reasoning": "默认: 保存结果"}

    def act(self, action_plan: Dict) -> Dict[str, Any]:
        action = action_plan.get("action", "")
        params = action_plan.get("params", {})

        action_map = {
            "load_metadata": self.action_load_metadata,
            "filter_by_category": self.action_filter_by_category,
            "download_images": self.action_download_images,
            "extract_co_purchase": self.action_extract_co_purchase,
            "extract_categories": self.action_extract_categories,
            "extract_brands": self.action_extract_brands,
            "validate_products": self.action_validate_products,
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

    def action_load_metadata(self, params: Dict) -> Dict:
        """加载产品元数据"""
        task = self._current_task
        categories = task.get("categories", [])
        products_per_category = task.get("products_per_category", 5000)
        file_path = task.get("metadata_path", None)

        all_products = {}
        for category in categories:
            # 确定文件路径
            cat_file = file_path
            if not cat_file:
                # 尝试自动查找
                possible = [
                    os.path.join(self.amazon_tools.data_dir, f"meta_{category}.json"),
                    os.path.join(self.amazon_tools.data_dir, f"meta_{category}.json.gz"),
                ]
                for p in possible:
                    if os.path.exists(p):
                        cat_file = p
                        break

            metadata = self.amazon_tools.load_metadata(
                category=category,
                file_path=cat_file,
                limit=products_per_category
            )

            all_products.update(metadata)

        self._accumulate_result({"products": all_products})

        logger.info(
            f"[ProductCollector] 加载 {len(all_products)} 个产品 "
            f"(来自 {len(categories)} 品类)"
        )

        return {
            "status": "success",
            "message": f"加载 {len(all_products)} 个产品",
            "product_count": len(all_products),
        }

    def action_filter_by_category(self, params: Dict) -> Dict:
        """过滤产品（需要图片/描述等）"""
        products = self._accumulated_results.get("products", {})
        task = self._current_task

        filtered = self.amazon_tools.filter_products(
            metadata=products,
            has_image=task.get("require_images", True),
            has_description=task.get("require_description", False),
            price_range=task.get("price_range", None),
            limit=task.get("max_products", None),
        )

        # 更新产品列表
        self._accumulated_results["products"] = filtered

        return {
            "status": "success",
            "message": f"过滤后 {len(filtered)} 产品",
            "product_count": len(filtered),
        }

    def action_download_images(self, params: Dict) -> Dict:
        """下载产品图片"""
        task = self._current_task
        products = self._accumulated_results.get("products", {})

        image_dir = task.get("output_dirs", {}).get("images", "./data/amazon_raw/images")
        max_images = task.get("max_images_per_product", 3)

        image_stats = self.amazon_tools.download_product_images(
            products,
            output_dir=image_dir,
            max_images_per_product=max_images,
            max_workers=5
        )

        # 更新产品的 image_count
        for asin, detail in image_stats.get("details", {}).items():
            if asin in products:
                products[asin]["actual_image_count"] = detail.get("count", 0)
                products[asin]["image_dir"] = detail.get("dir", "")

        self._accumulate_result({
            "images_downloaded": image_stats.get("downloaded", 0),
            "total_images": image_stats.get("total_images", 0),
        })

        return {
            "status": "success",
            "message": (
                f"图片下载: {image_stats.get('downloaded', 0)} 成功, "
                f"{image_stats.get('total_images', 0)} 张"
            ),
            **image_stats,
        }

    def action_extract_co_purchase(self, params: Dict) -> Dict:
        """提取共买/共看关系"""
        products = self._accumulated_results.get("products", {})

        co_buy = self.amazon_tools.extract_co_purchase_graph(products)
        co_view = self.amazon_tools.extract_co_view_graph(products)

        self._accumulate_result({
            "also_bought_edges": co_buy,
            "also_viewed_edges": co_view,
        })

        return {
            "status": "success",
            "message": f"共买 {len(co_buy)}, 共看 {len(co_view)}",
            "also_bought_count": len(co_buy),
            "also_viewed_count": len(co_view),
        }

    def action_extract_categories(self, params: Dict) -> Dict:
        """提取类目树"""
        products = self._accumulated_results.get("products", {})
        categories = self.amazon_tools.extract_category_tree(products)

        self._accumulate_result({"categories": categories})

        return {
            "status": "success",
            "message": f"提取 {len(categories)} 个类目节点",
            "category_count": len(categories),
        }

    def action_extract_brands(self, params: Dict) -> Dict:
        """提取品牌信息"""
        products = self._accumulated_results.get("products", {})
        brands = self.amazon_tools.extract_brands(products)

        self._accumulate_result({"brands": brands})

        return {
            "status": "success",
            "message": f"提取 {len(brands)} 个品牌",
            "brand_count": len(brands),
        }

    def action_validate_products(self, params: Dict) -> Dict:
        """验证产品数据"""
        products = self._accumulated_results.get("products", {})

        stats = {
            "total": len(products),
            "has_title": 0,
            "has_description": 0,
            "has_image": 0,
            "has_price": 0,
            "has_brand": 0,
            "has_also_buy": 0,
        }

        for asin, meta in products.items():
            if meta.get("title"):
                stats["has_title"] += 1
            if meta.get("description"):
                stats["has_description"] += 1
            if meta.get("image_url"):
                stats["has_image"] += 1
            if meta.get("price", 0) > 0:
                stats["has_price"] += 1
            if meta.get("brand"):
                stats["has_brand"] += 1
            if meta.get("also_buy"):
                stats["has_also_buy"] += 1

        return {
            "status": "success",
            "message": f"验证统计: {stats}",
            "stats": stats,
        }

    def action_save_results(self, params: Dict) -> Dict:
        """保存采集结果"""
        task = self._current_task
        products = self._accumulated_results.get("products", {})
        output_dir = task.get("output_dirs", {}).get("raw", "./data/amazon_raw")
        os.makedirs(output_dir, exist_ok=True)

        products_path = os.path.join(output_dir, "products.json")
        self.amazon_tools.save_products_json(products, products_path)

        # 写入 blackboard
        if self.blackboard:
            self.blackboard.write("product_collector", "products", products)
            self.blackboard.write("product_collector", "product_count", len(products))
            self.blackboard.write(
                "product_collector", "also_bought_edges",
                self._accumulated_results.get("also_bought_edges", [])
            )
            self.blackboard.write(
                "product_collector", "also_viewed_edges",
                self._accumulated_results.get("also_viewed_edges", [])
            )
            self.blackboard.write(
                "product_collector", "categories",
                self._accumulated_results.get("categories", {})
            )
            self.blackboard.write(
                "product_collector", "brands",
                self._accumulated_results.get("brands", {})
            )

        return {
            "status": "success",
            "message": f"保存 {len(products)} 个产品",
            "products_path": products_path,
        }

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
                logger.error(f"[ProductCollector] 初始化 Amazon 工具失败: {e}")
                return {"status": "error", "message": f"Amazon 工具初始化失败: {e}"}

        return self.run_react_loop(
            task=task,
            blackboard=self.blackboard or Blackboard(),
            message_bus=self.message_bus or MessageBus()
        )
