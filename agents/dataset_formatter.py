"""
数据集封装Agent - ReAct版
将数据封装成多模态属性图数据集，支持LLM驱动的迭代决策
"""
from typing import Dict, Any, Optional, List
from collections import defaultdict
from openai import OpenAI
from loguru import logger
import os
import json
from datetime import datetime

from .react_agent import ReActAgent
from .messages import Blackboard, MessageBus, Message


class DatasetFormatterAgent(ReActAgent):
    """数据集封装Agent — ReAct循环驱动"""

    def __init__(self, client: OpenAI, system_prompt: str,
                 mcp_tool_manager=None, blackboard=None, message_bus=None):
        super().__init__(
            client=client,
            system_prompt=system_prompt,
            mcp_tool_manager=mcp_tool_manager,
            agent_name="DatasetFormatterAgent",
            blackboard=blackboard,
            message_bus=message_bus
        )
        self.formatted_dataset: Dict = {}

    # ==================== ReAct 接口实现 ====================

    def get_available_actions(self) -> List[str]:
        return [
            "collect_data",       # 从blackboard/上游收集数据
            "build_nodes",        # 构建节点
            "build_edges",        # 构建边
            "build_metadata",     # 构建元数据
            "save_dataset",       # 保存数据集
            "validate_dataset",   # 验证数据集
            "request_graph_rebuild",  # 请求重建引用图
            "done"
        ]

    def is_done(self, observation: Dict) -> bool:
        results = observation.get("accumulated_results", {})
        return bool(results.get("output_path"))

    def observe(self, task: Dict, blackboard: Blackboard,
                messages: List[Message]) -> Dict[str, Any]:
        observation = super().observe(task, blackboard, messages)

        if blackboard:
            observation["collector_papers"] = blackboard.read("collector", "papers", [])
            observation["graph_edges_file"] = blackboard.read("graph", "edges_file", "")
            observation["graph_edge_count"] = blackboard.read("graph", "edge_count", 0)
            observation["cleaner_stats"] = blackboard.read("cleaner", "stats", {})

        results = observation.get("accumulated_results", {})
        observation["node_count"] = len(results.get("nodes", []))
        observation["edge_count"] = len(results.get("edges", []))
        observation["has_metadata"] = bool(results.get("metadata"))
        observation["dataset_saved"] = bool(results.get("output_path"))

        # 生成 phase_hint 引导 LLM 按正确顺序执行
        if observation["dataset_saved"]:
            observation["phase_hint"] = "数据集已保存完成，任务结束，请选择 done。"
        elif observation["node_count"] > 0 and observation["edge_count"] > 0 and observation["has_metadata"]:
            observation["phase_hint"] = "节点、边、元数据已全部就绪！下一步必须选择 save_dataset 保存数据集，不要再重复构建。"
        elif observation["node_count"] > 0 and observation["edge_count"] > 0:
            observation["phase_hint"] = "节点和边已构建完成，下一步选择 build_metadata 构建元数据，然后 save_dataset 保存。"
        elif observation["node_count"] > 0:
            observation["phase_hint"] = "节点已构建完成，下一步选择 build_edges 构建边。不要重复 build_nodes。"

        if observation["node_count"] > 0 and not observation["dataset_saved"]:
            if not observation["has_metadata"]:
                observation["suggested_next"] = (
                    "节点已构建，请先调用build_metadata构建元数据，再调用save_dataset保存"
                )
            else:
                observation["suggested_next"] = (
                    "节点和元数据已就绪，请调用save_dataset保存数据集"
                )

        return observation

    def act(self, action_plan: Dict) -> Dict[str, Any]:
        action = action_plan.get("action", "")

        if action == "collect_data":
            return self.action_collect_data(self._current_task, self._upstream_data)
        elif action == "build_nodes":
            return self.action_build_nodes()
        elif action == "build_edges":
            return self.action_build_edges()
        elif action == "build_metadata":
            return self.action_build_metadata(self._current_task)
        elif action == "save_dataset":
            return self.action_save_dataset(self._current_task)
        elif action == "validate_dataset":
            return self.action_validate_dataset()
        elif action == "request_graph_rebuild":
            return self.action_request_graph_rebuild()
        elif action == "done":
            return {"status": "success", "message": "数据集封装完成"}

        return {"status": "error", "message": f"未知行动: {action}"}

    def _default_action(self, observation: Dict) -> Dict[str, Any]:
        """LLM决策失败时的默认行动 — 按流水线顺序推进"""
        results = observation.get("accumulated_results", {})
        nodes = results.get("nodes", [])
        edges = results.get("edges", [])
        metadata = results.get("metadata")
        output_path = results.get("output_path")

        if output_path:
            return {"action": "done", "params": {}, "reasoning": "数据集已保存，完成"}
        if nodes and edges and metadata:
            return {"action": "save_dataset", "params": {}, "reasoning": "节点/边/元数据已就绪，保存数据集"}
        if nodes and edges:
            return {"action": "build_metadata", "params": {}, "reasoning": "节点和边已就绪，构建元数据"}
        if nodes:
            return {"action": "build_edges", "params": {}, "reasoning": "节点已就绪，构建边"}
        return {"action": "build_nodes", "params": {}, "reasoning": "开始构建节点"}

    # ==================== 行动方法 ====================

    def action_collect_data(self, task: Dict, upstream_data=None) -> Dict:
        """从blackboard或上游数据收集数据 — 支持多领域"""
        domain = task.get("domain", "arxiv")

        if domain == "reddit":
            return self._collect_data_reddit(task)
        elif domain == "amazon":
            return self._collect_data_amazon(task)
        elif domain in self._BUNDLE_DOMAIN_CONFIG or (
                task.get("graph_mode") == "homogeneous"
                and domain not in ("arxiv", "reddit", "amazon")):
            return self._collect_data_bundle(domain, task)
        else:
            if task.get("arxiv_unify"):
                return self._collect_data_arxiv_unified(task)
            return self._collect_data_arxiv(task, upstream_data)

    def _collect_data_arxiv(self, task: Dict, upstream_data=None) -> Dict:
        """ArXiv 数据收集（原有逻辑）"""
        cleaned_papers = []
        cleaned_multimodal = []
        edges_file = task.get("edges_file", "")

        if upstream_data:
            cleaned_papers = upstream_data.get("cleaned_papers", [])
            cleaned_multimodal = upstream_data.get("cleaned_multimodal",
                                      upstream_data.get("multimodal_data", []))
            if not edges_file:
                edges_file = upstream_data.get("edges_file", "")

        # 从blackboard补充
        if self.blackboard and not cleaned_papers:
            cleaned_papers = self.blackboard.read("cleaner", "cleaned_papers", [])
            collector_papers = self.blackboard.read("collector", "papers", [])
            if collector_papers and not cleaned_papers:
                cleaned_papers = collector_papers
            cleaned_multimodal = self.blackboard.read("collector", "multimodal_data",
                                       cleaned_multimodal)
            if not edges_file:
                edges_file = self.blackboard.read("graph", "edges_file", "")

        # 将GPU提取结果转换为_build_nodes兼容的格式
        normalized_multimodal = []
        for item in cleaned_multimodal:
            if "modality" in item:
                normalized_multimodal.append(item)
            elif "output_folder" in item:
                normalized_multimodal.append({
                    "modality": "text",
                    "source": item.get("pdf_path", ""),
                    "content": item.get("markdown_content", ""),
                    "path": item.get("md_path", ""),
                    "output_folder": item.get("output_folder", ""),
                    "image_count": item.get("image_count", 0),
                    "paper_name": item.get("paper_name", ""),
                })
                if item.get("image_count", 0) > 0:
                    normalized_multimodal.append({
                        "modality": "image",
                        "source": item.get("pdf_path", ""),
                        "content": "",
                        "path": item.get("images_folder", ""),
                        "paper_name": item.get("paper_name", ""),
                        "image_count": item.get("image_count", 0),
                    })

        self._formatter_data = {
            "domain": "arxiv",
            "cleaned_papers": cleaned_papers,
            "cleaned_multimodal": normalized_multimodal,
            "edges_file": edges_file
        }

        return {
            "status": "success",
            "papers_count": len(cleaned_papers),
            "multimodal_count": len(normalized_multimodal),
            "has_edges_file": bool(edges_file)
        }

    def _collect_data_arxiv_unified(self, task: Dict) -> Dict:
        """arxiv 统一 MAG 模式 (A′): 由 crawlers.arxiv.build_arxiv_graph 把
        reference_graph_builder 已构造的引用网络(benchmark/processed/graph_data.pt)
        统一成 newdomain 协议的单一 MAG(4 子域, label=子域, relation="cites")。
        跳过逐篇重爬/重抽引用, 直接拿到完整 nodes/edges/stats, 后续 build/save 走与
        steam/tmdb/wikiart 一致的透传路径。引用边由 agent(reference_graph_builder)造,
        本步只做统一+封装, 不造数据。
        """
        from crawlers import build_arxiv_graph
        nodes, edges, stats = build_arxiv_graph(task.get("graph_data_pt"))
        self._formatter_data = {
            "domain": "arxiv", "nodes": nodes, "edges": edges,
            "stats": stats, "unified": True,
        }
        logger.info(f"[DatasetFormatter] arxiv 统一 MAG: nodes={len(nodes)} "
                    f"edges={len(edges)} homophily={stats.get('homophily')}")
        return {
            "status": "success", "nodes_count": len(nodes),
            "edges_count": len(edges), "homophily": stats.get("homophily"),
        }

    def _collect_data_reddit(self, task: Dict) -> Dict:
        """Reddit 数据收集"""
        posts = []
        comments_map = {}
        user_metadata = {}
        subreddits = []

        if self.blackboard:
            posts = self.blackboard.read("subreddit_collector", "posts", [])
            comments_map = self.blackboard.read("post_collector", "comments_map", {})
            user_metadata = self.blackboard.read("post_collector", "user_metadata", {})
            subreddits = self.blackboard.read("subreddit_collector", "subreddits", [])

        self._formatter_data = {
            "domain": "reddit",
            "posts": posts,
            "comments_map": comments_map,
            "user_metadata": user_metadata,
            "subreddits": subreddits,
        }

        total_comments = sum(len(c) for c in comments_map.values())
        return {
            "status": "success",
            "posts_count": len(posts),
            "comments_count": total_comments,
            "users_count": len(user_metadata),
            "subreddits_count": len(subreddits),
        }

    def _collect_data_amazon(self, task: Dict) -> Dict:
        """Amazon 数据收集"""
        products = {}
        reviews = {}
        categories = {}
        brands = {}

        if self.blackboard:
            products = self.blackboard.read("product_collector", "products", {})
            reviews = self.blackboard.read("review_collector", "reviews", {})
            categories = self.blackboard.read("product_collector", "categories", {})
            brands = self.blackboard.read("product_collector", "brands", {})

        self._formatter_data = {
            "domain": "amazon",
            "products": products,
            "reviews": reviews,
            "categories": categories,
            "brands": brands,
        }

        total_reviews = sum(len(r) for r in reviews.values())
        return {
            "status": "success",
            "products_count": len(products),
            "reviews_count": total_reviews,
            "categories_count": len(categories),
            "brands_count": len(brands),
        }

    # ==================== Steam/TMDb/WikiArt 领域 (bundle 透传) ====================
    # 三域 GraphBuilder 已产出完整 bundle (nodes+edges+stats), formatter 只透传 + 元数据

    _BUNDLE_DOMAIN_CONFIG = {
        "steam": {
            "graph_ns": "steam_graph",
            "node_type": "game", "id_field": "appid",
            "edge_relations": ["shared_genre", "same_developer", "same_publisher", "shared_category"],
            "node_attributes": ["appid", "name", "markdown_length", "image_count",
                                "markdown_size", "label", "label_index"],
        },
        "tmdb": {
            "graph_ns": "tmdb_graph",
            "node_type": "movie", "id_field": "movie_id",
            "edge_relations": ["shared_genre", "same_director", "shared_actor"],
            "node_attributes": ["movie_id", "title", "markdown_length", "image_count",
                                "markdown_size", "label", "label_index"],
        },
        "wikiart": {
            "graph_ns": "wikiart_graph",
            "node_type": "artwork", "id_field": "painting_id",
            "edge_relations": ["same_artist", "shared_style"],
            "node_attributes": ["painting_id", "title", "markdown_length", "image_count",
                                "markdown_size", "label", "label_index"],
        },
    }

    def _collect_data_bundle(self, domain: str, task: Dict) -> Dict:
        """从 {domain}_graph 命名空间读取 GraphBuilder 产出的完整 bundle"""
        cfg = self._BUNDLE_DOMAIN_CONFIG.get(domain, {"graph_ns": f"{domain}_graph"})
        nodes, edges, stats = [], [], {}
        if self.blackboard:
            nodes = self.blackboard.read(cfg["graph_ns"], "nodes", [])
            edges = self.blackboard.read(cfg["graph_ns"], "edges", [])
            stats = self.blackboard.read(cfg["graph_ns"], "stats", {})
        self._formatter_data = {
            "domain": domain, "nodes": nodes, "edges": edges, "stats": stats, "bundle": True,
        }
        return {
            "status": "success",
            "nodes_count": len(nodes),
            "edges_count": len(edges),
            "homophily": stats.get("homophily"),
        }

    def _build_nodes_bundle(self, data: Dict) -> List[Dict]:
        """透传 GraphBuilder 已构建的节点 (含 label/label_index)"""
        nodes = data.get("nodes", [])
        logger.info(f"[DatasetFormatter] {data.get('domain')} 同构节点: {len(nodes)}")
        return nodes

    def _build_edges_bundle(self, data: Dict) -> List[Dict]:
        """透传 GraphBuilder 已构建的边"""
        edges = data.get("edges", [])
        logger.info(f"[DatasetFormatter] {data.get('domain')} 同构边: {len(edges)}")
        return edges

    def _build_metadata_bundle(self, domain: str, dataset_name: str,
                               nodes: List[Dict], edges: List[Dict]) -> Dict:
        """v3 风格元数据 (homogeneous, text+image 双模态, label_field)"""
        cfg = self._BUNDLE_DOMAIN_CONFIG.get(domain)
        if cfg is None:
            cfg = {
                "node_type": nodes[0].get("type", "item") if nodes else "item",
                "node_attributes": sorted({k for n in nodes for k in n.get("attributes", {})}),
                "edge_relations": sorted({e.get("attributes", {}).get("relation", "?") for e in edges}),
            }
        stats = self._formatter_data.get("stats", {}) if hasattr(self, '_formatter_data') and self._formatter_data else {}
        n_nodes = len(nodes)
        total_img = sum(n.get("attributes", {}).get("image_count", 0) for n in nodes)
        nodes_with_img = sum(1 for n in nodes if n.get("attributes", {}).get("image_count", 0) > 0)
        total_md = sum(n.get("attributes", {}).get("markdown_length", 0) for n in nodes)
        return {
            "name": dataset_name,
            "domain": domain,
            "graph_mode": "homogeneous",
            "created_at": datetime.now().strftime('%Y-%m-%d'),
            "statistics": {
                "node_count": n_nodes,
                "edge_count": len(edges),
                "total_images": total_img,
                "nodes_with_images": nodes_with_img,
                "total_markdown_chars": total_md,
                "avg_md_length": round(total_md / max(n_nodes, 1)),
                "avg_degree": stats.get("avg_degree", round(2 * len(edges) / max(n_nodes, 1), 1)),
                "homophily": stats.get("homophily", 0),
                "num_labels": stats.get("num_labels", 0),
                "label_distribution": stats.get("label_distribution", {}),
            },
            "schema": {
                "node_types": [cfg["node_type"]],
                "edge_types": ["co_attribute"],
                "node_attributes": cfg["node_attributes"],
                "edge_relations": cfg["edge_relations"],
                "modalities": ["text", "image"],
                "label_field": "label",
            },
        }

    def action_build_nodes(self) -> Dict:
        existing = self._accumulated_results.get("nodes", [])
        if existing:
            return {"status": "success", "nodes": existing,
                    "message": f"节点已构建({len(existing)}个)，跳过重复构建"}

        data = self._ensure_collected()
        domain = data.get("domain", "arxiv")

        if data.get("unified"):
            nodes = data.get("nodes", [])
        elif domain == "reddit":
            nodes = self._build_nodes_reddit(data)
        elif domain == "amazon":
            nodes = self._build_nodes_amazon(data)
        elif data.get("bundle") or domain in self._BUNDLE_DOMAIN_CONFIG:
            nodes = self._build_nodes_bundle(data)
        else:
            nodes = self._build_nodes(
                data.get("cleaned_papers", []),
                data.get("cleaned_multimodal", [])
            )

        if self.blackboard:
            self.blackboard.write("formatter", "nodes", nodes)
            self.blackboard.write("formatter", "node_count", len(nodes))

        return {"status": "success", "nodes": nodes}

    def action_build_edges(self) -> Dict:
        existing = self._accumulated_results.get("edges", [])
        if existing:
            return {"status": "success", "edges": existing,
                    "message": f"边已构建({len(existing)}条)，跳过重复构建"}

        data = self._ensure_collected()
        domain = data.get("domain", "arxiv")

        if data.get("unified"):
            edges = data.get("edges", [])
        elif domain == "reddit":
            edges = self._build_edges_reddit(data)
        elif domain == "amazon":
            edges = self._build_edges_amazon(data)
        elif data.get("bundle") or domain in self._BUNDLE_DOMAIN_CONFIG:
            edges = self._build_edges_bundle(data)
        else:
            edges = self._build_edges(
                self._accumulated_results.get("nodes", []),
                data.get("cleaned_papers", []),
                data.get("cleaned_multimodal", [])
            )

        if self.blackboard:
            self.blackboard.write("formatter", "edges", edges)
            self.blackboard.write("formatter", "edge_count", len(edges))

        return {"status": "success", "edges": edges}

    def action_build_metadata(self, task: Dict) -> Dict:
        data = self._ensure_collected()
        papers = data.get("cleaned_papers", [])
        multimodal = data.get("cleaned_multimodal", [])
        dataset_name = task.get("dataset_name",
                                f"multimodal_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        nodes = self._accumulated_results.get("nodes", [])
        edges = self._accumulated_results.get("edges", [])

        metadata = self._build_metadata(dataset_name, nodes, edges)
        return {"status": "success", "metadata": metadata}

    def action_save_dataset(self, task: Dict) -> Dict:
        nodes = self._accumulated_results.get("nodes", [])
        edges = self._accumulated_results.get("edges", [])
        metadata = self._accumulated_results.get("metadata", {})

        dataset_name = task.get("dataset_name",
                                f"multimodal_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        output_format = task.get("output_format", "json")
        output_dir = task.get("output_dir", "./data/datasets")

        os.makedirs(output_dir, exist_ok=True)

        attribute_graph = self.call_mcp_tool(
            "format_attribute_graph", nodes=nodes, edges=edges,
            metadata=metadata
        )

        output_path = os.path.join(output_dir, f"{dataset_name}.{output_format}")
        save_result = self.call_mcp_tool(
            "save_dataset",
            dataset=attribute_graph.get("graph", {}) if attribute_graph else {"nodes": nodes, "edges": edges},
            output_path=output_path, format=output_format
        )

        if save_result and save_result.get("status") == "success":
            nodes = self._accumulated_results.get("nodes", [])
            edges = self._accumulated_results.get("edges", [])
            modality_dist = self._count_modalities(nodes)

            # 更新metadata中的edge_count
            if metadata and "statistics" in metadata:
                metadata["statistics"]["edge_count"] = len(edges)

            result = {
                "status": "success",
                "output_path": save_result.get("path", output_path),
                "stats": {
                    "node_count": len(nodes),
                    "edge_count": len(edges),
                    "modality_distribution": modality_dist
                }
            }

            if self.blackboard:
                self.blackboard.write("formatter", "dataset_path", result["output_path"])
                self.blackboard.write("formatter", "final_stats", result["stats"])

            self.formatted_dataset = result
            return result
        else:
            return {
                "status": "error",
                "message": save_result.get("message", "保存失败") if save_result else "保存失败"
            }

    def action_validate_dataset(self) -> Dict:
        output_path = self._accumulated_results.get("output_path", "")
        if not output_path or not os.path.exists(output_path):
            return {
                "status": "error",
                "message": "数据集尚未保存，请先调用save_dataset"
            }

        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            node_count = len(data.get("nodes", []))
            edge_count = len(data.get("edges", []))

            issues = []
            if node_count == 0:
                issues.append("数据集没有节点")

            # 检查引用边
            graph_edges = self.blackboard.read("graph", "edge_count", 0) if self.blackboard else 0
            if graph_edges > 0 and edge_count == 0:
                issues.append("引用图有边但数据集没有引用边 — 可能需要重建")

            if issues:
                return {"status": "warning", "issues": issues}

            return {"status": "success", "node_count": node_count, "edge_count": edge_count}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def action_request_graph_rebuild(self) -> Dict:
        """请求引用图重建"""
        if self.message_bus:
            self.message_bus.send(Message(
                sender=self.agent_name,
                recipient="ReferenceGraphBuilder",
                msg_type="reprocess_request",
                content={"reason": "数据集中缺少引用边"},
                priority=2
            ))
        return {"status": "success", "message": "已请求重建引用图"}

    # ==================== execute 入口（兼容旧接口）====================

    def execute(self, task: Dict[str, Any],
                upstream_data: Optional[Dict] = None) -> Dict[str, Any]:
        self._current_task = task
        self._upstream_data = upstream_data
        # A retry/rebuild must recollect the current upstream bundle.
        self._formatter_data = None

        if self.blackboard and self.message_bus:
            result = self.run_react_loop(task, self.blackboard, self.message_bus)
            self._ensure_compatible_result(result)
            return result

        return self._execute_legacy(task, upstream_data)

    def _ensure_compatible_result(self, result: Dict):
        """确保ReAct结果与旧格式兼容"""
        if "output_path" not in result:
            result["output_path"] = self._accumulated_results.get("output_path", "")
        if "stats" not in result:
            result["stats"] = self._accumulated_results.get("stats", {
                "node_count": 0, "edge_count": 0, "modality_distribution": {}
            })

        evaluation = self.evaluate_own_results(result)
        description = self._generate_dataset_description(result)
        result["description"] = description

    def _execute_legacy(self, task, upstream_data=None):
        """原始执行逻辑（向后兼容）"""
        logger.info(f"[{self.agent_name}] 开始执行数据集封装任务")

        analysis = self.reason_about_task(task)
        logger.info(f"[{self.agent_name}] 任务理解: {analysis.get('understanding', '')}")

        if not upstream_data:
            return {"status": "error", "message": "没有接收到上游数据"}

        dataset_name = task.get("dataset_name", f"multimodal_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        output_format = task.get("output_format", "json")
        output_dir = task.get("output_dir", "./data/datasets")
        include_metadata = task.get("include_metadata", True)

        results = {
            "status": "success",
            "dataset_name": dataset_name,
            "nodes": [],
            "edges": [],
            "metadata": {},
            "output_path": "",
            "stats": {"node_count": 0, "edge_count": 0, "modality_distribution": {}}
        }

        os.makedirs(output_dir, exist_ok=True)

        cleaned_papers = upstream_data.get("cleaned_papers", [])
        cleaned_multimodal = upstream_data.get("cleaned_multimodal", [])

        # 存储edges_file和task供_build_edges使用
        self._formatter_data = {"edges_file": task.get("edges_file", "")}
        self._current_task = task

        nodes = self._build_nodes(cleaned_papers, cleaned_multimodal)
        results["nodes"] = nodes
        results["stats"]["node_count"] = len(nodes)

        edges = self._build_edges(nodes, cleaned_papers, cleaned_multimodal)
        results["edges"] = edges
        results["stats"]["edge_count"] = len(edges)

        modality_dist = self._count_modalities(nodes)
        results["stats"]["modality_distribution"] = modality_dist

        if include_metadata:
            results["metadata"] = self._build_metadata(
                dataset_name, nodes, edges)
            # 更新metadata中的edge_count
            results["metadata"]["statistics"]["edge_count"] = len(edges)

        attribute_graph = self.call_mcp_tool(
            "format_attribute_graph", nodes=nodes, edges=edges,
            metadata=results["metadata"])

        output_path = os.path.join(output_dir, f"{dataset_name}.{output_format}")
        save_result = self.call_mcp_tool(
            "save_dataset",
            dataset=attribute_graph.get("graph", {}) if attribute_graph else {"nodes": nodes, "edges": edges, "metadata": results["metadata"]},
            output_path=output_path, format=output_format)

        if save_result and save_result.get("status") == "success":
            results["output_path"] = save_result.get("path", output_path)
        else:
            results["status"] = "error"
            results["message"] = save_result.get("message", "保存失败") if save_result else "保存失败"

        self.formatted_dataset = results

        evaluation = self.evaluate_own_results(results)
        description = self._generate_dataset_description(results)
        results["description"] = description

        logger.info(f"[{self.agent_name}] 数据集封装完成: {results['stats']['node_count']}节点, {results['stats']['edge_count']}边")
        return results

    # ==================== 核心逻辑（完全不变）====================

    def _ensure_collected(self) -> Dict:
        """确保数据已收集，如果未收集则自动触发"""
        data = getattr(self, '_formatter_data', None)
        if data is None:
            logger.info(f"[{self.agent_name}] 自动触发数据收集")
            self.action_collect_data(self._current_task, self._upstream_data)
            data = getattr(self, '_formatter_data', {})
        return data

    def _build_nodes(self, papers: List[Dict], multimodal: List[Dict]) -> List[Dict]:
        """构建paper节点 — 一个论文一个节点，包含提取统计信息"""
        nodes = []

        # 从多模态数据中按output_folder分组统计
        folder_stats = {}
        for item in multimodal:
            output_folder = item.get("output_folder", "")
            if not output_folder:
                continue
            if output_folder not in folder_stats:
                folder_stats[output_folder] = {
                    "markdown_length": 0,
                    "image_count": 0,
                    "markdown_size": 0,
                }
            # 文本 — 统计markdown
            md_path = item.get("md_path", "") or item.get("path", "")
            if md_path and os.path.exists(md_path):
                folder_stats[output_folder]["md_path"] = md_path
                folder_stats[output_folder]["markdown_size"] = os.path.getsize(md_path)
                folder_stats[output_folder]["markdown_length"] = folder_stats[output_folder]["markdown_size"]
            else:
                md_content = item.get("content", "") or item.get("markdown_content", "")
                if md_content:
                    folder_stats[output_folder]["markdown_length"] = len(md_content)
                    folder_stats[output_folder]["markdown_size"] = len(md_content)
            # 图像
            img_count = item.get("image_count", 0)
            if img_count > 0:
                folder_stats[output_folder]["image_count"] = img_count

        for paper in papers:
            # 提取arxiv_id — 多种来源（collector和cleaner输出格式不同）
            arxiv_id = (paper.get("arxiv_id", "")
                        or paper.get("name", "")
                        or paper.get("paper_id", "")
                        or paper.get("source", {}).get("id", ""))
            if not arxiv_id:
                p = paper.get("path", "") or paper.get("paper_dir", "")
                arxiv_id = os.path.splitext(os.path.basename(p))[0]

            # 找到对应的提取文件夹
            md_length = 0
            md_size = 0
            img_count = 0

            # 方法1: 从paper的markdown_path找
            md_path = paper.get("markdown_path", "")
            if md_path and os.path.exists(md_path):
                md_size = os.path.getsize(md_path)
                md_length = md_size
                paper_dir = os.path.dirname(md_path)
                images_dir = os.path.join(paper_dir, "images")
                if os.path.isdir(images_dir):
                    img_count = len([f for f in os.listdir(images_dir)
                                     if f.endswith(('.png', '.jpg', '.jpeg'))])

            # 方法2: 从paper_dir直接找（cleaner输出用paper_dir）
            if md_size == 0:
                paper_dir = (paper.get("paper_dir", "")
                             or os.path.dirname(paper.get("path", "")))
                if paper_dir and os.path.isdir(paper_dir):
                    # 检查直接子文件
                    expected_md = os.path.join(paper_dir, f"{arxiv_id}.md")
                    if os.path.exists(expected_md):
                        md_size = os.path.getsize(expected_md)
                        md_length = md_size
                    images_dir = os.path.join(paper_dir, "images")
                    if os.path.isdir(images_dir):
                        img_count = len([f for f in os.listdir(images_dir)
                                         if f.endswith(('.png', '.jpg', '.jpeg'))])

                    # 检查子目录（GPU提取可能创建paper_0/等）
                    if md_size == 0:
                        for folder_key, stats in folder_stats.items():
                            # folder_key可能是 ./data/papers/paper_0
                            if folder_key.startswith(paper_dir):
                                md_size = stats.get("markdown_size", 0)
                                md_length = stats.get("markdown_length", 0)
                                img_count = stats.get("image_count", 0)
                                break

            # 方法3: 直接从folder_stats匹配（按paper_name匹配）
            if md_size == 0:
                paper_name = (paper.get("name", "")
                              or paper.get("arxiv_id", "")
                              or paper.get("paper_id", ""))
                for folder_key, stats in folder_stats.items():
                    folder_name = os.path.basename(folder_key.rstrip("/"))
                    if folder_name == paper_name:
                        md_size = stats.get("markdown_size", 0)
                        md_length = stats.get("markdown_length", 0)
                        img_count = stats.get("image_count", 0)
                        break

            node = {
                "id": arxiv_id,
                "type": "paper",
                "attributes": {
                    "arxiv_id": arxiv_id,
                    "markdown_length": md_length,
                    "image_count": img_count,
                    "markdown_size": md_size,
                }
            }
            nodes.append(node)

        # 去重：同一个arxiv_id可能出现多次（ReAct循环累积重复）
        seen = set()
        unique_nodes = []
        for node in nodes:
            if node["id"] not in seen:
                seen.add(node["id"])
                unique_nodes.append(node)

        return unique_nodes

    def _build_edges(self, nodes: List[Dict], papers: List[Dict], multimodal: List[Dict]) -> List[Dict]:
        """构建citation边 — 从引用图CSV文件读取"""
        edges = []
        edges_file = self._formatter_data.get("edges_file", "") if hasattr(self, '_formatter_data') and self._formatter_data else ""

        if edges_file and os.path.exists(edges_file):
            import csv
            node_ids = {n["id"] for n in nodes}
            try:
                with open(edges_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        source_id = row.get("source_id", "").strip()
                        target_id = row.get("target_id", "").strip()
                        ref_text = row.get("reference_text", "").strip()
                        # 保留 source 在节点集中的出边（target 可能是外部论文）
                        if source_id in node_ids:
                            edges.append({
                                "source": source_id,
                                "target": target_id,
                                "type": "citation",
                                "attributes": {"reference_text": ref_text}
                            })
                logger.info(f"[{self.agent_name}] 从CSV加载 {len(edges)} 条citation边")
            except Exception as e:
                logger.warning(f"[{self.agent_name}] 读取引用边CSV失败: {e}")
        else:
            logger.info(f"[{self.agent_name}] 无引用边文件，跳过边构建")

        return edges

    def _count_modalities(self, nodes: List[Dict]) -> Dict[str, int]:
        total_images = sum(n["attributes"].get("image_count", 0) for n in nodes)
        total_md_chars = sum(n["attributes"].get("markdown_length", 0) for n in nodes)
        return {
            "paper_count": len(nodes),
            "total_images": total_images,
            "total_markdown_chars": total_md_chars,
        }

    def _build_metadata(self, dataset_name: str, nodes: List[Dict], edges: List[Dict]) -> Dict:
        node_count = len(nodes)
        edge_count = len(edges)

        domain = "arxiv"
        if hasattr(self, '_formatter_data') and self._formatter_data:
            domain = self._formatter_data.get("domain", "arxiv")

        if domain == "reddit":
            return self._build_metadata_reddit(dataset_name, nodes, edges)
        elif domain == "amazon":
            return self._build_metadata_amazon(dataset_name, nodes, edges)
        elif (getattr(self, "_formatter_data", None) or {}).get("bundle") or domain in self._BUNDLE_DOMAIN_CONFIG:
            return self._build_metadata_bundle(domain, dataset_name, nodes, edges)
        else:
            return self._build_metadata_arxiv(dataset_name, nodes, edges)

    def _get_graph_mode(self) -> str:
        """获取当前 graph_mode"""
        if hasattr(self, '_current_task') and self._current_task:
            return self._current_task.get("graph_mode", "heterogeneous")
        return "heterogeneous"

    def _build_metadata_arxiv(self, dataset_name: str, nodes: List[Dict], edges: List[Dict]) -> Dict:
        node_count = len(nodes)
        edge_count = len(edges)
        total_images = sum(n.get("attributes", {}).get("image_count", 0) for n in nodes)
        total_md_chars = sum(n.get("attributes", {}).get("markdown_length", 0) for n in nodes)

        category = self._current_task.get("category", "") if hasattr(self, '_current_task') and self._current_task else ""

        return {
            "name": dataset_name,
            "domain": "arxiv",
            "created_at": datetime.now().strftime('%Y-%m-%d'),
            "category": category,
            "statistics": {
                "node_count": node_count,
                "edge_count": edge_count,
                "total_images": total_images,
                "total_markdown_chars": total_md_chars,
                "avg_images_per_paper": round(total_images / node_count, 1) if node_count > 0 else 0,
                "avg_md_length": round(total_md_chars / node_count) if node_count > 0 else 0,
            },
            "schema": {
                "node_types": ["paper"],
                "edge_types": ["citation"],
                "node_attributes": ["arxiv_id", "markdown_length", "image_count", "markdown_size"],
                "edge_attributes": ["reference_text"]
            }
        }

    def _build_metadata_reddit(self, dataset_name: str, nodes: List[Dict], edges: List[Dict]) -> Dict:
        node_count = len(nodes)
        edge_count = len(edges)
        graph_mode = self._get_graph_mode()

        # 按类型统计节点
        node_type_dist = {}
        for n in nodes:
            ntype = n.get("type", "unknown")
            node_type_dist[ntype] = node_type_dist.get(ntype, 0) + 1

        # 按类型统计边
        edge_type_dist = {}
        for e in edges:
            etype = e.get("type", "unknown")
            edge_type_dist[etype] = edge_type_dist.get(etype, 0) + 1

        if graph_mode == "homogeneous":
            schema = {
                "node_types": ["post"],
                "edge_types": ["co_comment"],
                "modalities": ["text", "image"],
                "label_field": "subreddit",
            }
        else:
            schema = {
                "node_types": ["user", "post", "comment", "subreddit"],
                "edge_types": ["author_of", "reply_to", "belongs_to", "crosspost"],
                "modalities": ["text", "image", "metadata"],
            }

        return {
            "name": dataset_name,
            "domain": "reddit",
            "graph_mode": graph_mode,
            "created_at": datetime.now().strftime('%Y-%m-%d'),
            "statistics": {
                "node_count": node_count,
                "edge_count": edge_count,
                "node_type_distribution": node_type_dist,
                "edge_type_distribution": edge_type_dist,
            },
            "schema": schema,
        }

    def _build_metadata_amazon(self, dataset_name: str, nodes: List[Dict], edges: List[Dict]) -> Dict:
        node_count = len(nodes)
        edge_count = len(edges)
        graph_mode = self._get_graph_mode()

        node_type_dist = {}
        for n in nodes:
            ntype = n.get("type", "unknown")
            node_type_dist[ntype] = node_type_dist.get(ntype, 0) + 1

        edge_type_dist = {}
        for e in edges:
            etype = e.get("type", "unknown")
            edge_type_dist[etype] = edge_type_dist.get(etype, 0) + 1

        if graph_mode == "homogeneous":
            schema = {
                "node_types": ["product"],
                "edge_types": ["co_purchase"],
                "modalities": ["text", "image"],
                "label_field": "leaf_category",
            }
        else:
            schema = {
                "node_types": ["product", "user", "category", "brand"],
                "edge_types": ["review", "also_bought", "also_viewed", "belongs_to", "made_by"],
                "modalities": ["text", "image", "price_rating"],
            }

        return {
            "name": dataset_name,
            "domain": "amazon",
            "graph_mode": graph_mode,
            "created_at": datetime.now().strftime('%Y-%m-%d'),
            "statistics": {
                "node_count": node_count,
                "edge_count": edge_count,
                "node_type_distribution": node_type_dist,
                "edge_type_distribution": edge_type_dist,
            },
            "schema": schema,
        }

    def _generate_dataset_description(self, results: Dict) -> str:
        prompt = f"""请根据以下信息生成数据集说明：
数据集名称: {results.get('dataset_name', '')}
节点数量: {results.get('stats', {}).get('node_count', 0)}
边数量: {results.get('stats', {}).get('edge_count', 0)}
模态分布: {json.dumps(results.get('stats', {}).get('modality_distribution', {}), ensure_ascii=False)}
请生成数据集的简要说明。"""
        return self.generate(prompt, temperature=0.3, max_tokens=500)

    def get_formatted_dataset(self) -> Dict:
        return self.formatted_dataset

    # ==================== Reddit 领域构建方法 ====================

    def _build_nodes_reddit(self, data: Dict) -> List[Dict]:
        """构建 Reddit 图节点 — 支持 homogeneous/heterogeneous 双模"""
        graph_mode = self._get_graph_mode()

        if graph_mode == "homogeneous":
            return self._build_nodes_reddit_homogeneous(data)
        else:
            return self._build_nodes_reddit_heterogeneous(data)

    def _build_nodes_reddit_homogeneous(self, data: Dict) -> List[Dict]:
        """构建 Reddit 同构图节点：只有 post"""
        nodes = []
        seen_ids = set()
        posts = data.get("posts", [])

        for post in posts:
            node_id = f"post_{post['post_id']}"
            if node_id not in seen_ids:
                seen_ids.add(node_id)
                nodes.append({
                    "id": node_id,
                    "type": "post",
                    "attributes": {
                        "post_id": post["post_id"],
                        "title": post.get("title", ""),
                        "text_length": post.get("text_length", 0),
                        "score": post.get("score", 0),
                        "upvote_ratio": post.get("upvote_ratio", 0),
                        "num_comments": post.get("num_comments", 0),
                        "has_image": post.get("has_image", False),
                        "created_utc": post.get("created_utc", 0),
                        "subreddit": post.get("subreddit", ""),  # label
                    }
                })

        logger.info(
            f"[DatasetFormatter] Reddit 同构节点: {len(nodes)} (post only)"
        )
        return nodes

    def _build_nodes_reddit_heterogeneous(self, data: Dict) -> List[Dict]:
        """构建 Reddit 异构图节点：user, post, comment, subreddit"""
        nodes = []
        seen_ids = set()

        posts = data.get("posts", [])
        comments_map = data.get("comments_map", {})
        user_metadata = data.get("user_metadata", {})
        subreddits = data.get("subreddits", [])

        # Subreddit 节点
        for sub in subreddits:
            node_id = f"subreddit_{sub.get('name', '')}"
            if node_id not in seen_ids:
                seen_ids.add(node_id)
                nodes.append({
                    "id": node_id,
                    "type": "subreddit",
                    "attributes": {
                        "name": sub.get("name", ""),
                        "subscribers": sub.get("subscribers", 0),
                        "description_length": sub.get("description_length", 0),
                    }
                })

        # 也从帖子中收集 subreddit（如果 subreddits 列表为空）
        for post in posts:
            sub_name = post.get("subreddit", "")
            if sub_name:
                node_id = f"subreddit_{sub_name}"
                if node_id not in seen_ids:
                    seen_ids.add(node_id)
                    nodes.append({
                        "id": node_id,
                        "type": "subreddit",
                        "attributes": {
                            "name": sub_name,
                            "subscribers": 0,
                            "description_length": 0,
                        }
                    })

        # Post 节点 — 多模态属性：文本(title + selftext) + 图片URL + 数值(score)
        for post in posts:
            node_id = f"post_{post['post_id']}"
            if node_id not in seen_ids:
                seen_ids.add(node_id)
                post_attrs = {
                    "title": post.get("title", ""),
                    "selftext": post.get("selftext", ""),
                    "title_length": post.get("title_length", len(post.get("title", ""))),
                    "text_length": post.get("text_length", 0),
                    "score": post.get("score", 0),
                    "upvote_ratio": post.get("upvote_ratio", 0),
                    "num_comments": post.get("num_comments", 0),
                    "has_image": post.get("has_image", False),
                    "image_count": post.get("image_count", 0),
                    "created_utc": post.get("created_utc", 0),
                    "link_flair_text": post.get("link_flair_text", ""),
                    "subreddit": post.get("subreddit", ""),
                }
                # 图片URL（多模态关键属性）
                if post.get("url"):
                    post_attrs["url"] = post["url"]
                if post.get("thumbnail") and post["thumbnail"].startswith("http"):
                    post_attrs["thumbnail"] = post["thumbnail"]

                nodes.append({
                    "id": node_id,
                    "type": "post",
                    "attributes": post_attrs,
                })

        # Comment 节点 — 文本属性
        for post_id, comments in comments_map.items():
            for comment in comments:
                node_id = f"comment_{comment['comment_id']}"
                if node_id not in seen_ids:
                    seen_ids.add(node_id)
                    nodes.append({
                        "id": node_id,
                        "type": "comment",
                        "attributes": {
                            "body": comment.get("body", ""),
                            "text_length": comment.get("body_length", 0),
                            "score": comment.get("score", 0),
                            "is_submitter": comment.get("is_submitter", False),
                            "created_utc": comment.get("created_utc", 0),
                            "depth": comment.get("depth", 0),
                        }
                    })

        # User 节点
        all_users = set()
        for post in posts:
            author = post.get("author", "")
            if author and author not in ("[deleted]", "[removed]"):
                all_users.add(author)
        for comments in comments_map.values():
            for comment in comments:
                author = comment.get("author", "")
                if author and author not in ("[deleted]", "[removed]"):
                    all_users.add(author)

        for username in all_users:
            node_id = f"user_{username}"
            if node_id not in seen_ids:
                seen_ids.add(node_id)
                meta = user_metadata.get(username, {})
                nodes.append({
                    "id": node_id,
                    "type": "user",
                    "attributes": {
                        "username": username,
                        "karma": meta.get("total_karma", 0),
                        "link_karma": meta.get("link_karma", 0),
                        "comment_karma": meta.get("comment_karma", 0),
                        "account_age_days": meta.get("account_age_days", 0),
                        "is_verified": meta.get("is_verified", False),
                    }
                })

        logger.info(
            f"[DatasetFormatter] Reddit 节点: {len(nodes)} "
            f"(types: {self._count_node_types(nodes)})"
        )
        return nodes

    def _build_edges_reddit(self, data: Dict) -> List[Dict]:
        """构建 Reddit 图边 — 支持 homogeneous/heterogeneous 双模"""
        graph_mode = self._get_graph_mode()

        if graph_mode == "homogeneous":
            return self._build_edges_reddit_homogeneous(data)
        else:
            return self._build_edges_reddit_heterogeneous(data)

    def _build_edges_reddit_homogeneous(self, data: Dict) -> List[Dict]:
        """构建 Reddit 同构图边：co-commenting (MAGB兼容)"""
        edges = []
        posts = data.get("posts", [])
        comments_map = data.get("comments_map", {})

        # 构建 user → set(post_ids) 映射
        user_to_posts = defaultdict(set)
        for post in posts:
            author = post.get("author", "")
            if author and author not in ("[deleted]", "[removed]"):
                user_to_posts[author].add(post["post_id"])

        for post_id, comments in comments_map.items():
            for comment in comments:
                author = comment.get("author", "")
                if author and author not in ("[deleted]", "[removed]"):
                    user_to_posts[author].add(post_id)

        # 对每个参与 >=2 帖子的用户，生成帖子对的无向边
        from itertools import combinations
        edge_set = set()
        for user, post_ids in user_to_posts.items():
            if len(post_ids) < 2:
                continue
            for p1, p2 in combinations(sorted(post_ids), 2):
                edge_key = frozenset([p1, p2])
                if edge_key not in edge_set:
                    edge_set.add(edge_key)
                    edges.append({
                        "source": f"post_{p1}",
                        "target": f"post_{p2}",
                        "type": "co_comment",
                        "attributes": {}
                    })
                    edges.append({
                        "source": f"post_{p2}",
                        "target": f"post_{p1}",
                        "type": "co_comment",
                        "attributes": {}
                    })

        logger.info(
            f"[DatasetFormatter] Reddit 同构边: {len(edges)} "
            f"({len(edges) // 2} 无向 co_comment 边)"
        )
        return edges

    def _build_edges_reddit_heterogeneous(self, data: Dict) -> List[Dict]:
        """构建 Reddit 异构图边"""
        edges = []
        posts = data.get("posts", [])
        comments_map = data.get("comments_map", {})
        post_ids = {p["post_id"] for p in posts}

        # author_of: user→post
        for post in posts:
            author = post.get("author", "")
            if author and author not in ("[deleted]", "[removed]"):
                edges.append({
                    "source": f"user_{author}",
                    "target": f"post_{post['post_id']}",
                    "type": "author_of",
                    "attributes": {"created_utc": post.get("created_utc", 0)}
                })

        # author_of: user→comment + reply_to: comment→post/comment
        for post_id, comments in comments_map.items():
            for comment in comments:
                author = comment.get("author", "")
                if author and author not in ("[deleted]", "[removed]"):
                    edges.append({
                        "source": f"user_{author}",
                        "target": f"comment_{comment['comment_id']}",
                        "type": "author_of",
                        "attributes": {"created_utc": comment.get("created_utc", 0)}
                    })

                parent_id = comment.get("parent_id", "")
                if parent_id.startswith("t3_"):
                    edges.append({
                        "source": f"comment_{comment['comment_id']}",
                        "target": f"post_{parent_id[3:]}",
                        "type": "reply_to",
                        "attributes": {"depth": comment.get("depth", 0)}
                    })
                elif parent_id.startswith("t1_"):
                    edges.append({
                        "source": f"comment_{comment['comment_id']}",
                        "target": f"comment_{parent_id[3:]}",
                        "type": "reply_to",
                        "attributes": {"depth": comment.get("depth", 0)}
                    })

        # belongs_to: post→subreddit
        for post in posts:
            subreddit = post.get("subreddit", "")
            if subreddit:
                edges.append({
                    "source": f"post_{post['post_id']}",
                    "target": f"subreddit_{subreddit}",
                    "type": "belongs_to",
                    "attributes": {}
                })

        # crosspost: post→post
        for post in posts:
            crosspost_parent = post.get("crosspost_parent", "")
            if crosspost_parent:
                parent_id = crosspost_parent.replace("t3_", "")
                if parent_id in post_ids:
                    edges.append({
                        "source": f"post_{post['post_id']}",
                        "target": f"post_{parent_id}",
                        "type": "crosspost",
                        "attributes": {"original_subreddit": post.get("subreddit", "")}
                    })

        edge_type_counts = {}
        for e in edges:
            etype = e.get("type", "")
            edge_type_counts[etype] = edge_type_counts.get(etype, 0) + 1

        logger.info(
            f"[DatasetFormatter] Reddit 边: {len(edges)} ({edge_type_counts})"
        )
        return edges

    # ==================== Amazon 领域构建方法 ====================

    def _build_nodes_amazon(self, data: Dict) -> List[Dict]:
        """构建 Amazon 图节点 — 支持 homogeneous/heterogeneous 双模"""
        graph_mode = self._get_graph_mode()

        if graph_mode == "homogeneous":
            return self._build_nodes_amazon_homogeneous(data)
        else:
            return self._build_nodes_amazon_heterogeneous(data)

    def _build_nodes_amazon_homogeneous(self, data: Dict) -> List[Dict]:
        """构建 Amazon 同构图节点：只有 product"""
        nodes = []
        seen_ids = set()
        products = data.get("products", {})

        for asin, meta in products.items():
            node_id = f"product_{asin}"
            if node_id not in seen_ids:
                seen_ids.add(node_id)
                # 标签 = 叶品类
                cat_path = meta.get("category_path", "")
                leaf_cat = ""
                if cat_path:
                    parts = [p.strip() for p in cat_path.split("|") if p.strip()]
                    if parts:
                        leaf_cat = parts[-1]

                nodes.append({
                    "id": node_id,
                    "type": "product",
                    "attributes": {
                        "asin": asin,
                        "title": meta.get("title", ""),
                        "title_length": meta.get("title_length", 0),
                        "price": meta.get("price", 0),
                        "image_count": meta.get("image_count", 0),
                        "has_image": bool(meta.get("image_url", "")),
                        "leaf_category": leaf_cat,  # label
                        "category_path": cat_path,
                    }
                })

        logger.info(
            f"[DatasetFormatter] Amazon 同构节点: {len(nodes)} (product only)"
        )
        return nodes

    def _build_nodes_amazon_heterogeneous(self, data: Dict) -> List[Dict]:
        """构建 Amazon 异构图节点：product, user, category, brand"""
        nodes = []
        seen_ids = set()

        products = data.get("products", {})
        reviews = data.get("reviews", {})
        categories = data.get("categories", {})
        brands = data.get("brands", {})

        # Product 节点
        for asin, meta in products.items():
            node_id = f"product_{asin}"
            if node_id not in seen_ids:
                seen_ids.add(node_id)
                # 多模态属性：文本(title + description) + 图片URL + 数值(price) + 类别(brand/category)
                attrs = {
                    "asin": asin,
                    "title": meta.get("title", ""),
                    "title_length": meta.get("title_length", 0),
                    "description": meta.get("description", ""),
                    "description_length": meta.get("description_length", 0),
                    "has_description": bool(meta.get("description", "")),
                    "price": meta.get("price", 0),
                    "brand": meta.get("brand", ""),
                    "image_count": meta.get("image_count", 0),
                    "has_image": bool(meta.get("image_url", "")),
                    "category_path": meta.get("category_path", ""),
                    "main_cat": meta.get("main_cat", ""),
                    "leaf_category": meta.get("category_path", "").split("|")[-1].strip() if meta.get("category_path") else "",
                }
                # 图片URL（多模态关键属性）
                if meta.get("image_url"):
                    attrs["image_url"] = meta["image_url"]
                if meta.get("image_urls_highres"):
                    attrs["image_urls_highres"] = meta["image_urls_highres"]
                elif meta.get("image_urls"):
                    attrs["image_urls"] = meta["image_urls"]

                nodes.append({
                    "id": node_id,
                    "type": "product",
                    "attributes": attrs,
                })

        # User 节点（从评论中提取）
        user_review_counts = {}
        user_total_rating = {}
        for review_list in reviews.values():
            for review in review_list:
                rid = review.get("reviewer_id", "")
                if rid:
                    user_review_counts[rid] = user_review_counts.get(rid, 0) + 1
                    user_total_rating[rid] = user_total_rating.get(rid, 0) + review.get("rating", 0)

        for rid, count in user_review_counts.items():
            node_id = f"user_{rid}"
            if node_id not in seen_ids:
                seen_ids.add(node_id)
                avg_rating = user_total_rating.get(rid, 0) / count if count > 0 else 0
                nodes.append({
                    "id": node_id,
                    "type": "user",
                    "attributes": {
                        "reviewer_id": rid,
                        "review_count": count,
                        "avg_rating": round(avg_rating, 2),
                    }
                })

        # Category 节点
        for cat_key, cat_info in categories.items():
            cat_name = cat_info.get("name", cat_key)
            node_id = f"cat_{cat_name}"
            if node_id not in seen_ids:
                seen_ids.add(node_id)
                nodes.append({
                    "id": node_id,
                    "type": "category",
                    "attributes": {
                        "name": cat_name,
                        "level": cat_info.get("level", 0),
                        "product_count": cat_info.get("count", 0),
                    }
                })

        # Brand 节点
        for brand_name, brand_info in brands.items():
            node_id = f"brand_{brand_name}"
            if node_id not in seen_ids:
                seen_ids.add(node_id)
                nodes.append({
                    "id": node_id,
                    "type": "brand",
                    "attributes": {
                        "name": brand_name,
                        "product_count": brand_info.get("product_count", 0),
                    }
                })

        logger.info(
            f"[DatasetFormatter] Amazon 节点: {len(nodes)} "
            f"(types: {self._count_node_types(nodes)})"
        )
        return nodes

    def _build_edges_amazon(self, data: Dict) -> List[Dict]:
        """构建 Amazon 图边 — 支持 homogeneous/heterogeneous 双模"""
        graph_mode = self._get_graph_mode()

        if graph_mode == "homogeneous":
            return self._build_edges_amazon_homogeneous(data)
        else:
            return self._build_edges_amazon_heterogeneous(data)

    def _build_edges_amazon_homogeneous(self, data: Dict) -> List[Dict]:
        """构建 Amazon 同构图边：co-purchasing (MAGB兼容)"""
        edges = []
        products = data.get("products", {})
        product_asins = set(products.keys())

        edge_set = set()
        for asin, meta in products.items():
            # 合并 also_buy + also_view
            related_all = list(set(
                meta.get("also_buy", []) + meta.get("also_view", [])
            ))

            for related in related_all:
                if related == asin:
                    continue
                if related not in product_asins:
                    continue

                edge_key = tuple(sorted([asin, related]))
                if edge_key not in edge_set:
                    edge_set.add(edge_key)
                    edges.append({
                        "source": f"product_{asin}",
                        "target": f"product_{related}",
                        "type": "co_purchase",
                        "attributes": {}
                    })
                    edges.append({
                        "source": f"product_{related}",
                        "target": f"product_{asin}",
                        "type": "co_purchase",
                        "attributes": {}
                    })

        logger.info(
            f"[DatasetFormatter] Amazon 同构边: {len(edges)} "
            f"({len(edges) // 2} 无向 co_purchase 边)"
        )
        return edges

    def _build_edges_amazon_heterogeneous(self, data: Dict) -> List[Dict]:
        """构建 Amazon 异构图边"""
        edges = []
        products = data.get("products", {})
        reviews = data.get("reviews", {})
        product_asins = set(products.keys())

        # review: user→product
        for asin, review_list in reviews.items():
            if asin not in product_asins:
                continue
            for review in review_list:
                edges.append({
                    "source": f"user_{review['reviewer_id']}",
                    "target": f"product_{asin}",
                    "type": "review",
                    "attributes": {
                        "rating": review.get("rating", 0),
                        "review_text": review.get("review_text", ""),
                        "review_text_length": review.get("review_text_length", 0),
                        "summary": review.get("summary", ""),
                        "helpful_votes": review.get("helpful_votes", 0),
                        "verified": review.get("verified", False),
                        "timestamp": review.get("timestamp", 0),
                    }
                })

        # also_bought: product→product
        for asin, meta in products.items():
            for related in meta.get("also_buy", []):
                if related in product_asins:
                    edges.append({
                        "source": f"product_{asin}",
                        "target": f"product_{related}",
                        "type": "also_bought",
                        "attributes": {}
                    })

        # also_viewed: product→product
        for asin, meta in products.items():
            for related in meta.get("also_view", []):
                if related in product_asins:
                    edges.append({
                        "source": f"product_{asin}",
                        "target": f"product_{related}",
                        "type": "also_viewed",
                        "attributes": {}
                    })

        # belongs_to: product→category
        for asin, meta in products.items():
            cat_path = meta.get("category_path", "")
            if cat_path:
                parts = [p.strip() for p in cat_path.split("|") if p.strip()]
                if parts:
                    leaf_cat = parts[-1]
                    edges.append({
                        "source": f"product_{asin}",
                        "target": f"cat_{leaf_cat}",
                        "type": "belongs_to",
                        "attributes": {"category_level": len(parts) - 1}
                    })

        # made_by: product→brand
        for asin, meta in products.items():
            brand = meta.get("brand", "")
            if brand:
                edges.append({
                    "source": f"product_{asin}",
                    "target": f"brand_{brand}",
                    "type": "made_by",
                    "attributes": {}
                })

        edge_type_counts = {}
        for e in edges:
            etype = e.get("type", "")
            edge_type_counts[etype] = edge_type_counts.get(etype, 0) + 1

        logger.info(
            f"[DatasetFormatter] Amazon 边: {len(edges)} ({edge_type_counts})"
        )
        return edges

    # ==================== 辅助方法 ====================

    @staticmethod
    def _count_node_types(nodes: List[Dict]) -> Dict[str, int]:
        """统计各类型节点数"""
        counts = {}
        for n in nodes:
            ntype = n.get("type", "unknown")
            counts[ntype] = counts.get(ntype, 0) + 1
        return counts
