"""
数据集封装Agent - ReAct版
将数据封装成多模态属性图数据集，支持LLM驱动的迭代决策
"""
from typing import Dict, Any, Optional, List
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
        """从blackboard或上游数据收集数据"""
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
                # GPU提取结果格式 → 转换
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

    def action_build_nodes(self) -> Dict:
        existing = self._accumulated_results.get("nodes", [])
        if existing:
            return {"status": "success", "nodes": existing,
                    "message": f"节点已构建({len(existing)}个)，跳过重复构建"}

        data = self._ensure_collected()
        papers = data.get("cleaned_papers", [])
        multimodal = data.get("cleaned_multimodal", [])

        nodes = self._build_nodes(papers, multimodal)

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
        papers = data.get("cleaned_papers", [])
        multimodal = data.get("cleaned_multimodal", [])
        nodes = self._accumulated_results.get("nodes", [])

        edges = self._build_edges(nodes, papers, multimodal)

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
        total_images = sum(n.get("attributes", {}).get("image_count", 0) for n in nodes)
        total_md_chars = sum(n.get("attributes", {}).get("markdown_length", 0) for n in nodes)

        category = self._current_task.get("category", "") if hasattr(self, '_current_task') and self._current_task else ""

        return {
            "name": dataset_name,
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
