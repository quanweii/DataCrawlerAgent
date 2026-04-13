"""
数据集封装Agent - 负责将数据封装成多模态属性图数据集的标准格式
"""
from typing import Dict, Any, Optional, List
from openai import OpenAI
from loguru import logger
import os
import json
from datetime import datetime

from .base_agent import BaseAgent


class DatasetFormatterAgent(BaseAgent):
    """数据集封装Agent - 第三个分管Agent"""

    def __init__(
        self,
        client: OpenAI,
        system_prompt: str,
        mcp_tool_manager: Optional[Any] = None
    ):
        super().__init__(
            client=client,
            system_prompt=system_prompt,
            mcp_tool_manager=mcp_tool_manager,
            agent_name="DatasetFormatterAgent"
        )
        self.formatted_dataset: Dict = {}

    def execute(
        self,
        task: Dict[str, Any],
        upstream_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        执行数据集封装任务

        Args:
            task: 封装参数，包含：
                - dataset_name: 数据集名称
                - output_format: 输出格式（json/parquet等）
                - output_dir: 输出目录
                - include_metadata: 是否包含元数据
            upstream_data: 上游Agent（DataCleanerAgent）传递的数据

        Returns:
            Dict[str, Any]: 封装结果
        """
        logger.info(f"[{self.agent_name}] 开始执行数据集封装任务")

        if not upstream_data:
            return {
                "status": "error",
                "message": "没有接收到上游数据"
            }

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
            "stats": {
                "node_count": 0,
                "edge_count": 0,
                "modality_distribution": {}
            }
        }

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 获取清洗后的数据
        cleaned_papers = upstream_data.get("cleaned_papers", [])
        cleaned_multimodal = upstream_data.get("cleaned_multimodal", [])

        # 构建属性图节点
        nodes = self._build_nodes(cleaned_papers, cleaned_multimodal)
        results["nodes"] = nodes
        results["stats"]["node_count"] = len(nodes)

        # 构建属性图边（关系）
        edges = self._build_edges(nodes, cleaned_papers, cleaned_multimodal)
        results["edges"] = edges
        results["stats"]["edge_count"] = len(edges)

        # 统计模态分布
        modality_dist = self._count_modalities(nodes)
        results["stats"]["modality_distribution"] = modality_dist

        # 构建元数据
        if include_metadata:
            results["metadata"] = self._build_metadata(
                dataset_name,
                cleaned_papers,
                cleaned_multimodal,
                upstream_data.get("stats", {})
            )

        # 格式化为属性图格式
        attribute_graph = self.call_mcp_tool(
            "format_attribute_graph",
            nodes=nodes,
            edges=edges,
            metadata=results["metadata"]
        )

        # 保存数据集
        output_path = os.path.join(output_dir, f"{dataset_name}.{output_format}")
        save_result = self.call_mcp_tool(
            "save_dataset",
            dataset=attribute_graph.get("graph", {}) if attribute_graph else {"nodes": nodes, "edges": edges},
            output_path=output_path,
            format=output_format
        )

        if save_result and save_result.get("status") == "success":
            results["output_path"] = save_result.get("path", output_path)
        else:
            results["status"] = "error"
            results["message"] = save_result.get("message", "保存失败") if save_result else "保存失败"

        # 保存格式化结果
        self.formatted_dataset = results

        # 生成数据集说明
        description = self._generate_dataset_description(results)
        results["description"] = description

        logger.info(f"[{self.agent_name}] 数据集封装完成: {results['stats']['node_count']}节点, {results['stats']['edge_count']}边")

        return results

    def _build_nodes(
        self,
        papers: List[Dict],
        multimodal: List[Dict]
    ) -> List[Dict]:
        """构建属性图节点"""
        nodes = []
        node_id_counter = 0

        # 创建论文节点
        for paper in papers:
            node = {
                "id": f"paper_{node_id_counter}",
                "type": "paper",
                "attributes": {
                    "title": paper.get("title", ""),
                    "source": paper.get("source", {}),
                    "path": paper.get("path", ""),
                    "cleaned": paper.get("cleaned", False),
                    "cleaned_at": paper.get("cleaned_at", "")
                },
                "modalities": ["text"]
            }
            nodes.append(node)
            node_id_counter += 1

        # 创建多模态数据节点
        for item in multimodal:
            modality = item.get("modality", "unknown")
            node = {
                "id": f"{modality}_{node_id_counter}",
                "type": modality,
                "attributes": {
                    "source_paper": item.get("source", ""),
                    "content": item.get("content", ""),
                    "path": item.get("path", ""),
                    "index": item.get("index", 0),
                    "cleaned": item.get("cleaned", False)
                },
                "modalities": [modality]
            }

            # 根据模态类型设置特定属性
            if modality == "text":
                node["attributes"]["word_count"] = len(item.get("content", "").split())
            elif modality == "image":
                node["attributes"]["image_path"] = item.get("path", "")
            elif modality == "table":
                node["attributes"]["row_count"] = len(item.get("content", [])) if isinstance(item.get("content"), list) else 0

            nodes.append(node)
            node_id_counter += 1

        return nodes

    def _build_edges(
        self,
        nodes: List[Dict],
        papers: List[Dict],
        multimodal: List[Dict]
    ) -> List[Dict]:
        """构建属性图边（关系）"""
        edges = []
        edge_id_counter = 0

        # 建立论文与多模态数据的关系
        paper_nodes = [n for n in nodes if n["type"] == "paper"]
        other_nodes = [n for n in nodes if n["type"] != "paper"]

        for paper_node in paper_nodes:
            paper_path = paper_node["attributes"].get("path", "")

            # 查找属于该论文的多模态数据
            for other_node in other_nodes:
                if other_node["attributes"].get("source_paper") == paper_path:
                    edge = {
                        "id": f"edge_{edge_id_counter}",
                        "source": paper_node["id"],
                        "target": other_node["id"],
                        "type": "contains",
                        "attributes": {
                            "relationship": "paper_contains_modality",
                            "modality": other_node["type"]
                        }
                    }
                    edges.append(edge)
                    edge_id_counter += 1

        # 建立多模态数据之间的关联（如同页文本和图像）
        for i, node1 in enumerate(other_nodes):
            for node2 in other_nodes[i+1:]:
                if node1["attributes"].get("source_paper") == node2["attributes"].get("source_paper"):
                    # 来自同一篇论文的多模态数据建立关联
                    edge = {
                        "id": f"edge_{edge_id_counter}",
                        "source": node1["id"],
                        "target": node2["id"],
                        "type": "related",
                        "attributes": {
                            "relationship": "same_source_paper"
                        }
                    }
                    edges.append(edge)
                    edge_id_counter += 1

        return edges

    def _count_modalities(self, nodes: List[Dict]) -> Dict[str, int]:
        """统计各模态的节点数量"""
        distribution = {}
        for node in nodes:
            for modality in node.get("modalities", []):
                distribution[modality] = distribution.get(modality, 0) + 1
        return distribution

    def _build_metadata(
        self,
        dataset_name: str,
        papers: List[Dict],
        multimodal: List[Dict],
        cleaning_stats: Dict
    ) -> Dict:
        """构建数据集元数据"""
        return {
            "name": dataset_name,
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0",
            "statistics": {
                "paper_count": len(papers),
                "multimodal_item_count": len(multimodal),
                "cleaning_stats": cleaning_stats
            },
            "schema": {
                "node_types": ["paper", "text", "image", "table"],
                "edge_types": ["contains", "related"],
                "attribute_keys": ["title", "source", "path", "content", "modality"]
            },
            "format": "multimodal_attribute_graph",
            "description": "多模态属性图数据集，包含论文及其关联的多模态数据"
        }

    def _generate_dataset_description(self, results: Dict) -> str:
        """生成数据集说明"""
        prompt = f"""请根据以下信息生成数据集说明：

数据集名称: {results['dataset_name']}
节点数量: {results['stats']['node_count']}
边数量: {results['stats']['edge_count']}
模态分布: {json.dumps(results['stats']['modality_distribution'], ensure_ascii=False)}

元数据:
{json.dumps(results['metadata'], ensure_ascii=False, indent=2)}

请生成数据集的详细说明，包括数据结构、使用方法和注意事项。"""

        return self.generate(prompt, temperature=0.3, max_tokens=800)

    def get_formatted_dataset(self) -> Dict:
        """获取格式化后的数据集"""
        return self.formatted_dataset
