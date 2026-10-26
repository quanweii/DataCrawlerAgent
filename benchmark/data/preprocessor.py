"""
数据预处理模块 - 从raw data构建PyG Data对象
扫描论文目录，提取多模态内容，构建图结构
"""
import os
import re
import json
import csv
import glob
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from loguru import logger


class AMMAGPreprocessor:
    """AMMAG数据集预处理器"""

    # 支持的领域及其标签映射
    DOMAIN_MAP = {
        "cs_si": 0,
        "cs_lo": 1,
        "math_lo": 2,
        "physics_socph": 3,
    }

    # 各领域内的子分类标签（用于域内节点分类）
    SUBCATEGORY_MAP = {
        "cs_si": {
            "cs.SI": 0,
            "cs.LG": 1,
            "physics.soc-ph": 2,
            "cs.CY": 3,
            "cs.AI": 4,
            "other": 5,
        },
        "cs_lo": {
            "cs.LO": 0,
            "cs.PL": 1,
            "math.LO": 2,
            "cs.AI": 3,
            "cs.FL": 4,
            "cs.CC": 5,
            "other": 6,
        },
        "math_lo": {
            "math.LO": 0,
            "cs.LO": 1,
            "math.CO": 2,
            "math.GN": 3,
            "math.GR": 4,
            "math.CT": 5,
            "other": 6,
        },
        "physics_socph": {
            "physics.soc-ph": 0,
            "cs.SI": 1,
            "cond-mat.stat-mech": 2,
            "physics.data-an": 3,
            "q-bio.PE": 4,
            "other": 5,
        },
    }

    def __init__(self, data_root: str = "./data"):
        self.data_root = data_root
        self._metadata_cache = None

    def scan_papers(self, domain: str) -> List[Dict]:
        """扫描某个领域的论文目录，返回论文信息列表"""
        domain_dir = os.path.join(self.data_root, f"{domain}_papers")
        if not os.path.isdir(domain_dir):
            logger.warning(f"目录不存在: {domain_dir}")
            return []

        # 加载元数据获取子分类标签
        metadata = self._load_arxiv_metadata()
        subcat_map = self.SUBCATEGORY_MAP.get(domain, {})

        papers = []
        for paper_dir in sorted(os.listdir(domain_dir)):
            paper_path = os.path.join(domain_dir, paper_dir)
            if not os.path.isdir(paper_path):
                continue

            # 从元数据获取子分类标签
            subcat_label = 0
            if metadata and subcat_map and paper_dir in metadata:
                cats = metadata[paper_dir].get("categories", "")
                cat_list = cats.split()
                subcat_label = self._assign_subcategory(cat_list, domain)

            info = {
                "arxiv_id": paper_dir,
                "domain": domain,
                "label": self.DOMAIN_MAP.get(domain, -1),
                "subcat_label": subcat_label,
                "pdf_path": os.path.join(paper_path, f"{paper_dir}.pdf"),
                "md_path": os.path.join(paper_path, f"{paper_dir}.md"),
                "npy_path": os.path.join(paper_path, f"{paper_dir}.npy"),
                "images_dir": os.path.join(paper_path, "images"),
                "has_pdf": os.path.exists(os.path.join(paper_path, f"{paper_dir}.pdf")),
                "has_md": os.path.exists(os.path.join(paper_path, f"{paper_dir}.md")),
                "has_images": os.path.isdir(os.path.join(paper_path, "images")),
            }

            if info["has_md"]:
                info["images"] = self._list_images(info["images_dir"])
                info["num_images"] = len(info["images"])
                info["text_length"] = self._get_text_length(info["md_path"])
                info["formulas"] = self._extract_formulas(info["md_path"])
                info["num_formulas"] = len(info["formulas"])
                papers.append(info)

        logger.info(f"[{domain}] 扫描完成: {len(papers)} 篇论文有有效数据")
        return papers

    def _list_images(self, images_dir: str) -> List[str]:
        """列出图片目录中的所有图片文件"""
        if not os.path.isdir(images_dir):
            return []
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        return sorted([
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if os.path.splitext(f)[1].lower() in exts
        ])

    def _load_arxiv_metadata(self) -> Optional[Dict]:
        """加载arXiv元数据（按ID索引）"""
        if self._metadata_cache is not None:
            return self._metadata_cache

        metadata_path = os.path.join(
            os.path.dirname(self.data_root), "arxiv-metadata-cleaned.json"
        )
        # 也尝试项目根目录
        if not os.path.exists(metadata_path):
            metadata_path = os.path.join(self.data_root, "..", "arxiv-metadata-cleaned.json")

        if not os.path.exists(metadata_path):
            logger.warning(f"未找到元数据文件: {metadata_path}")
            return None

        logger.info(f"加载元数据: {metadata_path} ...")
        meta = {}
        count = 0
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    p = json.loads(line.strip())
                    aid = p.get("id", "")
                    meta[aid] = {"categories": p.get("categories", ""), "title": p.get("title", "")}
                    count += 1
                except Exception:
                    continue
        logger.info(f"元数据加载完成: {count} 篇")
        self._metadata_cache = meta
        return meta

    # domain directory name → arXiv primary category
    DOMAIN_TO_ARXIV_CAT = {
        "cs_si": "cs.SI",
        "cs_lo": "cs.LO",
        "math_lo": "math.LO",
        "physics_socph": "physics.soc-ph",
    }

    def _assign_subcategory(self, cat_list: List[str], domain: str) -> int:
        """根据论文的categories分配子分类标签"""
        subcat_map = self.SUBCATEGORY_MAP.get(domain, {})
        if not subcat_map:
            return 0

        domain_cat = self.DOMAIN_TO_ARXIV_CAT.get(domain, domain.replace("_", "."))
        non_primary = [c for c in cat_list if c != domain_cat]

        # 仅含primary category的论文 → 归入primary子类(标签0)
        if not non_primary:
            return subcat_map.get(domain_cat, 0)

        # 按非primary category分配子类
        for cat in non_primary:
            if cat in subcat_map:
                return subcat_map[cat]

        return subcat_map.get("other", 0)

    def _get_text_length(self, md_path: str) -> int:
        """获取markdown文件的字符数"""
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                return len(f.read())
        except Exception:
            return 0

    def _extract_formulas(self, md_path: str) -> List[str]:
        """从markdown文件中提取LaTeX公式"""
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            return []

        # 块级公式 $$...$$
        block_formulas = re.findall(r'\$\$(.+?)\$\$', text, re.DOTALL)
        # 行内公式 $...$ (排除 $$)
        inline_formulas = re.findall(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', text)

        formulas = block_formulas + inline_formulas
        # 过滤太短的公式（可能是误匹配）
        return [f.strip() for f in formulas if len(f.strip()) > 3]

    def load_citation_edges(self, domain: str) -> List[Tuple[str, str]]:
        """加载引用边CSV文件"""
        graphs_dir = os.path.join(self.data_root, f"{domain}_graphs")
        if not os.path.isdir(graphs_dir):
            return []

        edges = []
        for csv_file in glob.glob(os.path.join(graphs_dir, "*.csv")):
            # 读取文件并清除NUL字节
            with open(csv_file, "rb") as f:
                raw = f.read()
            cleaned = raw.replace(b"\x00", b"").decode("utf-8", errors="replace")

            import io
            reader = csv.DictReader(io.StringIO(cleaned))
            for row in reader:
                try:
                    src = row.get("source_id", "").strip()
                    dst = row.get("target_id", "").strip()
                    if src and dst:
                        edges.append((src, dst))
                except Exception:
                    continue

        logger.info(f"[{domain}] 加载引用边: {len(edges)} 条")
        return edges

    def build_graph_data(
        self,
        domains: List[str],
        output_dir: str = "./benchmark/processed",
    ) -> Dict:
        """
        构建完整的图数据，支持多领域合并。
        仅保留两端都在数据集内的引用边（方案A）。

        Returns:
            dict with keys: node_ids, node_labels, edge_index, paper_info, statistics
        """
        os.makedirs(output_dir, exist_ok=True)

        all_papers = []
        paper_id_map = {}  # arxiv_id -> integer index

        for domain in domains:
            papers = self.scan_papers(domain)
            for p in papers:
                if p["arxiv_id"] not in paper_id_map:
                    paper_id_map[p["arxiv_id"]] = len(paper_id_map)
                    all_papers.append(p)

        logger.info(f"总共 {len(all_papers)} 篇论文, {len(paper_id_map)} 个唯一节点")

        # 构建边：仅保留两端都在 paper_id_map 中的边（过滤外部节点）
        src_list, dst_list = [], []
        total_raw_edges = 0
        for domain in domains:
            edges = self.load_citation_edges(domain)
            total_raw_edges += len(edges)
            for src, dst in edges:
                # 过滤自环（论文引用自身是数据噪声）
                if src != dst and src in paper_id_map and dst in paper_id_map:
                    src_list.append(paper_id_map[src])
                    dst_list.append(paper_id_map[dst])

        kept_edges = len(src_list)
        self_loops_filtered = total_raw_edges - len(edges)  # approximate
        logger.info(f"引用边: 原始{total_raw_edges}条, 保留域内(去自环){kept_edges}条 "
                     f"({kept_edges/total_raw_edges*100:.1f}%)")

        if src_list:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        labels = torch.tensor([p["label"] for p in all_papers], dtype=torch.long)
        subcat_labels = torch.tensor([p.get("subcat_label", 0) for p in all_papers], dtype=torch.long)

        # 统计信息
        stats = self._compute_statistics(all_papers, edge_index, subcat_labels, domains)

        result = {
            "paper_id_map": paper_id_map,
            "labels": labels,
            "edge_index": edge_index,
            "papers": all_papers,
            "statistics": stats,
        }

        # 保存
        torch.save({
            "edge_index": edge_index,
            "labels": labels,
            "subcat_labels": subcat_labels,
            "paper_ids": list(paper_id_map.keys()),
        }, os.path.join(output_dir, "graph_data.pt"))

        with open(os.path.join(output_dir, "statistics.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        logger.info(f"图数据已保存到 {output_dir}")
        logger.info(f"节点数: {len(all_papers)}, 边数: {edge_index.size(1)}, "
                     f"标签分布: {dict(stats.get('label_distribution', {}))}")

        return result

    def _compute_statistics(
        self,
        papers: List[Dict],
        edge_index: torch.Tensor,
        labels: torch.Tensor,
        domains: List[str],
    ) -> Dict:
        """计算数据集统计信息"""
        num_nodes = len(papers)
        num_edges = edge_index.size(1)

        # 文本长度统计
        text_lengths = [p["text_length"] for p in papers]
        # 图片数统计
        img_counts = [p["num_images"] for p in papers]
        # 公式数统计
        formula_counts = [p["num_formulas"] for p in papers]

        # 度统计
        if num_edges > 0:
            degrees = torch.zeros(num_nodes, dtype=torch.long)
            degrees.scatter_add_(0, edge_index[0], torch.ones(num_edges, dtype=torch.long))
            degrees.scatter_add_(0, edge_index[1], torch.ones(num_edges, dtype=torch.long))
            avg_degree = degrees.float().mean().item()
        else:
            avg_degree = 0.0

        # 标签分布
        label_dist = defaultdict(int)
        for p in papers:
            label_dist[p["domain"]] += 1

        # 各领域统计
        domain_stats = {}
        for domain in domains:
            domain_papers = [p for p in papers if p["domain"] == domain]
            if not domain_papers:
                continue
            domain_stats[domain] = {
                "num_papers": len(domain_papers),
                "avg_text_length": np.mean([p["text_length"] for p in domain_papers]),
                "avg_images": np.mean([p["num_images"] for p in domain_papers]),
                "avg_formulas": np.mean([p["num_formulas"] for p in domain_papers]),
                "papers_with_images": sum(1 for p in domain_papers if p["num_images"] > 0),
                "papers_with_formulas": sum(1 for p in domain_papers if p["num_formulas"] > 0),
            }

        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "avg_degree": avg_degree,
            "num_classes": len(set(labels.tolist())),
            "label_distribution": dict(label_dist),
            "text_length": {
                "mean": float(np.mean(text_lengths)),
                "median": float(np.median(text_lengths)),
                "min": int(np.min(text_lengths)),
                "max": int(np.max(text_lengths)),
            },
            "image_count": {
                "mean": float(np.mean(img_counts)),
                "median": float(np.median(img_counts)),
                "papers_with_images": sum(1 for c in img_counts if c > 0),
                "papers_without_images": sum(1 for c in img_counts if c == 0),
            },
            "formula_count": {
                "mean": float(np.mean(formula_counts)),
                "median": float(np.median(formula_counts)),
                "papers_with_formulas": sum(1 for c in formula_counts if c > 0),
                "papers_without_formulas": sum(1 for c in formula_counts if c == 0),
            },
            "domain_stats": domain_stats,
        }
