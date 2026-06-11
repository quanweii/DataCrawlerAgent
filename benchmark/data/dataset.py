"""
PyG Dataset - AMMAG 多模态属性图数据集
支持多域合并、子图提取、跨域分割、per-domain评估
"""
import os
import torch
import numpy as np
from typing import List, Tuple, Optional
from torch_geometric.data import Data
from loguru import logger


class AMMAGDataset:
    """AMMAG多模态属性图数据集"""

    DOMAIN_MAP = {"cs_si": 0, "cs_lo": 1, "math_lo": 2, "physics_socph": 3}
    DOMAIN_NAMES = ["cs.SI", "cs.LO", "math.LO", "physics.soc-ph"]
    DOMAIN_KEYS = ["cs_si", "cs_lo", "math_lo", "physics_socph"]

    # 每域子分类类别数（修复后：从0开始连续）
    SUBCAT_CLASSES = {
        "cs_si": 6,
        "cs_lo": 7,
        "math_lo": 7,
        "physics_socph": 6,
    }

    def __init__(
        self,
        root: str,
        domains: List[str] = None,
        modality: str = "text",
        label_type: str = "domain",
    ):
        """
        Args:
            root: 项目根目录
            domains: 要加载的域列表（None=全部4域）
            modality: 模态字符串，如 "text", "image+formula", "text+image+formula"
            label_type: "domain"=4类跨域标签, "subcat"=每域子分类标签
        """
        self.root = root
        self.domains = domains or self.DOMAIN_KEYS
        self.modality = modality
        self.label_type = label_type
        self.processed_dir = os.path.join(root, "benchmark", "processed")

        self.data, self._graph_raw, self._domain_labels = self._load_data()
        logger.info(f"Dataset loaded: {self.data.num_nodes} nodes, "
                     f"{self.data.num_edges} edges, x.shape={self.data.x.shape}, "
                     f"label_type={label_type}, classes={self.data.y.unique().numel()}")

    def _load_data(self) -> Tuple[Data, dict, torch.Tensor]:
        """从processed目录加载graph + embeddings"""
        graph_path = os.path.join(self.processed_dir, "graph_data.pt")
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"未找到 graph_data.pt: {graph_path}")

        graph_data = torch.load(graph_path, map_location="cpu")
        edge_index = graph_data["edge_index"]
        domain_labels = graph_data["labels"]  # 4-class domain labels (always)
        paper_ids = graph_data["paper_ids"]

        if self.label_type == "subcat":
            labels = graph_data["subcat_labels"]
        else:
            labels = graph_data["labels"]

        embeddings = self._load_embeddings(paper_ids)

        data = Data(
            x=embeddings,
            edge_index=edge_index,
            y=labels,
            num_nodes=len(paper_ids),
        )
        return data, graph_data, domain_labels

    def _load_embeddings(self, paper_ids: List[str]) -> torch.Tensor:
        modality_parts = self.modality.split("+")
        embeddings_list = []

        for mod in modality_parts:
            emb_path = os.path.join(self.processed_dir, f"{mod}_embeddings.pt")
            if not os.path.exists(emb_path):
                logger.warning(f"未找到 {mod} embedding: {emb_path}")
                continue

            emb_data = torch.load(emb_path, map_location="cpu")
            emb_ids = emb_data["ids"]
            emb_tensor = emb_data["embeddings"]

            id_to_idx = {pid: i for i, pid in enumerate(emb_ids)}
            dim = emb_tensor.size(1)
            aligned = torch.zeros(len(paper_ids), dim)
            matched = 0
            for j, pid in enumerate(paper_ids):
                if pid in id_to_idx:
                    aligned[j] = emb_tensor[id_to_idx[pid]]
                    matched += 1

            logger.info(f"  {mod}: {matched}/{len(paper_ids)} matched, dim={dim}")
            embeddings_list.append(aligned)

        if not embeddings_list:
            raise ValueError(f"无有效的embedding，modality={self.modality}")

        return torch.cat(embeddings_list, dim=1)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.data

    # ---- 分割方法 ----

    def get_split_masks(self, split: str = "supervised", train_ratio=0.6,
                        val_ratio=0.2, seed: int = 42, few_shot_k: int = -1):
        """标准监督/few-shot分割"""
        num_nodes = self.data.num_nodes
        labels = self.data.y.cpu().numpy()
        num_classes = int(labels.max()) + 1

        rng = np.random.RandomState(seed)
        indices = np.arange(num_nodes)

        if split == "few_shot" and few_shot_k > 0:
            train_indices = []
            for c in range(num_classes):
                class_indices = indices[labels == c]
                if len(class_indices) >= few_shot_k:
                    chosen = rng.choice(class_indices, few_shot_k, replace=False)
                else:
                    chosen = class_indices
                train_indices.extend(chosen)

            remaining = np.array([i for i in indices if i not in set(train_indices)])
            rng.shuffle(remaining)
            val_size = int(len(remaining) * 0.2)
            val_indices = remaining[:val_size]
            test_indices = remaining[val_size:]
        else:
            rng.shuffle(indices)
            train_size = int(num_nodes * train_ratio)
            val_size = int(num_nodes * val_ratio)
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True

        return train_mask, val_mask, test_mask

    def get_domain_indices(self, domain_name: str) -> torch.Tensor:
        """返回某域节点的index tensor"""
        domain_idx = self.DOMAIN_MAP.get(domain_name)
        if domain_idx is None:
            raise ValueError(f"Unknown domain: {domain_name}")
        return (self._domain_labels == domain_idx).nonzero(as_tuple=True)[0]

    def get_cross_domain_masks(
        self, source_domain: str, target_domain: str, seed: int = 42
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        跨域迁移分割：所有非target域节点用于训练/验证，target域用于测试
        GNN在完整图上forward（利用跨域边），训练时看到多种domain标签
        """
        tgt_indices = self.get_domain_indices(target_domain)
        tgt_idx = self.DOMAIN_MAP[target_domain]

        # 所有非target域的节点
        non_tgt_mask = self._domain_labels != tgt_idx
        non_tgt_indices = non_tgt_mask.nonzero(as_tuple=True)[0]

        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(non_tgt_indices))
        non_tgt_np = non_tgt_indices.numpy()

        val_size = int(len(non_tgt_np) * 0.2)
        val_indices = non_tgt_np[perm[:val_size]]
        train_indices = non_tgt_np[perm[val_size:]]

        train_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[tgt_indices] = True

        return train_mask, val_mask, test_mask

    def get_domain_subgraph(self, domain_name: str) -> Data:
        """提取单域子图，节点重编号，边过滤，返回新Data"""
        node_indices = self.get_domain_indices(domain_name)
        node_set = set(node_indices.tolist())

        # 旧→新节点映射
        old_to_new = {}
        for new_idx, old_idx in enumerate(node_indices.tolist()):
            old_to_new[old_idx] = new_idx

        # 过滤边（两端都在域内）
        ei = self.data.edge_index
        src, dst = ei[0].tolist(), ei[1].tolist()
        new_src, new_dst = [], []
        for s, d in zip(src, dst):
            if s in node_set and d in node_set:
                new_src.append(old_to_new[s])
                new_dst.append(old_to_new[d])

        if new_src:
            new_edge_index = torch.tensor([new_src, new_dst], dtype=torch.long)
        else:
            new_edge_index = torch.zeros((2, 0), dtype=torch.long)

        # 过滤特征和标签
        new_x = self.data.x[node_indices]
        new_y = self.data.y[node_indices]

        domain_idx = self.DOMAIN_MAP[domain_name]
        logger.info(f"子图 {domain_name}: {len(node_indices)} nodes, "
                     f"{len(new_src)} edges, classes={new_y.unique().numel()}")

        return Data(
            x=new_x,
            edge_index=new_edge_index,
            y=new_y,
            num_nodes=len(node_indices),
        )

    @staticmethod
    def compute_per_domain_accuracy(
        pred: torch.Tensor,
        labels: torch.Tensor,
        domain_labels: torch.Tensor,
        test_mask: torch.Tensor,
    ) -> dict:
        """计算test_mask中每个域的accuracy"""
        from sklearn.metrics import accuracy_score
        results = {}
        for name, idx in AMMAGDataset.DOMAIN_MAP.items():
            domain_mask = (domain_labels[test_mask.cpu()] == idx)
            if domain_mask.sum() == 0:
                continue
            acc = accuracy_score(
                labels[test_mask][domain_mask].cpu().numpy(),
                pred[test_mask][domain_mask].cpu().numpy(),
            )
            results[name] = float(acc)
        return results
