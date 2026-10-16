"""
多模态编码器 - 融合文本、图像、公式embedding
支持：concat, attention（可学习加权）
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional
from loguru import logger


class ModalityAttentionFusion(nn.Module):
    """
    可学习的模态注意力融合
    对每种模态学一个权重标量，加权求和后投影
    比简单拼接更灵活，能自适应调节模态贡献
    """

    def __init__(self, modality_dims: Dict[str, int], output_dim: int = 256):
        """
        Args:
            modality_dims: {"text": 384, "image": 512, "formula": 384}
            output_dim: 投影后的输出维度
        """
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.output_dim = output_dim

        # 每种模态一个投影层，映射到统一维度
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in modality_dims.items()
        })

        # 可学习的模态权重（初始化为均匀）
        self.modality_weights = nn.Parameter(
            torch.ones(len(self.modality_names)) / len(self.modality_names)
        )

    def forward(self, modality_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            modality_embeddings: {"text": Tensor(N, 384), "image": Tensor(N, 512), ...}
        Returns:
            Tensor(N, output_dim)
        """
        # 投影到统一维度
        projected = []
        for name in self.modality_names:
            if name in modality_embeddings:
                projected.append(self.projections[name](modality_embeddings[name]))

        # softmax归一化权重
        weights = F.softmax(self.modality_weights, dim=0)

        # 加权求和
        result = torch.zeros_like(projected[0])
        for i, p in enumerate(projected):
            result = result + weights[i] * p

        return result

    def get_weights(self) -> Dict[str, float]:
        """返回当前学到的模态权重"""
        weights = F.softmax(self.modality_weights, dim=0)
        return {name: weights[i].item() for i, name in enumerate(self.modality_names)}


class MultimodalEncoder:
    """多模态特征融合编码器"""

    def __init__(self, fusion: str = "concat"):
        """
        Args:
            fusion: 融合方式
                - concat: 简单拼接
                - mean: 平均
                - weighted: 加权平均（需要权重）
        """
        self.fusion = fusion

    def fuse(
        self,
        embeddings_dict: Dict[str, torch.Tensor],
        ids: List[str],
        output_dir: str,
        modalities: Optional[List[str]] = None,
    ) -> dict:
        """
        融合多种模态的embedding

        Args:
            embeddings_dict: {modality: {"ids": [...], "embeddings": Tensor}}
            ids: 需要融合的论文ID列表（取交集）
            output_dir: 输出目录
            modalities: 要使用的模态列表，None表示全部

        Returns:
            dict with save path and metadata
        """
        if modalities is None:
            modalities = list(embeddings_dict.keys())

        # 对齐ID顺序
        id_sets = []
        for mod in modalities:
            if mod in embeddings_dict:
                id_sets.append(set(embeddings_dict[mod]["ids"]))

        if not id_sets:
            logger.error("无有效的模态embedding")
            return {}

        common_ids = set(ids)
        for s in id_sets:
            common_ids = common_ids & s
        common_ids = sorted(common_ids)

        if not common_ids:
            logger.error("论文ID交集为空")
            return {}

        # 按common_ids的顺序收集各模态embedding
        aligned_embs = []
        for mod in modalities:
            if mod not in embeddings_dict:
                continue
            data = embeddings_dict[mod]
            id_to_idx = {pid: i for i, pid in enumerate(data["ids"])}
            indices = [id_to_idx[pid] for pid in common_ids]
            emb = data["embeddings"][indices]
            aligned_embs.append(emb)

        # 融合
        if self.fusion == "concat":
            fused = torch.cat(aligned_embs, dim=1)
        elif self.fusion == "mean":
            stacked = torch.stack(aligned_embs, dim=0)
            fused = stacked.mean(dim=0)
        else:
            fused = torch.cat(aligned_embs, dim=1)

        os.makedirs(output_dir, exist_ok=True)
        mod_str = "+".join(modalities)
        save_path = os.path.join(output_dir, f"multimodal_{mod_str}_embeddings.pt")
        torch.save({"ids": common_ids, "embeddings": fused}, save_path)

        logger.info(f"多模态融合完成: modalities={modalities}, "
                     f"papers={len(common_ids)}, dim={fused.size(1)}")
        return {"path": save_path, "ids": common_ids, "dim": fused.size(1)}
