"""
编码策略对比实验 - OpenMAG Q4
对比独立编码器 vs 统一编码器的跨模态检索质量
"""
import torch
import torch.nn.functional as F
import numpy as np
from loguru import logger


def compute_retrieval_metrics(text_embs, image_embs, ks=(1, 5, 10, 20)):
    """
    计算跨模态检索指标

    Args:
        text_embs: (N, D) 文本特征
        image_embs: (N, D) 图像特征
        ks: Recall@K的K值列表

    Returns:
        dict with T2I and I2T metrics
    """
    # L2归一化
    text_embs = F.normalize(text_embs.float(), p=2, dim=1)
    image_embs = F.normalize(image_embs.float(), p=2, dim=1)

    # 相似度矩阵
    sim_matrix = text_embs @ image_embs.t()  # (N, N)

    results = {}

    # Text-to-Image
    t2i_ranks = []
    for i in range(sim_matrix.size(0)):
        scores = sim_matrix[i]
        sorted_indices = torch.argsort(scores, descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        t2i_ranks.append(rank)

    t2i_ranks = np.array(t2i_ranks)
    results["t2i_mrr"] = float(np.mean(1.0 / t2i_ranks))
    for k in ks:
        results[f"t2i_hits@{k}"] = float(np.mean(t2i_ranks <= k))

    # Image-to-Text
    i2t_ranks = []
    for j in range(sim_matrix.size(1)):
        scores = sim_matrix[:, j]
        sorted_indices = torch.argsort(scores, descending=True)
        rank = (sorted_indices == j).nonzero(as_tuple=True)[0].item() + 1
        i2t_ranks.append(rank)

    i2t_ranks = np.array(i2t_ranks)
    results["i2t_mrr"] = float(np.mean(1.0 / i2t_ranks))
    for k in ks:
        results[f"i2t_hits@{k}"] = float(np.mean(i2t_ranks <= k))

    return results


def run_encoding_comparison(
    unified_text_embs, unified_image_embs,
    independent_text_embs=None, independent_image_embs=None,
    sample_size=None,
):
    """
    对比统一编码器和独立编码器的检索质量

    Args:
        unified_text_embs: 统一编码器的文本特征 (N, D)
        unified_image_embs: 统一编码器的图像特征 (N, D)
        independent_text_embs: 独立编码器的文本特征 (N, D') (可选)
        independent_image_embs: 独立编码器的图像特征 (N, D') (可选)
        sample_size: 采样大小（大数据集时使用）

    Returns:
        dict with metrics for each encoding strategy
    """
    results = {}

    # 过滤零向量节点（没有图片的论文）— 先对齐样本数
    n_common = min(unified_text_embs.size(0), unified_image_embs.size(0))
    unified_text_embs = unified_text_embs[:n_common]
    unified_image_embs = unified_image_embs[:n_common]
    if independent_text_embs is not None:
        independent_text_embs = independent_text_embs[:n_common]
    if independent_image_embs is not None:
        independent_image_embs = independent_image_embs[:n_common]

    text_norm = unified_text_embs.norm(dim=1)
    image_norm = unified_image_embs.norm(dim=1)
    valid_mask = (text_norm > 1e-6) & (image_norm > 1e-6)
    valid_indices = valid_mask.nonzero(as_tuple=True)[0]

    if len(valid_indices) < 100:
        logger.warning(f"有效样本过少: {len(valid_indices)}")
        return {}

    if sample_size and len(valid_indices) > sample_size:
        perm = torch.randperm(len(valid_indices))[:sample_size]
        valid_indices = valid_indices[perm]

    logger.info(f"编码对比实验: {len(valid_indices)} 个有效样本")

    # Unified (CLIP-Large 768d)
    text_u = unified_text_embs[valid_indices]
    image_u = unified_image_embs[valid_indices]
    unified_metrics = compute_retrieval_metrics(text_u, image_u)
    unified_metrics["encoder_type"] = "unified_clip_large_768d"
    results["unified"] = unified_metrics

    logger.info(f"Unified encoder: T2I MRR={unified_metrics['t2i_mrr']:.4f}, "
                f"I2T MRR={unified_metrics['i2t_mrr']:.4f}")

    # Independent (如果提供了独立编码器的特征)
    if independent_text_embs is not None and independent_image_embs is not None:
        # 对齐维度：投影到相同维度
        min_dim = min(independent_text_embs.size(1), independent_image_embs.size(1))
        text_i = independent_text_embs[valid_indices][:, :min_dim]
        image_i = independent_image_embs[valid_indices][:, :min_dim]

        ind_metrics = compute_retrieval_metrics(text_i, image_i)
        ind_metrics["encoder_type"] = "independent_sbert_clipbase"
        results["independent"] = ind_metrics

        logger.info(f"Independent encoder: T2I MRR={ind_metrics['t2i_mrr']:.4f}, "
                    f"I2T MRR={ind_metrics['i2t_mrr']:.4f}")

    return results
