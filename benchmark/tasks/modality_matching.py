"""
模态匹配与对齐实验 - OpenMAG P2
Modality Matching: 二分类判断模态对是否来自同一节点
Modality Alignment: CKA/CCA衡量模态间对齐程度
Modality Retrieval: 跨模态检索 MRR/Hits@K
"""
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
from loguru import logger


# ===== CKA (Centered Kernel Alignment) =====

def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """计算线性CKA相似度"""
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    hsic_xx = (X @ X.T).norm()
    hsic_yy = (Y @ Y.T).norm()
    hsic_xy = (X @ Y.T).norm()

    if hsic_xx < 1e-10 or hsic_yy < 1e-10:
        return 0.0
    return float((hsic_xy / (hsic_xx * hsic_yy).sqrt()).item())


# ===== CCA (Canonical Correlation Analysis) =====

def compute_cca(X: torch.Tensor, Y: torch.Tensor, k: int = 10) -> Dict:
    """计算CCA相关系数"""
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    n = X.size(0)
    dim_x, dim_y = X.size(1), Y.size(1)

    cov_xx = (X.T @ X) / (n - 1)
    cov_yy = (Y.T @ Y) / (n - 1)
    cov_xy = (X.T @ Y) / (n - 1)

    # Regularize
    reg = 1e-6
    cov_xx += reg * torch.eye(dim_x, device=X.device)
    cov_yy += reg * torch.eye(dim_y, device=Y.device)

    try:
        L_x = torch.linalg.cholesky(cov_xx)
        L_y = torch.linalg.cholesky(cov_yy)
        L_x_inv = torch.inverse(L_x)
        L_y_inv = torch.inverse(L_y)
        T = L_x_inv @ cov_xy @ L_y_inv.T
        svd_vals = torch.linalg.svdvals(T)
        mean_corr = float(svd_vals[:k].mean().item())
        all_corr = svd_vals.tolist()
    except Exception:
        mean_corr = 0.0
        all_corr = []

    return {
        "mean_cca_correlation": mean_corr,
        "top_k_correlations": all_corr[:k],
    }


# ===== Modality Matching =====

def run_modality_matching_experiment(
    dataset,
    model_name: str = "gcn",
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.5,
    lr: float = 0.001,
    num_epochs: int = 300,
    num_runs: int = 3,
    device: str = "cuda",
    output_dir: str = "./benchmark/results/modality_tasks",
) -> Dict:
    """
    模态匹配实验：给定两个模态特征，判断是否来自同一节点
    """
    from benchmark.models.gnn import get_model
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

    os.makedirs(output_dir, exist_ok=True)

    data = dataset.data
    # 加载各模态特征
    text_embs = _load_single_modality(dataset, "text")
    image_embs = _load_single_modality(dataset, "image")
    formula_embs = _load_single_modality(dataset, "formula")

    modality_pairs = [
        ("text", "image", text_embs, image_embs),
        ("text", "formula", text_embs, formula_embs),
        ("image", "formula", image_embs, formula_embs),
    ]

    all_results = {}

    for mod_a_name, mod_b_name, embs_a, embs_b in modality_pairs:
        key = f"modmatch_{mod_a_name}_{mod_b_name}"
        logger.info(f"\n--- Modality Matching: {mod_a_name} vs {mod_b_name} ---")

        n = embs_a.size(0)
        # 生成正负样本对
        pos_pairs = torch.arange(n)
        neg_pairs = torch.randperm(n)
        # 确保负样本不是自己
        while (neg_pairs == pos_pairs).any():
            mask = neg_pairs == pos_pairs
            neg_pairs[mask] = torch.randperm(n)[:mask.sum()]

        labels = torch.cat([torch.ones(n), torch.zeros(n)])
        pair_features_a = torch.cat([embs_a, embs_a])
        pair_features_b = torch.cat([embs_b[pos_pairs], embs_b[neg_pairs]])

        # 划分
        rng = np.random.RandomState(42)
        perm = rng.permutation(len(labels))
        train_end = int(len(perm) * 0.6)
        val_end = int(len(perm) * 0.8)

        train_idx = perm[:train_end]
        val_idx = perm[train_end:val_end]
        test_idx = perm[val_end:]

        run_results = []
        for run in range(num_runs):
            torch.manual_seed(run * 42 + 1)

            # MLP classifier on concatenated + difference features
            dim = embs_a.size(1)
            clf = nn.Sequential(
                nn.Linear(dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            ).to(device)

            opt = torch.optim.Adam(clf.parameters(), lr=lr)

            train_a = pair_features_a[train_idx].to(device)
            train_b = pair_features_b[train_idx].to(device)
            train_y = labels[train_idx].to(device)
            val_a = pair_features_a[val_idx].to(device)
            val_b = pair_features_b[val_idx].to(device)
            val_y = labels[val_idx].to(device)
            test_a = pair_features_a[test_idx].to(device)
            test_b = pair_features_b[test_idx].to(device)
            test_y = labels[test_idx].to(device)

            best_val, best_test = 0, {}
            patience_counter = 0

            for epoch in range(1, num_epochs + 1):
                clf.train()
                opt.zero_grad()

                x_train = torch.cat([train_a, train_b, train_a - train_b], dim=1)
                out = clf(x_train).squeeze()
                loss = F.binary_cross_entropy_with_logits(out, train_y)
                loss.backward()
                opt.step()

                if epoch % 10 == 0:
                    clf.eval()
                    with torch.no_grad():
                        x_val = torch.cat([val_a, val_b, val_a - val_b], dim=1)
                        pred_val = (clf(x_val).squeeze() > 0).float()
                        val_acc = (pred_val == val_y).float().mean().item()

                        x_test = torch.cat([test_a, test_b, test_a - test_b], dim=1)
                        pred_test = torch.sigmoid(clf(x_test).squeeze())
                        pred_binary = (pred_test > 0.5).float()
                        test_acc = (pred_binary == test_y).float().mean().item()
                        test_f1 = f1_score(test_y.cpu(), pred_binary.cpu(), average='macro')
                        try:
                            test_auc = roc_auc_score(test_y.cpu().numpy(), pred_test.cpu().numpy())
                        except Exception:
                            test_auc = 0.0

                        if val_acc > best_val:
                            best_val = val_acc
                            best_test = {"accuracy": test_acc, "f1": test_f1, "auc": test_auc}
                            patience_counter = 0
                        else:
                            patience_counter += 10
                        if patience_counter >= 50:
                            break

            run_results.append(best_test)
            logger.info(f"  Run {run+1}: acc={best_test['accuracy']:.4f}, "
                        f"f1={best_test['f1']:.4f}, auc={best_test['auc']:.4f}")

        summary = {
            "modality_a": mod_a_name, "modality_b": mod_b_name,
            "task": "modality_matching",
        }
        for metric in ["accuracy", "f1", "auc"]:
            vals = [r[metric] for r in run_results]
            summary[f"{metric}_mean"] = float(np.mean(vals))
            summary[f"{metric}_std"] = float(np.std(vals))
        summary["num_runs"] = num_runs
        summary["runs"] = run_results
        all_results[key] = summary

    # 保存
    save_path = os.path.join(output_dir, "modality_matching_results.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"模态匹配结果保存到 {save_path}")
    return all_results


# ===== Modality Alignment =====

def run_modality_alignment_experiment(
    dataset,
    output_dir: str = "./benchmark/results/modality_tasks",
) -> Dict:
    """
    模态对齐实验：CKA/CCA衡量不同模态嵌入间的对齐程度
    """
    os.makedirs(output_dir, exist_ok=True)

    text_embs = _load_single_modality(dataset, "text")
    image_embs = _load_single_modality(dataset, "image")
    formula_embs = _load_single_modality(dataset, "formula")

    # 过滤有图像的节点
    valid = (image_embs.norm(dim=1) > 1e-6) & (text_embs.norm(dim=1) > 1e-6) & (formula_embs.norm(dim=1) > 1e-6)
    idx = valid.nonzero(as_tuple=True)[0]
    logger.info(f"有效节点: {len(idx)}")

    text_e = text_embs[idx]
    image_e = image_embs[idx]
    formula_e = formula_embs[idx]

    pairs = [
        ("text", "image", text_e, image_e),
        ("text", "formula", text_e, formula_e),
        ("image", "formula", image_e, formula_e),
    ]

    results = {}
    for name_a, name_b, embs_a, embs_b in pairs:
        key = f"align_{name_a}_{name_b}"
        logger.info(f"计算 {name_a}↔{name_b} 对齐度...")

        cka_score = linear_cka(embs_a, embs_b)
        cca_result = compute_cca(embs_a, embs_b, k=10)

        # 余弦相似度统计
        cos_sim = F.cosine_similarity(embs_a, embs_b, dim=1)

        results[key] = {
            "modality_a": name_a,
            "modality_b": name_b,
            "task": "modality_alignment",
            "cka": cka_score,
            "mean_cca": cca_result["mean_cca_correlation"],
            "mean_cosine_similarity": float(cos_sim.mean().item()),
            "std_cosine_similarity": float(cos_sim.std().item()),
            "num_valid_nodes": len(idx),
        }
        logger.info(f"  CKA={cka_score:.4f}, CCA={cca_result['mean_cca_correlation']:.4f}, "
                     f"CosSim={cos_sim.mean():.4f}")

    save_path = os.path.join(output_dir, "modality_alignment_results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"模态对齐结果保存到 {save_path}")
    return results


# ===== Helper =====

def _load_single_modality(dataset, modality: str) -> torch.Tensor:
    """加载单个模态的嵌入"""
    from benchmark.data.dataset import AMMAGDataset
    ds = AMMAGDataset(
        root=dataset.root,
        domains=dataset.domains,
        modality=modality,
        label_type=dataset.label_type,
    )
    return ds.data.x.clone()
