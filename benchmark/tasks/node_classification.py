"""
节点分类任务 - 支持监督学习、Few-shot、per-domain accuracy
"""
import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score


def train_and_eval(
    model: nn.Module,
    data,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    num_epochs: int = 500,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    patience: int = 50,
    device: str = "cuda",
    domain_labels: Optional[torch.Tensor] = None,
) -> Dict:
    """
    训练并评估节点分类模型

    Args:
        domain_labels: 4-class domain labels，用于计算per-domain accuracy

    Returns:
        dict with train/val/test accuracy and f1, best_epoch, per_domain_acc
    """
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_epoch = 0
    best_state = None
    patience_counter = 0

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        # 梯度裁剪：稳定 GIN / GCNII / 高维输入训练，消除偶发崩溃（与 LP 循环一致）。
        # 实测：wikiart GIN+text+image 由 0.47±0.118(崩溃) → 0.78±0.014(稳定)。
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 10 == 0 or epoch == num_epochs:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                pred = out.argmax(dim=-1)

                train_acc = accuracy_score(data.y[train_mask].cpu(), pred[train_mask].cpu())
                val_acc = accuracy_score(data.y[val_mask].cpu(), pred[val_mask].cpu())

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 10

                if patience_counter >= patience:
                    break

    training_time = time.time() - start_time

    # 用最佳模型评估
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=-1)

        test_acc = accuracy_score(data.y[test_mask].cpu(), pred[test_mask].cpu())
        test_f1 = f1_score(data.y[test_mask].cpu(), pred[test_mask].cpu(), average="macro")
        val_f1 = f1_score(data.y[val_mask].cpu(), pred[val_mask].cpu(), average="macro")

    results = {
        "test_acc": float(test_acc),
        "test_f1": float(test_f1),
        "val_acc": float(best_val_acc),
        "val_f1": float(val_f1),
        "train_acc": float(train_acc),
        "best_epoch": best_epoch,
        "training_time": training_time,
    }

    # per-domain accuracy
    if domain_labels is not None:
        from benchmark.data.dataset import AMMAGDataset
        per_domain = AMMAGDataset.compute_per_domain_accuracy(
            pred.cpu(), data.y.cpu(), domain_labels.cpu(), test_mask.cpu()
        )
        results["per_domain_acc"] = per_domain

    return results


def run_node_classification(
    dataset,
    model_name: str,
    modality: str,
    num_runs: int = 5,
    num_epochs: int = 500,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.5,
    lr: float = 0.01,
    device: str = "cuda",
    split: str = "supervised",
    few_shot_k: int = -1,
    output_dir: str = "./benchmark/results",
    domain_labels: Optional[torch.Tensor] = None,
    save_prefix: str = "nc",
) -> Dict:
    """
    运行节点分类实验（多次重复）

    Args:
        domain_labels: 用于per-domain accuracy计算
        save_prefix: 文件名前缀（"nc" or "subcat_nc"）
    """
    from benchmark.models.gnn import get_model

    os.makedirs(output_dir, exist_ok=True)
    data = dataset.data.to(device)
    in_dim = data.x.size(1)
    out_dim = int(data.y.max().item()) + 1

    all_results = []
    for run in range(num_runs):
        seed = run * 42 + 1
        train_mask, val_mask, test_mask = dataset.get_split_masks(
            split=split, seed=seed, few_shot_k=few_shot_k
        )

        model = get_model(
            model_name, in_dim, hidden_dim, out_dim,
            num_layers=num_layers, dropout=dropout,
        )

        result = train_and_eval(
            model, data, train_mask, val_mask, test_mask,
            num_epochs=num_epochs, lr=lr, device=device,
            domain_labels=domain_labels,
        )
        all_results.append(result)
        logger.info(f"[Run {run+1}/{num_runs}] {model_name}+{modality}: "
                     f"test_acc={result['test_acc']:.4f}, test_f1={result['test_f1']:.4f}")

    # 汇总
    metrics = ["test_acc", "test_f1", "val_acc", "training_time"]
    summary = {}
    for m in metrics:
        values = [r[m] for r in all_results]
        summary[f"{m}_mean"] = float(np.mean(values))
        summary[f"{m}_std"] = float(np.std(values))

    # 汇总per-domain accuracy
    if all_results and "per_domain_acc" in all_results[0]:
        from benchmark.data.dataset import AMMAGDataset
        per_domain_summary = {}
        for dname in AMMAGDataset.DOMAIN_KEYS:
            vals = [r["per_domain_acc"].get(dname, 0) for r in all_results if "per_domain_acc" in r]
            if vals:
                per_domain_summary[f"{dname}_acc_mean"] = float(np.mean(vals))
                per_domain_summary[f"{dname}_acc_std"] = float(np.std(vals))
        summary["per_domain_acc"] = per_domain_summary

    summary["num_runs"] = num_runs
    summary["model"] = model_name
    summary["modality"] = modality
    summary["split"] = split
    summary["few_shot_k"] = few_shot_k
    summary["runs"] = all_results

    # 保存
    save_name = f"{save_prefix}_{model_name}_{modality}_{split}"
    if few_shot_k > 0:
        save_name += f"_k{few_shot_k}"
    save_path = os.path.join(output_dir, f"{save_name}.json")
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"结果保存到 {save_path}")
    logger.info(f"总结: test_acc={summary['test_acc_mean']:.4f}±{summary['test_acc_std']:.4f}, "
                 f"test_f1={summary['test_f1_mean']:.4f}±{summary['test_f1_std']:.4f}")

    return summary
