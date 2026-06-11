"""
噪声鲁棒性实验 - OpenMAG Q7-Q8
文本噪声、图像噪声、标签噪声、边噪声
"""
import torch
import numpy as np
from loguru import logger


def inject_feature_noise(features: torch.Tensor, noise_ratio: float,
                         noise_type: str = "gaussian") -> torch.Tensor:
    """对特征注入噪声

    Args:
        features: 节点特征矩阵 (N, D)
        noise_ratio: 噪声比例 [0, 1]
        noise_type: "gaussian" 或 "shuffle"
    """
    if noise_ratio == 0.0:
        return features

    noisy = features.clone()
    n_nodes = features.size(0)
    n_noisy = int(n_nodes * noise_ratio)
    noisy_indices = np.random.choice(n_nodes, n_noisy, replace=False)

    if noise_type == "gaussian":
        # 用高斯噪声替换
        mean = features.mean(dim=0)
        std = features.std(dim=0) + 1e-8
        noise = torch.randn(n_noisy, features.size(1)) * std + mean
        noisy[noisy_indices] = noise
    elif noise_type == "shuffle":
        # 随机打乱（切断语义对齐）
        perm = np.random.permutation(n_nodes)
        noisy[noisy_indices] = features[perm[:n_noisy]]
    elif noise_type == "zero":
        noisy[noisy_indices] = 0.0

    return noisy


def inject_label_noise(labels: torch.Tensor, noise_ratio: float,
                       num_classes: int) -> torch.Tensor:
    """对标签注入噪声，随机翻转为错误标签"""
    if noise_ratio == 0.0:
        return labels

    noisy = labels.clone()
    n_nodes = labels.size(0)
    n_noisy = int(n_nodes * noise_ratio)
    noisy_indices = np.random.choice(n_nodes, n_noisy, replace=False)

    for idx in noisy_indices:
        original = labels[idx].item()
        wrong_labels = [c for c in range(num_classes) if c != original]
        noisy[idx] = np.random.choice(wrong_labels)

    return noisy


def inject_edge_noise(edge_index: torch.Tensor, num_nodes: int,
                      noise_ratio: float) -> torch.Tensor:
    """添加随机虚假边"""
    if noise_ratio == 0.0:
        return edge_index

    n_edges = edge_index.size(1)
    n_new = int(n_edges * noise_ratio)

    existing = set(
        (edge_index[0, i].item(), edge_index[1, i].item())
        for i in range(n_edges)
    )

    new_edges = []
    while len(new_edges) < n_new:
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)
        if src != dst and (src, dst) not in existing:
            new_edges.append((src, dst))
            existing.add((src, dst))

    if not new_edges:
        return edge_index

    new_edge_index = torch.tensor(new_edges, dtype=edge_index.dtype).t()
    return torch.cat([edge_index, new_edge_index], dim=1)


def run_noise_experiment(
    task_fn, data, model_cls, model_params,
    noise_type: str, noise_ratios=None, num_runs=3, **kwargs
):
    """
    运行噪声实验

    Args:
        task_fn: 训练评估函数 (model, data, **kwargs) -> dict
        data: PyG Data对象
        model_cls: 模型类
        model_params: 模型参数dict
        noise_type: "text", "image", "label", "edge"
        noise_ratios: 噪声比例列表
        num_runs: 运行次数

    Returns:
        dict mapping noise_ratio -> {mean, std} of accuracy
    """
    if noise_ratios is None:
        noise_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    results = {}
    num_classes = kwargs.get("num_classes", data.y.max().item() + 1)

    for ratio in noise_ratios:
        accs = []
        for run in range(num_runs):
            seed = run * 42 + 1
            torch.manual_seed(seed)
            np.random.seed(seed)

            # 创建噪声数据副本
            noisy_data = data.clone()

            if noise_type == "text":
                # 对文本特征部分注入噪声，保留其余模态
                text_dim = kwargs.get("text_dim", 768)
                noisy_x = noisy_data.x.clone()
                noisy_x[:, :text_dim] = inject_feature_noise(
                    noisy_data.x[:, :text_dim], ratio, "shuffle"
                )
                noisy_data.x = noisy_x
            elif noise_type == "image":
                image_start = kwargs.get("image_start", 768)
                image_end = kwargs.get("image_end", 768 * 2)
                noisy_data.x = torch.cat([
                    noisy_data.x[:, :image_start],
                    inject_feature_noise(noisy_data.x[:, image_start:image_end], ratio, "shuffle"),
                    noisy_data.x[:, image_end:]
                ], dim=1)
            elif noise_type == "label":
                train_mask = noisy_data.train_mask
                noisy_data.y = noisy_data.y.clone()
                noisy_data.y[train_mask] = inject_label_noise(
                    noisy_data.y[train_mask], ratio, num_classes
                )
            elif noise_type == "edge":
                noisy_data.edge_index = inject_edge_noise(
                    noisy_data.edge_index, noisy_data.num_nodes, ratio
                )

            model = model_cls(**model_params)
            # Filter kwargs to only include parameters accepted by task_fn
            task_kwargs = {k: v for k, v in kwargs.items()
                           if k in ("device", "epochs", "lr", "patience", "weight_decay")}
            res = task_fn(model, noisy_data, **task_kwargs)
            accs.append(res.get("test_acc", 0.0))

        results[f"ratio_{ratio:.1f}"] = {
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs)),
        }
        logger.info(f"Noise {noise_type} ratio={ratio:.1f}: "
                     f"acc={np.mean(accs):.4f}±{np.std(accs):.4f}")

    return {
        "noise_type": noise_type,
        "noise_ratios": noise_ratios,
        "results": results,
    }
