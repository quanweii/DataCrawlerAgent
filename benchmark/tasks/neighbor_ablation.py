"""
邻居数消融实验 - OpenMAG Q2
验证图结构对多模态学习的影响
通过控制每个节点的最大邻居数量，观察性能变化
"""
import torch
import torch.nn.functional as F
import numpy as np
from loguru import logger


def truncate_neighbors(edge_index, num_nodes, max_neighbors):
    """
    限制每个节点的最大邻居数量

    Args:
        edge_index: 原始边索引
        num_nodes: 节点数
        max_neighbors: 每个节点保留的最大邻居数

    Returns:
        截断后的edge_index
    """
    if max_neighbors <= 0:
        # 无图结构，返回空边
        return torch.zeros((2, 0), dtype=edge_index.dtype, device=edge_index.device)

    # 构建邻接表
    from collections import defaultdict
    adj = defaultdict(list)
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        adj[src].append(dst)

    # 截断
    new_edges = []
    for src, neighbors in adj.items():
        if len(neighbors) > max_neighbors:
            # 随机采样
            indices = np.random.choice(len(neighbors), max_neighbors, replace=False)
            neighbors = [neighbors[i] for i in indices]
        for dst in neighbors:
            new_edges.append([src, dst])

    if not new_edges:
        return edge_index

    return torch.tensor(new_edges, dtype=edge_index.dtype, device=edge_index.device).t()


def run_neighbor_ablation(
    model_cls, model_params, data, device="cuda",
    neighbor_counts=None, num_runs=3, epochs=500, lr=0.001, patience=25,
):
    """
    运行邻居数消融实验

    Args:
        model_cls: 模型类 (当K=0时自动用MLP)
        model_params: 模型参数dict
        data: PyG Data
        neighbor_counts: 邻居数列表 [0, 2, 4, 6, 8]
        num_runs: 运行次数

    Returns:
        dict mapping neighbor_count -> {mean, std} of accuracy
    """
    if neighbor_counts is None:
        neighbor_counts = [0, 2, 4, 6, 8]

    results = {}
    num_classes = data.y.max().item() + 1

    for k in neighbor_counts:
        accs = []

        for run in range(num_runs):
            seed = run * 42 + 1
            torch.manual_seed(seed)
            np.random.seed(seed)

            # 构建截断图
            if k == 0:
                truncated_edges = torch.zeros((2, 0), dtype=data.edge_index.dtype)
            else:
                truncated_edges = truncate_neighbors(
                    data.edge_index, data.num_nodes, k
                )

            # 创建数据副本
            data_k = data.clone()
            data_k.edge_index = truncated_edges

            # 当K=0时，没有边，GNN退化为MLP
            model = model_cls(**model_params).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = torch.nn.CrossEntropyLoss()

            x = data_k.x.to(device)
            edge_index = data_k.edge_index.to(device)
            y = data_k.y.to(device)
            train_mask = data_k.train_mask.to(device)
            val_mask = data_k.val_mask.to(device)
            test_mask = data_k.test_mask.to(device)

            best_val_acc = 0
            best_test_acc = 0
            p_counter = 0

            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                out = model(x, edge_index)
                loss = criterion(out[train_mask], y[train_mask])
                loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    out = model(x, edge_index)
                    pred = out.argmax(dim=1)
                    val_acc = (pred[val_mask] == y[val_mask]).float().mean().item()
                    test_acc = (pred[test_mask] == y[test_mask]).float().mean().item()

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    p_counter = 0
                else:
                    p_counter += 1

                if p_counter >= patience:
                    break

            accs.append(best_test_acc)

        result = {
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs)),
        }
        results[f"k_{k}"] = result
        logger.info(f"Neighbor K={k}: acc={result['mean']:.4f}±{result['std']:.4f}")

    return {
        "experiment": "neighbor_ablation",
        "neighbor_counts": neighbor_counts,
        "results": results,
    }
