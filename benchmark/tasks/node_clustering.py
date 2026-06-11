"""
节点聚类任务 - OpenMAG Q5
使用GNN提取节点嵌入后做K-Means聚类，计算NMI和ARI
"""
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from loguru import logger


def extract_gnn_embeddings(model, data, device="cuda"):
    """从GNN模型提取输出logits作为节点嵌入"""
    model.eval()
    model = model.to(device)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    with torch.no_grad():
        out = model(x, edge_index)

    return out.cpu().numpy()


def run_node_clustering(
    model, data, num_classes, device="cuda", use_raw_features=False
):
    if use_raw_features:
        features = data.x.numpy()
    else:
        features = extract_gnn_embeddings(model, data, device)

    labels_true = data.y.numpy()
    kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=10)
    labels_pred = kmeans.fit_predict(features)

    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    ari = adjusted_rand_score(labels_true, labels_pred)
    return {"nmi": nmi, "ari": ari}


def run_clustering_experiment(
    model_name, model, data, num_classes, num_runs=3, device="cuda", dataset=None
):
    """多次运行聚类实验：监督训练后提取输出logits做KMeans"""
    results = {"nmi": [], "ari": []}

    for run in range(num_runs):
        import random
        seed = run * 42 + 1
        random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)

        # Generate masks if needed
        if not hasattr(data, 'train_mask') or data.train_mask is None:
            if dataset is not None:
                data.train_mask, data.val_mask, data.test_mask = dataset.get_split_masks(seed=seed)
            else:
                n = data.num_nodes
                perm = np.random.permutation(n)
                tr = int(n * 0.6); va = int(n * 0.2)
                data.train_mask = torch.zeros(n, dtype=torch.bool)
                data.val_mask = torch.zeros(n, dtype=torch.bool)
                data.test_mask = torch.zeros(n, dtype=torch.bool)
                data.train_mask[perm[:tr]] = True
                data.val_mask[perm[tr:tr+va]] = True
                data.test_mask[perm[tr+va:]] = True

        model = _train_model_supervised(model, data, device, epochs=200, lr=0.01)
        res = run_node_clustering(model, data, num_classes, device)
        results["nmi"].append(res["nmi"])
        results["ari"].append(res["ari"])

    return {
        "model": model_name,
        "nmi_mean": float(np.mean(results["nmi"])),
        "nmi_std": float(np.std(results["nmi"])),
        "ari_mean": float(np.mean(results["ari"])),
        "ari_std": float(np.std(results["ari"])),
    }


def _train_model_supervised(model, data, device="cuda", epochs=200, lr=0.01):
    """监督训练模型用于聚类评估"""
    import torch.nn as nn
    model.train()
    model = model.to(device)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        if torch.isnan(loss):
            break
        loss.backward()
        optimizer.step()
    return model
