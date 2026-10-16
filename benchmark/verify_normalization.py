#!/usr/bin/env python3
"""
快速验证：Image特征L2归一化对多模态性能的影响
对比实验：
  A) 原始特征（Image未归一化）
  B) Image L2归一化后的特征
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def l2_normalize(embeddings: torch.Tensor) -> torch.Tensor:
    """L2归一化，零向量保持为零"""
    norms = embeddings.norm(dim=1, keepdim=True)
    norms = norms.clamp(min=1e-8)
    return embeddings / norms


def load_embeddings(processed_dir, modality, normalize_image=False):
    """加载并拼接特征"""
    parts = modality.split("+")
    embeddings_list = []

    # 加载paper_ids用于对齐
    graph = torch.load(os.path.join(processed_dir, "graph_data.pt"), map_location="cpu")
    paper_ids = graph["paper_ids"]

    for mod in parts:
        path = os.path.join(processed_dir, f"{mod}_embeddings.pt")
        data = torch.load(path, map_location="cpu")
        emb_ids = data["ids"]
        emb_tensor = data["embeddings"]

        id_to_idx = {pid: i for i, pid in enumerate(emb_ids)}
        dim = emb_tensor.size(1)
        aligned = torch.zeros(len(paper_ids), dim)
        for j, pid in enumerate(paper_ids):
            if pid in id_to_idx:
                aligned[j] = emb_tensor[id_to_idx[pid]]

        # 只对image做L2归一化
        if mod == "image" and normalize_image:
            aligned = l2_normalize(aligned)

        embeddings_list.append(aligned)

    return torch.cat(embeddings_list, dim=1)


def train_gcn(x, edge_index, y, train_mask, val_mask, test_mask,
              hidden_dim=256, lr=0.01, weight_decay=5e-4, epochs=500, device="cuda"):
    """快速训练GCN"""
    from torch_geometric.nn import GCNConv

    class GCN(nn.Module):
        def __init__(self, in_dim, h, out_dim):
            super().__init__()
            self.conv1 = GCNConv(in_dim, h)
            self.conv2 = GCNConv(h, out_dim)
        def forward(self, x, ei):
            x = self.conv1(x, ei).relu()
            x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
            return self.conv2(x, ei)

    in_dim = x.size(1)
    out_dim = int(y.max()) + 1
    model = GCN(in_dim, hidden_dim, out_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    x, edge_index, y = x.to(device), edge_index.to(device), y.to(device)
    train_mask, val_mask, test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)

    best_val_acc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(x, edge_index).argmax(dim=-1)
                val_acc = accuracy_score(y[val_mask].cpu(), pred[val_mask].cpu())
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 10
                if patience_counter >= 50:
                    break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(x, edge_index).argmax(dim=-1)
        test_acc = accuracy_score(y[test_mask].cpu(), pred[test_mask].cpu())

    return test_acc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processed_dir = os.path.join(_project_root, "benchmark", "processed")

    # 加载图结构
    graph = torch.load(os.path.join(processed_dir, "graph_data.pt"), map_location="cpu")
    edge_index = graph["edge_index"]
    y = graph["labels"]
    num_nodes = len(graph["paper_ids"])

    # 测试的模态组合
    combos = ["text", "text+image", "text+formula", "text+image+formula"]
    num_runs = 5

    print("=" * 80)
    print("  验证实验：Image L2归一化对多模态性能的影响")
    print("  GCN, 5次运行, 60/20/20 split")
    print("=" * 80)
    print(f"{'模态':<20} {'原始Acc':>12} {'归一化Acc':>12} {'差异(pp)':>12}")
    print("-" * 80)

    for modality in combos:
        results_orig = []
        results_norm = []

        for run in range(num_runs):
            seed = run * 42 + 1
            rng = np.random.RandomState(seed)
            perm = rng.permutation(num_nodes)
            t = int(num_nodes * 0.6)
            v = int(num_nodes * 0.2)

            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            train_mask[perm[:t]] = True
            val_mask[perm[t:t+v]] = True
            test_mask[perm[t+v:]] = True

            # 原始特征
            x_orig = load_embeddings(processed_dir, modality, normalize_image=False)
            acc_orig = train_gcn(x_orig, edge_index, y, train_mask, val_mask, test_mask, device=device)
            results_orig.append(acc_orig)

            # Image归一化后的特征
            x_norm = load_embeddings(processed_dir, modality, normalize_image=True)
            acc_norm = train_gcn(x_norm, edge_index, y, train_mask, val_mask, test_mask, device=device)
            results_norm.append(acc_norm)

        mean_orig = np.mean(results_orig) * 100
        mean_norm = np.mean(results_norm) * 100
        diff = mean_norm - mean_orig

        marker = ""
        if diff > 0.3:
            marker = " ↑"
        elif diff < -0.3:
            marker = " ↓"

        print(f"{modality:<20} {mean_orig:>11.2f}% {mean_norm:>11.2f}% {diff:>+10.2f}pp{marker}")

    print("-" * 80)
    print("说明: ↑ 表示归一化后提升>0.3pp, ↓ 表示下降>0.3pp")
    print()


if __name__ == "__main__":
    main()
