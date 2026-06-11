"""
注意力融合对比实验

对比三种融合策略：
1. 简单拼接 (Concat) - 基线
2. 可学习加权求和 (Attention Fusion) - 先投影到统一维度再加权
3. 各模态独立投影后拼接 (Projected Concat) - 投影后拼接

实验设置：GraphSAGE + Text+Image+Formula, 5次运行
"""
import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


class ProjectedConcat(nn.Module):
    """各模态投影到统一维度后拼接"""
    def __init__(self, modality_dims, proj_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.projections = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(dim, proj_dim), nn.ReLU())
            for name, dim in modality_dims.items()
        })
        fused_dim = proj_dim * len(modality_dims)

        from torch_geometric.nn import SAGEConv
        self.convs = nn.ModuleList([
            SAGEConv(fused_dim, hidden_dim),
            SAGEConv(hidden_dim, out_dim),
        ])
        self.dropout = dropout

    def forward(self, modality_embs, edge_index):
        projected = [self.projections[name](emb) for name, emb in modality_embs.items()]
        x = torch.cat(projected, dim=1)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class AttentionFusion(nn.Module):
    """可学习模态权重融合"""
    def __init__(self, modality_dims, proj_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.proj_dim = proj_dim
        n_mods = len(modality_dims)

        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, proj_dim)
            for name, dim in modality_dims.items()
        })
        # 可学习标量权重
        self.mod_weights = nn.Parameter(torch.ones(n_mods))

        from torch_geometric.nn import SAGEConv
        self.convs = nn.ModuleList([
            SAGEConv(proj_dim, hidden_dim),
            SAGEConv(hidden_dim, out_dim),
        ])
        self.dropout = dropout

    def forward(self, modality_embs, edge_index):
        weights = F.softmax(self.mod_weights, dim=0)
        projected = []
        for i, (name, emb) in enumerate(modality_embs.items()):
            projected.append(weights[i] * self.projections[name](emb))

        x = sum(projected)  # 加权求和，输出 proj_dim 维
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

    def get_weights(self):
        return F.softmax(self.mod_weights, dim=0).detach().cpu()


class ConcatBaseline(nn.Module):
    """简单拼接基线（重实现用于公平对比）"""
    def __init__(self, modality_dims, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        fused_dim = sum(modality_dims.values())

        from torch_geometric.nn import SAGEConv
        self.convs = nn.ModuleList([
            SAGEConv(fused_dim, hidden_dim),
            SAGEConv(hidden_dim, out_dim),
        ])
        self.dropout = dropout

    def forward(self, modality_embs, edge_index):
        x = torch.cat(list(modality_embs.values()), dim=1)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


def train_and_eval(model, modality_embs, edge_index, labels, train_mask, val_mask, test_mask,
                   num_epochs=500, lr=0.01, device="cuda"):
    model = model.to(device)
    for k in modality_embs:
        modality_embs[k] = modality_embs[k].to(device)
    edge_index = edge_index.to(device)
    labels = labels.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(modality_embs, edge_index)
        loss = criterion(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                out = model(modality_embs, edge_index)
                pred = out.argmax(dim=-1)
                val_acc = accuracy_score(labels[val_mask].cpu(), pred[val_mask].cpu())
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 10
                if patience_counter >= 50:
                    break

    # 测试
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(modality_embs, edge_index)
        pred = out.argmax(dim=-1)
        test_acc = accuracy_score(labels[test_mask].cpu(), pred[test_mask].cpu())
        test_f1 = f1_score(labels[test_mask].cpu(), pred[test_mask].cpu(), average="macro")

    return {"test_acc": float(test_acc), "test_f1": float(test_f1), "best_val_acc": float(best_val_acc)}


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(project_root, "processed")
    output_dir = os.path.join(project_root, "results", "fusion")
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    graph_data = torch.load(os.path.join(processed_dir, "graph_data.pt"), map_location="cpu")
    text_emb = torch.load(os.path.join(processed_dir, "text_embeddings.pt"), map_location="cpu")
    image_emb = torch.load(os.path.join(processed_dir, "image_embeddings.pt"), map_location="cpu")
    formula_emb = torch.load(os.path.join(processed_dir, "formula_embeddings.pt"), map_location="cpu")

    # 对齐
    paper_ids = graph_data["paper_ids"]
    text_id_map = {pid: i for i, pid in enumerate(text_emb["ids"])}
    image_id_map = {pid: i for i, pid in enumerate(image_emb["ids"])}
    formula_id_map = {pid: i for i, pid in enumerate(formula_emb["ids"])}

    indices = [i for i, pid in enumerate(paper_ids)
               if pid in text_id_map and pid in image_id_map and pid in formula_id_map]
    logger.info(f"对齐论文数: {len(indices)}/{len(paper_ids)}")

    # 构建模态embedding字典
    t_emb = text_emb["embeddings"][torch.tensor([text_id_map[paper_ids[i]] for i in indices])]
    i_emb = image_emb["embeddings"][torch.tensor([image_id_map[paper_ids[i]] for i in indices])]
    f_emb = formula_emb["embeddings"][torch.tensor([formula_id_map[paper_ids[i]] for i in indices])]

    labels = graph_data["labels"][indices]
    num_nodes = len(indices)
    num_classes = int(labels.max().item()) + 1

    # 重映射edge_index
    old_to_new = {}
    for new_idx, old_idx in enumerate(indices):
        old_to_new[old_idx] = new_idx
    edge_index = graph_data["edge_index"]
    mask = torch.tensor([(s.item() in old_to_new and d.item() in old_to_new)
                          for s, d in zip(edge_index[0], edge_index[1])])
    ei = edge_index[:, mask]
    new_ei = torch.tensor([[old_to_new[s.item()], old_to_new[d.item()]]
                            for s, d in zip(ei[0], ei[1])], dtype=torch.long).t()
    logger.info(f"图: {num_nodes}节点, {new_ei.size(1)}边, {num_classes}类")

    modality_dims = {"text": 384, "image": 512, "formula": 384}
    num_runs = 5
    all_results = {}

    # 三种融合方法
    methods = {
        "concat": lambda: ConcatBaseline(modality_dims, 256, num_classes),
        "attention": lambda: AttentionFusion(modality_dims, 256, 256, num_classes),
        "projected_concat": lambda: ProjectedConcat(modality_dims, 256, 256, num_classes),
    }

    for method_name, model_fn in methods.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"融合方法: {method_name}")
        logger.info(f"{'='*60}")

        run_results = []
        learned_weights_list = []

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
            val_mask[perm[t:t + v]] = True
            test_mask[perm[t + v:]] = True

            modality_embs = {"text": t_emb.clone(), "image": i_emb.clone(), "formula": f_emb.clone()}
            model = model_fn()

            result = train_and_eval(
                model, modality_embs, new_ei, labels,
                train_mask, val_mask, test_mask,
                device=device,
            )
            run_results.append(result)

            # 记录学到的模态权重
            if method_name == "attention" and hasattr(model, 'get_weights'):
                w = model.get_weights()
                learned_weights_list.append({
                    "text": float(w[0]),
                    "image": float(w[1]),
                    "formula": float(w[2]),
                })

            logger.info(f"  Run {run+1}/{num_runs}: acc={result['test_acc']*100:.2f}%, "
                        f"f1={result['test_f1']*100:.2f}%")

        accs = [r["test_acc"] for r in run_results]
        f1s = [r["test_f1"] for r in run_results]

        summary = {
            "method": method_name,
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
            "num_runs": num_runs,
            "runs": run_results,
        }
        if learned_weights_list:
            summary["learned_weights"] = learned_weights_list

        all_results[method_name] = summary
        logger.info(f"  >> {method_name}: {np.mean(accs)*100:.2f}±{np.std(accs)*100:.2f}%")

    # 保存
    with open(os.path.join(output_dir, "fusion_comparison.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # 打印对比表
    print("\n" + "=" * 70)
    print("融合方法对比结果 (GraphSAGE + T+I+F, 5 runs)")
    print("=" * 70)
    print(f"{'方法':<25} {'Accuracy':>15} {'Macro-F1':>15}")
    print("-" * 55)
    for method, s in all_results.items():
        print(f"{method:<25} {s['acc_mean']*100:>7.2f}±{s['acc_std']*100:.2f}% "
              f"{s['f1_mean']*100:>7.2f}±{s['f1_std']*100:.2f}%")

    if "attention" in all_results and "learned_weights" in all_results["attention"]:
        weights = all_results["attention"]["learned_weights"]
        print(f"\n注意力融合学到的模态权重 (5次运行平均):")
        for mod in ["text", "image", "formula"]:
            mean_w = np.mean([w[mod] for w in weights])
            std_w = np.std([w[mod] for w in weights])
            print(f"  {mod}: {mean_w:.3f} ± {std_w:.3f}")

    logger.info(f"\n结果保存到 {output_dir}/fusion_comparison.json")


if __name__ == "__main__":
    main()
