"""
训练集比例学习曲线实验 - 标准benchmark实验
在1%, 5%, 10%, 20%, 40%, 60%, 80%, 100%训练集比例下评估性能
"""
import os
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score


def run_learning_curve(
    dataset,
    model_name: str = "graphsage",
    modality: str = "text+image+formula",
    train_ratios: List[float] = None,
    num_runs: int = 5,
    num_epochs: int = 500,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.5,
    lr: float = 0.01,
    device: str = "cuda",
    output_dir: str = "./benchmark/results/learning_curve",
) -> Dict:
    """
    训练集比例学习曲线：评估不同训练集大小下的模型性能
    """
    from benchmark.models.gnn import get_model

    if train_ratios is None:
        train_ratios = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    os.makedirs(output_dir, exist_ok=True)
    data = dataset.data.to(device)
    in_dim = data.x.size(1)
    out_dim = int(data.y.max().item()) + 1
    num_nodes = data.num_nodes

    all_results = {}
    for ratio in train_ratios:
        run_accs = []
        run_f1s = []

        for run in range(num_runs):
            seed = run * 42 + 1
            rng = np.random.RandomState(seed)

            # 按比例划分
            perm = rng.permutation(num_nodes)
            n_train = int(num_nodes * ratio)
            n_val = int(num_nodes * 0.1)
            n_test = num_nodes - n_train - n_val

            if n_train < out_dim:
                n_train = out_dim
                n_val = max(int((num_nodes - n_train) * 0.1), 1)
                n_test = num_nodes - n_train - n_val

            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            train_mask[perm[:n_train]] = True
            val_mask[perm[n_train:n_train + n_val]] = True
            test_mask[perm[n_train + n_val:]] = True

            model = get_model(
                model_name, in_dim, hidden_dim, out_dim,
                num_layers=num_layers, dropout=dropout,
            ).to(device)

            result = _train_and_eval(
                model, data, train_mask, val_mask, test_mask,
                num_epochs=num_epochs, lr=lr, device=device,
            )
            run_accs.append(result["test_acc"])
            run_f1s.append(result["test_f1"])

        summary = {
            "train_ratio": ratio,
            "train_size": n_train,
            "acc_mean": float(np.mean(run_accs)),
            "acc_std": float(np.std(run_accs)),
            "f1_mean": float(np.mean(run_f1s)),
            "f1_std": float(np.std(run_f1s)),
            "num_runs": num_runs,
            "runs_acc": [float(x) for x in run_accs],
            "runs_f1": [float(x) for x in run_f1s],
        }
        all_results[f"ratio_{ratio}"] = summary
        logger.info(f"ratio={ratio:.0%} ({n_train} nodes): "
                    f"acc={summary['acc_mean']*100:.2f}±{summary['acc_std']*100:.2f}%, "
                    f"f1={summary['f1_mean']*100:.2f}±{summary['f1_std']*100:.2f}%")

    all_results["config"] = {
        "model": model_name, "modality": modality,
        "train_ratios": train_ratios, "num_runs": num_runs,
    }

    save_path = os.path.join(output_dir, f"lc_{model_name}_{modality}.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"学习曲线结果保存到 {save_path}")
    return all_results


def run_multi_model_learning_curve(
    dataset,
    model_names: list = None,
    modality: str = "text+image+formula",
    device: str = "cuda",
    output_dir: str = "./benchmark/results/learning_curve",
) -> Dict:
    """多模型学习曲线对比"""
    if model_names is None:
        model_names = ["mlp", "gcn", "gat", "graphsage", "gin", "gt"]

    all_results = {}
    for model_name in model_names:
        logger.info(f"=== 学习曲线: {model_name} ===")
        result = run_learning_curve(
            dataset, model_name=model_name, modality=modality,
            device=device, output_dir=output_dir,
        )
        all_results[model_name] = result

    save_path = os.path.join(output_dir, "multi_model_learning_curve.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    return all_results


def _train_and_eval(model, data, train_mask, val_mask, test_mask,
                    num_epochs=500, lr=0.01, device="cuda", patience=50):
    import torch.nn.functional as F

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                pred = out.argmax(dim=-1)
                val_acc = accuracy_score(data.y[val_mask].cpu(), pred[val_mask].cpu())
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 10
                if patience_counter >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=-1)
        test_acc = accuracy_score(data.y[test_mask].cpu(), pred[test_mask].cpu())
        test_f1 = f1_score(data.y[test_mask].cpu(), pred[test_mask].cpu(), average="macro")

    return {"test_acc": float(test_acc), "test_f1": float(test_f1)}
