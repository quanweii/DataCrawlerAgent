"""
多数据集验证实验 - OpenMAG P0
使用标准图数据集（Cora, CiteSeer, PubMed）验证模型泛化性
将CLIP 768d编码投影到统一空间进行跨数据集对比
"""
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict
from loguru import logger


def load_pyg_dataset(name: str, device: str = "cpu"):
    """加载标准PyG数据集并投影到768d"""
    from torch_geometric.datasets import Planetoid

    if name.lower() == "cora":
        dataset = Planetoid(root="/tmp/Cora", name="Cora")
    elif name.lower() == "citeseer":
        dataset = Planetoid(root="/tmp/CiteSeer", name="CiteSeer")
    elif name.lower() == "pubmed":
        dataset = Planetoid(root="/tmp/PubMed", name="PubMed")
    else:
        raise ValueError(f"Unknown dataset: {name}")

    data = dataset[0]
    in_dim = data.x.size(1)
    num_classes = dataset.num_classes

    # 投影到768d统一空间
    proj = nn.Linear(in_dim, 768)
    nn.init.xavier_normal_(proj.weight)
    with torch.no_grad():
        data.x = proj(data.x)

    logger.info(f"数据集 {name}: {data.num_nodes} nodes, {data.num_edges} edges, "
                f"features: {data.x.shape}, classes: {num_classes}")

    return data, num_classes


def run_cross_dataset_experiment(
    model_names: list = None,
    dataset_names: list = None,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.5,
    lr: float = 0.001,
    num_epochs: int = 500,
    num_runs: int = 5,
    device: str = "cuda",
    output_dir: str = "./benchmark/results/cross_dataset",
) -> Dict:
    """
    在多个标准数据集上运行模型对比实验
    所有数据集特征统一投影到768d
    """
    from benchmark.models.gnn import get_model

    if model_names is None:
        model_names = ["mlp", "gcn", "gat", "graphsage", "gin", "revgat"]
    if dataset_names is None:
        dataset_names = ["cora"]

    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    for ds_name in dataset_names:
        logger.info(f"\n{'='*60}")
        logger.info(f"数据集: {ds_name}")
        logger.info(f"{'='*60}")

        data, num_classes = load_pyg_dataset(ds_name, device)
        data = data.to(device)
        in_dim = data.x.size(1)  # 768

        for model_name in model_names:
            key = f"{ds_name}_{model_name}"
            run_results = []

            for run in range(num_runs):
                seed = run * 42 + 1
                torch.manual_seed(seed)
                np.random.seed(seed)

                model = get_model(model_name, in_dim, hidden_dim, num_classes,
                                  num_layers=num_layers, dropout=dropout).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

                best_val_acc = 0
                best_test_acc = 0
                patience_counter = 0

                for epoch in range(1, num_epochs + 1):
                    model.train()
                    optimizer.zero_grad()
                    out = model(data.x, data.edge_index)
                    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
                    loss.backward()
                    optimizer.step()

                    if epoch % 10 == 0:
                        model.eval()
                        with torch.no_grad():
                            pred = model(data.x, data.edge_index).argmax(dim=1)
                            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
                            test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
                            if val_acc > best_val_acc:
                                best_val_acc = val_acc
                                best_test_acc = test_acc
                                patience_counter = 0
                            else:
                                patience_counter += 10
                            if patience_counter >= 50:
                                break

                run_results.append({"test_accuracy": best_test_acc})
                logger.info(f"  [Run {run+1}] {model_name}: {best_test_acc:.4f}")

            accs = [r["test_accuracy"] for r in run_results]
            summary = {
                "dataset": ds_name,
                "model": model_name,
                "feature_dim": 768,
                "test_accuracy_mean": float(np.mean(accs)),
                "test_accuracy_std": float(np.std(accs)),
                "num_runs": num_runs,
                "runs": run_results,
            }
            all_results[key] = summary
            logger.info(f"  {key}: {np.mean(accs):.4f}±{np.std(accs):.4f}")

    # 保存结果
    save_path = os.path.join(output_dir, "cross_dataset_results.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"跨数据集结果保存到 {save_path}")

    # 打印对比表
    logger.info(f"\n{'='*70}")
    logger.info("跨数据集对比结果 (Test Accuracy)")
    logger.info(f"{'='*70}")
    header = f"{'Dataset':<12}" + "".join(f"{m:<16}" for m in model_names)
    logger.info(header)
    for ds_name in dataset_names:
        row = f"{ds_name:<12}"
        for model_name in model_names:
            key = f"{ds_name}_{model_name}"
            if key in all_results:
                r = all_results[key]
                row += f"{r['test_accuracy_mean']:.4f}±{r['test_accuracy_std']:.3f}  "
            else:
                row += f"{'N/A':<16}"
        logger.info(row)

    return all_results
