"""
Few-shot学习曲线实验 - 补充1/5/20-shot
绘制完整的 shot数 vs Accuracy 学习曲线
"""
import os
import json
import torch
import numpy as np
from typing import Dict
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score


def run_few_shot_curve(
    dataset,
    model_names: list = None,
    modalities: list = None,
    shot_list: list = None,
    num_runs: int = 10,
    num_epochs: int = 500,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.5,
    lr: float = 0.01,
    device: str = "cuda",
    output_dir: str = "./benchmark/results/fewshot_curve",
) -> Dict:
    if model_names is None:
        model_names = ["gcn", "gat", "graphsage", "gin", "gt"]
    if modalities is None:
        modalities = ["text", "text+formula", "text+image+formula"]
    if shot_list is None:
        shot_list = [1, 3, 5, 10, 20]

    from benchmark.models.gnn import get_model

    os.makedirs(output_dir, exist_ok=True)
    data = dataset.data.to(device)
    in_dim = data.x.size(1)
    out_dim = int(data.y.max().item()) + 1

    all_results = {}
    for model_name in model_names:
        for modality in modalities:
            for k in shot_list:
                key = f"{model_name}_{modality}_k{k}"
                run_accs = []
                run_f1s = []

                for run in range(num_runs):
                    seed = run * 42 + 1
                    train_mask, val_mask, test_mask = dataset.get_split_masks(
                        split="few_shot", seed=seed, few_shot_k=k
                    )

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
                    "model": model_name,
                    "modality": modality,
                    "k": k,
                    "acc_mean": float(np.mean(run_accs)),
                    "acc_std": float(np.std(run_accs)),
                    "f1_mean": float(np.mean(run_f1s)),
                    "f1_std": float(np.std(run_f1s)),
                    "num_runs": num_runs,
                }
                all_results[key] = summary
                logger.info(f"{key}: acc={summary['acc_mean']*100:.2f}±{summary['acc_std']*100:.2f}%, "
                            f"f1={summary['f1_mean']*100:.2f}±{summary['f1_std']*100:.2f}%")

    save_path = os.path.join(output_dir, "fewshot_curve_results.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Few-shot曲线结果保存到 {save_path}")
    return all_results


def _train_and_eval(model, data, train_mask, val_mask, test_mask,
                    num_epochs=500, lr=0.01, device="cuda", patience=50):
    import torch.nn as nn
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
