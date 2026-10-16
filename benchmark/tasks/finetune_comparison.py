"""
编码器微调对比实验 - OpenMAG Q3
对比 Frozen vs Trainable 编码器配置:
- T-Frozen + V-Frozen
- T-Train + V-Frozen
- T-Frozen + V-Train
- T-Train + V-Train
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loguru import logger


class ModalityProjectionHead(nn.Module):
    """可学习的模态投影头，模拟编码器微调"""

    def __init__(self, dim=768, hidden_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        # 初始化为近似恒等映射
        nn.init.xavier_uniform_(self.proj[0].weight, gain=0.01)
        nn.init.xavier_uniform_(self.proj[2].weight, gain=0.01)

    def forward(self, x):
        return x + self.proj(x)  # 残差连接


class FinetuneWrapper(nn.Module):
    """包装GNN模型，添加可选的模态投影头"""

    def __init__(self, gnn_model, text_trainable=False, image_trainable=False,
                 formula_trainable=False, dim=768, proj_hidden=256):
        super().__init__()
        self.gnn = gnn_model
        self.text_trainable = text_trainable
        self.image_trainable = image_trainable
        self.formula_trainable = formula_trainable

        if text_trainable:
            self.text_proj = ModalityProjectionHead(dim, proj_hidden)
        if image_trainable:
            self.image_proj = ModalityProjectionHead(dim, proj_hidden)
        if formula_trainable:
            self.formula_proj = ModalityProjectionHead(dim, proj_hidden)

    def forward(self, x, edge_index):
        return self.gnn(x, edge_index)


def apply_finetune_projections(wrapper, x, text_dim=768, image_dim=768,
                                formula_dim=768, has_image=True, has_formula=True):
    """对特征应用投影头"""
    parts = []
    offset = 0

    # Text
    text_feat = x[:, offset:offset + text_dim]
    if wrapper.text_trainable:
        text_feat = wrapper.text_proj(text_feat)
    parts.append(text_feat)
    offset += text_dim

    # Image
    if has_image:
        img_feat = x[:, offset:offset + image_dim]
        if wrapper.image_trainable:
            img_feat = wrapper.image_proj(img_feat)
        parts.append(img_feat)
        offset += image_dim

    # Formula
    if has_formula:
        form_feat = x[:, offset:offset + formula_dim]
        if wrapper.formula_trainable:
            form_feat = wrapper.formula_proj(form_feat)
        parts.append(form_feat)

    return torch.cat(parts, dim=1)


def run_finetune_comparison(
    model_cls, model_params, data, device="cuda",
    num_runs=3, epochs=500, lr=0.001, patience=25, dim=768
):
    """
    运行四种微调配置对比

    Returns:
        list of results dicts
    """
    configs = [
        {"name": "T-Frozen+V-Frozen", "text": False, "image": False},
        {"name": "T-Train+V-Frozen", "text": True, "image": False},
        {"name": "T-Frozen+V-Train", "text": False, "image": True},
        {"name": "T-Train+V-Train", "text": True, "image": True},
    ]

    all_results = []

    for config in configs:
        accs = []
        f1s = []

        for run in range(num_runs):
            seed = run * 42 + 1
            torch.manual_seed(seed)
            np.random.seed(seed)

            gnn_model = model_cls(**model_params)
            wrapper = FinetuneWrapper(
                gnn_model,
                text_trainable=config["text"],
                image_trainable=config["image"],
                dim=dim,
            ).to(device)

            optimizer = torch.optim.Adam(wrapper.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            y = data.y.to(device)
            train_mask = data.train_mask.to(device)
            val_mask = data.val_mask.to(device)
            test_mask = data.test_mask.to(device)

            best_val_acc = 0
            best_test_acc = 0
            patience_counter = 0

            for epoch in range(epochs):
                wrapper.train()
                optimizer.zero_grad()

                h = apply_finetune_projections(wrapper, x, dim, dim, dim,
                                                has_image=True, has_formula=False)
                out = wrapper.gnn(h, edge_index)
                loss = criterion(out[train_mask], y[train_mask])
                loss.backward()
                optimizer.step()

                # Eval
                wrapper.eval()
                with torch.no_grad():
                    h = apply_finetune_projections(wrapper, x, dim, dim, dim,
                                                    has_image=True, has_formula=False)
                    out = wrapper.gnn(h, edge_index)
                    pred = out.argmax(dim=1)
                    val_acc = (pred[val_mask] == y[val_mask]).float().mean().item()
                    test_acc = (pred[test_mask] == y[test_mask]).float().mean().item()

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break

            accs.append(best_test_acc)

        result = {
            "config": config["name"],
            "test_acc_mean": float(np.mean(accs)),
            "test_acc_std": float(np.std(accs)),
        }
        all_results.append(result)
        logger.info(f"Finetune {config['name']}: acc={result['test_acc_mean']:.4f}±{result['test_acc_std']:.4f}")

    return all_results
