"""
PEFT策略对比实验 - OpenMAG Q6
对比 None / Adapter / LoRA 三种参数高效微调策略
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loguru import logger


class AdapterModule(nn.Module):
    """Bottleneck Adapter"""

    def __init__(self, dim=768, bottleneck_dim=64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, dim)
        nn.init.xavier_uniform_(self.down.weight, gain=0.01)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(F.gelu(self.down(x)))


class LoRAModule(nn.Module):
    """Low-Rank Adaptation"""

    def __init__(self, dim=768, rank=8):
        super().__init__()
        self.A = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, dim))

    def forward(self, x):
        return x + (x @ self.A @ self.B)


class PEFTModel(nn.Module):
    """包装GNN模型，在输入端添加PEFT模块"""

    def __init__(self, gnn_model, peft_type="none", dim=768,
                 bottleneck_dim=64, lora_rank=8):
        super().__init__()
        self.gnn = gnn_model
        self.peft_type = peft_type

        if peft_type == "adapter":
            self.peft = AdapterModule(dim, bottleneck_dim)
        elif peft_type == "lora":
            self.peft = LoRAModule(dim, lora_rank)
        else:
            self.peft = None

    def forward(self, x, edge_index):
        if self.peft is not None:
            x = self.peft(x)
        return self.gnn(x, edge_index)


def run_peft_comparison(
    model_cls, model_params, data, device="cuda",
    peft_configs=None, num_runs=3, epochs=500, lr=0.001, patience=25,
    dim=768, raw_dim=None,
):
    """
    对比不同PEFT策略

    Args:
        model_cls: GNN模型类
        model_params: 模型参数
        peft_configs: PEFT配置列表
        dim: 特征维度

    Returns:
        list of result dicts
    """
    if peft_configs is None:
        peft_configs = [
            {"name": "None", "type": "none"},
            {"name": "Adapter(b=64)", "type": "adapter", "bottleneck_dim": 64},
            {"name": "LoRA(r=8)", "type": "lora", "lora_rank": 8},
            {"name": "LoRA(r=16)", "type": "lora", "lora_rank": 16},
        ]

    all_results = []

    for config in peft_configs:
        accs = []

        for run in range(num_runs):
            seed = run * 42 + 1
            torch.manual_seed(seed)
            np.random.seed(seed)

            gnn = model_cls(**model_params)
            from benchmark.models.gnn import ProjectedModel
            peft_model = PEFTModel(
                gnn, peft_type=config["type"], dim=dim,
                bottleneck_dim=config.get("bottleneck_dim", 64),
                lora_rank=config.get("lora_rank", 8),
            )
            # Wrap PEFT+GNN with projection if raw_dim differs
            if raw_dim and raw_dim != dim:
                model = ProjectedModel(peft_model, raw_dim, dim).to(device)
            else:
                model = peft_model.to(device)

            # 冻结GNN参数（只训练PEFT + 分类头）
            if config["type"] != "none":
                for name, param in model.named_parameters():
                    if "peft" not in name:
                        param.requires_grad = False

            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=lr
            )
            criterion = nn.CrossEntropyLoss()

            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            y = data.y.to(device)
            train_mask = data.train_mask.to(device)
            val_mask = data.val_mask.to(device)
            test_mask = data.test_mask.to(device)

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
            "peft_strategy": config["name"],
            "test_acc_mean": float(np.mean(accs)),
            "test_acc_std": float(np.std(accs)),
        }
        all_results.append(result)
        logger.info(f"PEFT {config['name']}: acc={result['test_acc_mean']:.4f}±{result['test_acc_std']:.4f}")

    return all_results
