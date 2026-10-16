"""
效率分析实验 - OpenMAG Q9-Q10
理论复杂度分析 + 实际训练时间和显存分析
"""
import torch
import torch.nn.functional as F
import numpy as np
import time
from loguru import logger


def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


def profile_training_efficiency(
    model, data, device="cuda", epochs=10, lr=0.001
):
    """
    分析训练效率

    Returns:
        dict with time_per_epoch, peak_memory, params
    """
    model = model.to(device)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # 显存统计
    if device != "cpu":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    times = []
    for epoch in range(epochs):
        model.train()

        if device != "cpu":
            torch.cuda.synchronize()
        t0 = time.time()

        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        if device != "cpu":
            torch.cuda.synchronize()
        t1 = time.time()
        times.append(t1 - t0)

    peak_mem = 0
    if device != "cpu":
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB

    params_info = count_parameters(model)

    return {
        "time_per_epoch_ms": float(np.mean(times) * 1000),
        "time_std_ms": float(np.std(times) * 1000),
        "peak_memory_mb": float(peak_mem),
        "total_params": params_info["total_params"],
        "trainable_params": params_info["trainable_params"],
    }


def run_efficiency_analysis(
    models_dict, data, device="cuda", epochs=50, lr=0.001
):
    """
    对比多个模型的效率

    Args:
        models_dict: {model_name: (model_cls, model_params)}

    Returns:
        list of efficiency results
    """
    results = []

    for name, (cls, params) in models_dict.items():
        logger.info(f"Profiling {name}...")
        model = cls(**params)

        try:
            eff = profile_training_efficiency(model, data, device, epochs, lr)
            eff["model"] = name
            results.append(eff)
            logger.info(f"  {name}: {eff['time_per_epoch_ms']:.1f}ms/epoch, "
                        f"{eff['peak_memory_mb']:.0f}MB, {eff['total_params']} params")
        except Exception as e:
            logger.error(f"  {name} failed: {e}")
            results.append({
                "model": name, "error": str(e),
                "time_per_epoch_ms": 0, "peak_memory_mb": 0,
                "total_params": 0, "trainable_params": 0,
            })

    return results


def generate_complexity_table():
    """生成理论复杂度分析表（OpenMAG Table 6）"""
    complexity_data = [
        {
            "model": "MLP",
            "training": "O(L|V|d^2)",
            "inference": "O(L|V|d^2)",
            "memory": "O(L|V|d)",
        },
        {
            "model": "GCN",
            "training": "O(L|E|d + L|V|d^2)",
            "inference": "O(L|E|d + L|V|d^2)",
            "memory": "O(L|V|d + |E|)",
        },
        {
            "model": "GAT",
            "training": "O(L|E|dH + L|V|d^2H)",
            "inference": "O(L|E|dH + L|V|d^2H)",
            "memory": "O(L|V|dH + |E|)",
        },
        {
            "model": "GraphSAGE",
            "training": "O(L|E|d + L|V|d^2)",
            "inference": "O(L|E|d + L|V|d^2)",
            "memory": "O(L|V|d + |E|)",
        },
        {
            "model": "GIN",
            "training": "O(L|E|d + L|V|d^2)",
            "inference": "O(L|E|d + L|V|d^2)",
            "memory": "O(L|V|d + |E|)",
        },
        {
            "model": "TransformerConv",
            "training": "O(L|E|dH + L|V|d^2H)",
            "inference": "O(L|E|dH + L|V|d^2H)",
            "memory": "O(L|V|dH + |E|)",
        },
        {
            "model": "RevGAT",
            "training": "O(L|E|dH + L|V|d^2H)",
            "inference": "O(L|E|dH + L|V|d^2H)",
            "memory": "O(L|V|dH + |E|)",
        },
    ]
    return complexity_data
