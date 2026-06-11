"""
可视化分析 - t-SNE降维、模态消融柱状图、跨域热力图
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from loguru import logger

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DOMAIN_LABELS = ["cs.SI", "cs.LO", "math.LO", "physics.soc-ph"]
DOMAIN_KEYS = ["cs_si", "cs_lo", "math_lo", "physics_socph"]
MOD_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD", "#8C5613"]


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: str,
    label_names: list = None,
    perplexity: int = 30,
    sample_size: int = 0,
):
    """t-SNE可视化，支持采样加速"""
    if sample_size > 0 and len(embeddings) > sample_size:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(embeddings), sample_size, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(embeddings) - 1),
                random_state=42)
    coords = tsne.fit_transform(embeddings)

    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 2)))

    fig, ax = plt.subplots(figsize=(8, 7))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = label_names[label] if label_names and label < len(label_names) else str(label)
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[colors[i]], label=name, alpha=0.6, s=15)

    ax.set_title(title, fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"t-SNE图已保存: {output_path}")


def plot_ablation_bar(
    results: dict,
    domains: list = None,
    output_path: str = None,
    metric: str = "test_acc_mean",
):
    """
    模态消融柱状图 - 各领域×各模态的分组柱状图 (4子图)

    Args:
        results: {modality: {per_domain_acc: {domain: mean, domain_std: std}, test_acc_mean: x}}
    """
    if domains is None:
        domains = DOMAIN_KEYS
    domain_labels = [DOMAIN_LABELS[DOMAIN_KEYS.index(d)] for d in domains]
    modalities = ["text", "image", "formula", "text+image", "text+formula",
                  "image+formula", "text+image+formula"]
    mod_labels = ["Text", "Image", "Formula", "T+I", "T+F", "I+F", "T+I+F"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for idx, domain in enumerate(domains):
        ax = axes[idx]
        values = []
        for mod in modalities:
            r = results.get(mod, {})
            pd = r.get("per_domain_acc", {})
            val = pd.get(f"{domain}_acc_mean", r.get(metric, 0))
            values.append(val)

        bars = ax.bar(range(len(modalities)), values, color=MOD_COLORS)
        ax.set_xticks(range(len(modalities)))
        ax.set_xticklabels(mod_labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Accuracy")
        ax.set_title(domain_labels[idx], fontsize=12)
        ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", fontsize=8)

    plt.suptitle("Per-Domain Modality Ablation (GCN)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"消融柱状图已保存: {output_path}")


def plot_cross_domain_heatmap(
    results: dict,
    output_path: str,
    metric: str = "test_acc_mean",
):
    """
    跨域迁移热力图

    Args:
        results: {(src, tgt): {metric: value}} or dict with source_target keys
    """
    domains = DOMAIN_KEYS
    domain_labels = DOMAIN_LABELS

    matrix = np.zeros((4, 4))
    counts = np.zeros((4, 4))

    for key, val in results.items():
        if isinstance(key, tuple):
            src, tgt = key
        else:
            continue
        i = DOMAIN_KEYS.index(src) if src in DOMAIN_KEYS else -1
        j = DOMAIN_KEYS.index(tgt) if tgt in DOMAIN_KEYS else -1
        if i >= 0 and j >= 0:
            matrix[i][j] = val.get(metric, 0)
            counts[i][j] = 1

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(domain_labels, rotation=45)
    ax.set_yticklabels(domain_labels)
    ax.set_xlabel("Target Domain")
    ax.set_ylabel("Source Domain")
    ax.set_title("Cross-Domain Transfer Performance")

    for i in range(4):
        for j in range(4):
            if counts[i][j] > 0:
                color = "white" if matrix[i][j] > 0.6 else "black"
                ax.text(j, i, f"{matrix[i][j]:.3f}", ha="center", va="center",
                        color=color)
            elif i == j:
                ax.text(j, i, "-", ha="center", va="center", color="gray")

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"跨域热力图已保存: {output_path}")


def plot_few_shot_comparison(
    results: dict,
    output_path: str,
):
    """Few-shot结果对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (k, title) in enumerate([(3, "3-Shot"), (10, "10-Shot")]):
        ax = axes[ax_idx]
        models = sorted(set(v["model"] for v in results.values()))
        modalities = ["text", "text+image+formula"]
        mod_labels = ["Text", "Multi-modal"]

        x = np.arange(len(models))
        width = 0.35

        for m_idx, mod in enumerate(modalities):
            vals = []
            for model in models:
                key = f"{model}_{mod}_k={k}"
                vals.append(results.get(key, {}).get("test_acc_mean", 0))
            ax.bar(x + m_idx * width, vals, width, label=mod_labels[m_idx],
                   color=MOD_COLORS[m_idx])

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(models)
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.legend()

    plt.suptitle("Few-Shot Node Classification", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Few-shot对比图已保存: {output_path}")


# ==================== OpenMAG 新增可视化 ====================


def plot_noise_robustness(results: dict, output_path: str):
    """Q7-Q8: 噪声鲁棒性曲线 (4子图: text/image/label/edge)"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    noise_types = ["text", "image", "label", "edge"]
    titles = ["(a) Text Noise", "(b) Image Noise", "(c) Label Noise", "(d) Edge Noise"]

    for idx, noise_type in enumerate(noise_types):
        ax = axes[idx]
        data = results.get(noise_type, {}).get("results", {})
        if not data:
            continue

        ratios = []
        accs = []
        stds = []
        for key in sorted(data.keys()):
            ratio = float(key.replace("ratio_", ""))
            ratios.append(ratio)
            accs.append(data[key]["mean"])
            stds.append(data[key]["std"])

        ax.errorbar(ratios, accs, yerr=stds, marker="o", capsize=3,
                     color="#4C72B0", linewidth=2)
        ax.set_xlabel("Noise Ratio")
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_title(titles[idx])
        ax.set_ylim(0, max(accs) * 1.1 if accs else 1)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Robustness Analysis (GCN)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"噪声鲁棒性图已保存: {output_path}")


def plot_neighbor_ablation(results: dict, output_path: str):
    """Q2: 邻居数消融曲线"""
    data = results.get("results", {})
    if not data:
        return

    ks, accs, stds = [], [], []
    for key in sorted(data.keys()):
        k = int(key.replace("k_", ""))
        ks.append(k)
        accs.append(data[key]["mean"])
        stds.append(data[key]["std"])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(ks, accs, yerr=stds, marker="s", capsize=4,
                 color="#C44E52", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Neighbors (K)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Graph Structure Effect (Neighbor Ablation)")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)

    for k, acc in zip(ks, accs):
        ax.annotate(f"{acc:.3f}", (k, acc), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"邻居消融图已保存: {output_path}")


def plot_effectiveness_table(nc_results, lp_results, cluster_results, output_path: str):
    """Q5: 模型有效性对比柱状图"""
    models = [r["model"] for r in nc_results]
    nc_accs = [r["acc_mean"] for r in nc_results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # NC
    ax = axes[0]
    bars = ax.bar(range(len(models)), nc_accs, color="#4C72B0")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Node Classification")
    for bar, val in zip(bars, nc_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", fontsize=8)

    # LP
    ax = axes[1]
    lp_models = [r["model"] for r in lp_results]
    lp_mrr = [r.get("mrr_mean", 0) for r in lp_results]
    bars = ax.bar(range(len(lp_models)), lp_mrr, color="#55A868")
    ax.set_xticks(range(len(lp_models)))
    ax.set_xticklabels(lp_models, rotation=45, ha="right")
    ax.set_ylabel("MRR")
    ax.set_title("Link Prediction")

    # Clustering
    ax = axes[2]
    cl_models = [r["model"] for r in cluster_results]
    cl_nmi = [r["nmi_mean"] for r in cluster_results]
    cl_ari = [r["ari_mean"] for r in cluster_results]
    x = np.arange(len(cl_models))
    ax.bar(x - 0.15, cl_nmi, 0.3, label="NMI", color="#C44E52")
    ax.bar(x + 0.15, cl_ari, 0.3, label="ARI", color="#8172B2")
    ax.set_xticks(x)
    ax.set_xticklabels(cl_models, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Node Clustering")
    ax.legend()

    plt.suptitle("Model Effectiveness Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"有效性对比图已保存: {output_path}")


def plot_peft_comparison(results: list, output_path: str):
    """Q6: PEFT策略对比"""
    names = [r["peft_strategy"] for r in results]
    means = [r["test_acc_mean"] for r in results]
    stds = [r["test_acc_std"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(range(len(names)), means, yerr=stds, capsize=4,
                   color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"][:len(names)])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("PEFT Strategy Comparison (GCN)")

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"PEFT对比图已保存: {output_path}")


def plot_efficiency(empirical: list, output_path: str):
    """Q10: 效率分析 - 时间 vs 精度散点图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    names = [r["model"] for r in empirical if "error" not in r]
    times = [r["time_per_epoch_ms"] for r in empirical if "error" not in r]
    mems = [r["peak_memory_mb"] for r in empirical if "error" not in r]
    params = [r["total_params"] / 1e6 for r in empirical if "error" not in r]

    # Time
    ax1.barh(range(len(names)), times, color="#4C72B0")
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names)
    ax1.set_xlabel("Time per Epoch (ms)")
    ax1.set_title("Training Time")

    # Memory
    ax2.barh(range(len(names)), mems, color="#C44E52")
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names)
    ax2.set_xlabel("Peak GPU Memory (MB)")
    ax2.set_title("Memory Usage")

    plt.suptitle("Efficiency Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"效率分析图已保存: {output_path}")


def plot_finetune_comparison(results: list, output_path: str):
    """Q3: 微调策略对比"""
    names = [r["config"] for r in results]
    means = [r["test_acc_mean"] for r in results]
    stds = [r["test_acc_std"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(range(len(names)), means, yerr=stds, capsize=4,
                   color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Frozen vs Trainable Encoder Comparison")

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"微调对比图已保存: {output_path}")

