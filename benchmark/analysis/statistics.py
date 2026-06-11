"""
数据集统计分析脚本 (实验0)
生成 AMMAG 数据集的全面统计报告和可视化图表
"""
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from loguru import logger
from collections import defaultdict

# 支持中文
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_statistics(processed_dir: str) -> dict:
    """加载预计算的统计信息"""
    stats_path = os.path.join(processed_dir, "statistics.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def plot_degree_distribution(edge_index, output_dir: str):
    """绘制度分布图"""
    import torch
    num_nodes = edge_index.max().item() + 1
    degrees = torch.zeros(num_nodes, dtype=torch.long)
    degrees.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), dtype=torch.long))
    degrees.scatter_add_(0, edge_index[1], torch.ones(edge_index.size(1), dtype=torch.long))
    deg_np = degrees.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 线性刻度
    axes[0].hist(deg_np, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].set_xlabel("Degree")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Degree Distribution (Linear)")
    axes[0].axvline(np.mean(deg_np), color="red", linestyle="--", label=f"Mean={np.mean(deg_np):.1f}")
    axes[0].legend()

    # 对数刻度
    axes[1].hist(deg_np, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Degree")
    axes[1].set_ylabel("Count (log)")
    axes[1].set_title("Degree Distribution (Log)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "degree_distribution.png"), dpi=150)
    plt.close()
    logger.info(f"度分布图已保存")


def plot_modality_distribution(stats: dict, output_dir: str):
    """绘制各领域的模态分布对比图"""
    domain_stats = stats.get("domain_stats", {})
    if not domain_stats:
        logger.warning("无领域统计数据")
        return

    domains = list(domain_stats.keys())
    domain_labels = [d.replace("_", ".").upper() for d in domains]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 平均图片数
    avg_imgs = [domain_stats[d].get("avg_images", 0) for d in domains]
    axes[0].bar(domain_labels, avg_imgs, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"][:len(domains)])
    axes[0].set_title("Avg Images per Paper")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(avg_imgs):
        axes[0].text(i, v + 0.1, f"{v:.1f}", ha="center")

    # 平均公式数
    avg_formulas = [domain_stats[d].get("avg_formulas", 0) for d in domains]
    axes[1].bar(domain_labels, avg_formulas, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"][:len(domains)])
    axes[1].set_title("Avg Formulas per Paper")
    axes[1].set_ylabel("Count")
    for i, v in enumerate(avg_formulas):
        axes[1].text(i, v + 0.1, f"{v:.1f}", ha="center")

    # 平均文本长度
    avg_text = [domain_stats[d].get("avg_text_length", 0) / 1000 for d in domains]
    axes[2].bar(domain_labels, avg_text, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"][:len(domains)])
    axes[2].set_title("Avg Text Length (x1000 chars)")
    axes[2].set_ylabel("Characters (K)")
    for i, v in enumerate(avg_text):
        axes[2].text(i, v + 0.5, f"{v:.1f}K", ha="center")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "modality_distribution.png"), dpi=150)
    plt.close()
    logger.info(f"模态分布图已保存")


def plot_label_distribution(stats: dict, output_dir: str):
    """绘制标签分布图"""
    label_dist = stats.get("label_distribution", {})
    if not label_dist:
        return

    domains = list(label_dist.keys())
    counts = [label_dist[d] for d in domains]
    labels = [d.replace("_", ".").upper() for d in domains]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"][:len(domains)])
    ax.set_ylabel("Number of Papers")
    ax.set_title("Label Distribution")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                str(count), ha="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "label_distribution.png"), dpi=150)
    plt.close()
    logger.info(f"标签分布图已保存")


def generate_comparison_table(stats: dict) -> str:
    """生成与现有数据集的对比表（LaTeX格式）"""
    n_nodes = stats.get("num_nodes", 0)
    n_edges = stats.get("num_edges", 0)
    avg_deg = stats.get("avg_degree", 0)
    n_classes = stats.get("num_classes", 0)

    table = r"""
\begin{table}[h]
\centering
\caption{Comparison with existing graph benchmarks}
\label{tab:dataset_comparison}
\begin{tabular}{lccccccc}
\toprule
Dataset & Domain & Nodes & Edges & Avg.Deg & Classes & Text & Image & Formula \\
\midrule
Cora & Academic & 2,708 & 5,429 & 4.0 & 7 & BoW & \xmark & \xmark \\
CiteSeer & Academic & 3,327 & 4,732 & 2.8 & 6 & BoW & \xmark & \xmark \\
PubMed & Academic & 19,717 & 44,338 & 4.5 & 3 & BoW & \xmark & \xmark \\
ogbn-arxiv & Academic & 169K & 1.16M & 13.7 & 40 & 128-dim & \xmark & \xmark \\
MAGB-Movies & E-commerce & 16K & 218K & 27.2 & 20 & Raw & Raw & \xmark \\
MAGB-Reddit & Social & 15K & 566K & 75.5 & 20 & Raw & Raw & \xmark \\
\midrule
AMMAG (ours) & Academic & """ + f"{n_nodes:,}" + r""" & """ + f"{n_edges:,}" + r""" & """ + f"{avg_deg:.1f}" + r""" & """ + str(n_classes) + r""" & Raw & Raw & \cmark \\
\bottomrule
\end{tabular}
\end{table}
"""
    return table


def generate_report(stats: dict, output_dir: str):
    """生成完整的统计报告"""
    report = []
    report.append("=" * 60)
    report.append("AMMAG Dataset Statistics Report")
    report.append("=" * 60)

    report.append(f"\nTotal Nodes (Papers): {stats.get('num_nodes', 0):,}")
    report.append(f"Total Edges (Citations): {stats.get('num_edges', 0):,}")
    report.append(f"Average Degree: {stats.get('avg_degree', 0):.2f}")
    report.append(f"Number of Classes: {stats.get('num_classes', 0)}")

    report.append(f"\n--- Text Statistics ---")
    tl = stats.get("text_length", {})
    report.append(f"  Mean length: {tl.get('mean', 0):.0f} chars")
    report.append(f"  Median length: {tl.get('median', 0):.0f} chars")

    report.append(f"\n--- Image Statistics ---")
    ic = stats.get("image_count", {})
    report.append(f"  Mean images/paper: {ic.get('mean', 0):.2f}")
    report.append(f"  Papers with images: {ic.get('papers_with_images', 0):,}")
    report.append(f"  Papers without images: {ic.get('papers_without_images', 0):,}")

    report.append(f"\n--- Formula Statistics ---")
    fc = stats.get("formula_count", {})
    report.append(f"  Mean formulas/paper: {fc.get('mean', 0):.2f}")
    report.append(f"  Papers with formulas: {fc.get('papers_with_formulas', 0):,}")
    report.append(f"  Papers without formulas: {fc.get('papers_without_formulas', 0):,}")

    report.append(f"\n--- Domain Breakdown ---")
    for domain, ds in stats.get("domain_stats", {}).items():
        report.append(f"  {domain}:")
        report.append(f"    Papers: {ds.get('num_papers', 0):,}")
        report.append(f"    Avg images: {ds.get('avg_images', 0):.2f}")
        report.append(f"    Avg formulas: {ds.get('avg_formulas', 0):.2f}")
        report.append(f"    Avg text length: {ds.get('avg_text_length', 0):.0f} chars")

    report_text = "\n".join(report)
    with open(os.path.join(output_dir, "statistics_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)

    # 保存LaTeX对比表
    latex_table = generate_comparison_table(stats)
    with open(os.path.join(output_dir, "comparison_table.tex"), "w", encoding="utf-8") as f:
        f.write(latex_table)

    print(report_text)
    logger.info(f"统计报告已保存到 {output_dir}/statistics_report.txt")


def main():
    parser = argparse.ArgumentParser(description="AMMAG数据集统计分析")
    parser.add_argument("--processed_dir", type=str, default="./benchmark/processed",
                        help="预处理数据目录")
    parser.add_argument("--output_dir", type=str, default="./benchmark/analysis/output",
                        help="输出目录")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    stats = load_statistics(args.processed_dir)
    if not stats:
        logger.error("未找到统计信息，请先运行数据预处理")
        return

    generate_report(stats, args.output_dir)
    plot_label_distribution(stats, args.output_dir)
    plot_modality_distribution(stats, args.output_dir)

    # 度分布图需要edge_index
    graph_path = os.path.join(args.processed_dir, "graph_data.pt")
    if os.path.exists(graph_path):
        import torch
        data = torch.load(graph_path, map_location="cpu")
        if "edge_index" in data:
            plot_degree_distribution(data["edge_index"], args.output_dir)

    logger.info("统计分析完成!")


if __name__ == "__main__":
    main()
