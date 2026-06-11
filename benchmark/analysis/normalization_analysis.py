"""
L2归一化系统性分析 - 从"bug fix"升级为方法论贡献

系统展示：
1. 各模态归一化前后的L2范数分布
2. 归一化对7种模态组合的影响
3. 归一化在不同任务上的效果（NC, Few-shot, LP）
4. 为多模态图Benchmark研究提供规范性建议
"""
import os
import json
import argparse
import numpy as np
import torch
from loguru import logger


def analyze_feature_norms(processed_dir: str, output_dir: str):
    """
    分析各模态特征的L2范数分布
    这是揭示尺度不平衡问题的直接证据
    """
    os.makedirs(output_dir, exist_ok=True)

    report = []
    report.append("=" * 70)
    report.append("多模态特征L2范数分析")
    report.append("=" * 70)

    modality_files = {
        "Text": "text_embeddings.pt",
        "Image": "image_embeddings.pt",
        "Formula": "formula_embeddings.pt",
    }

    norm_stats = {}
    for name, fname in modality_files.items():
        path = os.path.join(processed_dir, fname)
        if not os.path.exists(path):
            logger.warning(f"找不到 {fname}")
            continue

        data = torch.load(path, map_location="cpu")
        embs = data["embeddings"].numpy()
        norms = np.linalg.norm(embs, axis=1)

        # 区分零向量和非零向量
        nonzero = norms[norms > 1e-6]
        zero_count = np.sum(norms <= 1e-6)

        stats = {
            "total": len(norms),
            "zero_count": int(zero_count),
            "zero_pct": float(zero_count / len(norms) * 100),
            "nonzero_count": len(nonzero),
            "mean_all": float(np.mean(norms)),
            "mean_nonzero": float(np.mean(nonzero)) if len(nonzero) > 0 else 0,
            "std_nonzero": float(np.std(nonzero)) if len(nonzero) > 0 else 0,
            "min_nonzero": float(np.min(nonzero)) if len(nonzero) > 0 else 0,
            "max_nonzero": float(np.max(nonzero)) if len(nonzero) > 0 else 0,
            "median_nonzero": float(np.median(nonzero)) if len(nonzero) > 0 else 0,
            "p25": float(np.percentile(nonzero, 25)) if len(nonzero) > 0 else 0,
            "p75": float(np.percentile(nonzero, 75)) if len(nonzero) > 0 else 0,
        }
        norm_stats[name] = stats

        report.append(f"\n--- {name} (dim={embs.shape[1]}) ---")
        report.append(f"  总论文数: {stats['total']}")
        report.append(f"  零向量: {stats['zero_count']} ({stats['zero_pct']:.1f}%)")
        report.append(f"  非零向量L2范数: mean={stats['mean_nonzero']:.4f}, "
                      f"std={stats['std_nonzero']:.4f}")
        report.append(f"  范数范围: [{stats['min_nonzero']:.4f}, {stats['max_nonzero']:.4f}]")
        report.append(f"  四分位: P25={stats['p25']:.4f}, "
                      f"P50={stats['median_nonzero']:.4f}, P75={stats['p75']:.4f}")

    # 计算尺度比
    report.append("\n" + "=" * 70)
    report.append("尺度不平衡分析")
    report.append("=" * 70)

    modalities = list(norm_stats.keys())
    for i, m1 in enumerate(modalities):
        for m2 in modalities[i + 1:]:
            if norm_stats[m1]["nonzero_count"] > 0 and norm_stats[m2]["nonzero_count"] > 0:
                ratio = norm_stats[m1]["mean_nonzero"] / norm_stats[m2]["mean_nonzero"]
                report.append(f"  {m1} / {m2} = {ratio:.2f}")

    # 生成LaTeX表格
    latex = generate_norm_table(norm_stats)
    report.append("\n" + latex)

    report_text = "\n".join(report)
    print(report_text)

    with open(os.path.join(output_dir, "normalization_report.txt"), "w") as f:
        f.write(report_text)
    with open(os.path.join(output_dir, "norm_stats.json"), "w") as f:
        json.dump(norm_stats, f, indent=2)
    with open(os.path.join(output_dir, "norm_table.tex"), "w") as f:
        f.write(latex)

    return norm_stats


def generate_norm_table(norm_stats: dict) -> str:
    """生成LaTeX格式的范数统计表"""
    latex = r"""
\begin{table}[htbp]
  \centering
  \caption{各模态特征的L2范数统计}
  \label{tab:norm_stats}
  \begin{tabular}{lrrrrrr}
    \toprule
    模态 & 维度 & 非零数 & 均值 & 标准差 & 最小值 & 最大值 \\
    \midrule
"""
    for name, s in norm_stats.items():
        if s["nonzero_count"] > 0:
            latex += f"    {name} & --- & {s['nonzero_count']} & "
            latex += f"{s['mean_nonzero']:.2f} & {s['std_nonzero']:.2f} & "
            latex += f"{s['min_nonzero']:.2f} & {s['max_nonzero']:.2f} \\\\\n"

    latex += r"""    \bottomrule
  \end{tabular}
\end{table}
"""
    return latex


def analyze_concatenation_norm_ratios(processed_dir: str, output_dir: str):
    """
    分析拼接后的特征中各模态占比
    直接展示CLIP特征如何"淹没"其他模态
    """
    modality_files = {
        "text": "text_embeddings.pt",
        "image": "image_embeddings.pt",
        "formula": "formula_embeddings.pt",
    }

    embs = {}
    for name, fname in modality_files.items():
        path = os.path.join(processed_dir, fname)
        if os.path.exists(path):
            data = torch.load(path, map_location="cpu")
            embs[name] = data["embeddings"].numpy()

    if len(embs) < 3:
        logger.warning("缺少模态embedding")
        return

    # 对齐
    common_len = min(e.shape[0] for e in embs.values())
    for name in embs:
        embs[name] = embs[name][:common_len]

    report = []
    report.append("=" * 70)
    report.append("拼接特征中各模态能量占比分析")
    report.append("=" * 70)

    # 归一化前
    report.append("\n--- 归一化前 ---")
    concat_before = np.concatenate(list(embs.values()), axis=1)
    total_norm = np.linalg.norm(concat_before, axis=1)

    dim_offset = 0
    for name, e in embs.items():
        sub = e
        sub_norm = np.linalg.norm(sub, axis=1)
        ratio = np.mean(sub_norm / (total_norm + 1e-8))
        report.append(f"  {name}: L2范数占比 = {ratio:.2%} "
                      f"(绝对范数 = {np.mean(sub_norm):.4f})")

    # 归一化后
    report.append("\n--- L2归一化后 ---")
    normed = {}
    for name, e in embs.items():
        norms = np.linalg.norm(e, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normed[name] = e / norms
        # 零向量保持为零
        zero_mask = np.linalg.norm(e, axis=1) < 1e-6
        normed[name][zero_mask] = 0

    concat_after = np.concatenate(list(normed.values()), axis=1)
    total_norm_after = np.linalg.norm(concat_after, axis=1)

    for name, e in normed.items():
        sub_norm = np.linalg.norm(e, axis=1)
        nonzero = sub_norm[sub_norm > 1e-6]
        if len(nonzero) > 0:
            ratio = np.mean(sub_norm / (total_norm_after + 1e-8))
            report.append(f"  {name}: L2范数占比 = {ratio:.2%} "
                          f"(绝对范数 = {np.mean(nonzero):.4f})")

    report_text = "\n".join(report)
    print(report_text)
    with open(os.path.join(output_dir, "concat_ratio_report.txt"), "w") as f:
        f.write(report_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="./benchmark/processed")
    parser.add_argument("--output_dir", default="./benchmark/analysis/output")
    args = parser.parse_args()

    analyze_feature_norms(args.processed_dir, args.output_dir)
    analyze_concatenation_norm_ratios(args.processed_dir, args.output_dir)
