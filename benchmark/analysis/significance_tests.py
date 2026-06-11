"""
统计显著性检验 - 验证多模态 vs 单模态的差异是否显著
配对t检验 + Wilcoxon符号秩检验
"""
import os
import json
import numpy as np
from scipy import stats
from loguru import logger


def load_results(results_dir: str, pattern_prefix: str = "nc_") -> dict:
    """加载实验结果JSON文件"""
    results = {}
    for f in os.listdir(results_dir):
        if f.startswith(pattern_prefix) and f.endswith(".json"):
            path = os.path.join(results_dir, f)
            with open(path, "r") as fp:
                data = json.load(fp)
            key = f.replace(pattern_prefix, "").replace(".json", "")
            results[key] = data
    return results


def paired_significance_test(
    values_a: list,
    values_b: list,
    name_a: str = "A",
    name_b: str = "B",
    alpha: float = 0.05,
) -> dict:
    """
    配对显著性检验：配对t检验 + Wilcoxon符号秩检验
    """
    a = np.array(values_a)
    b = np.array(values_b)

    if len(a) != len(b):
        return {"error": "length mismatch"}

    # 配对t检验（参数检验）
    t_stat, t_pvalue = stats.ttest_rel(a, b)

    # Wilcoxon符号秩检验（非参数检验）
    try:
        w_stat, w_pvalue = stats.wilcoxon(a, b)
    except ValueError:
        w_stat, w_pvalue = np.nan, np.nan

    # 效应量 (Cohen's d for paired samples)
    diff = a - b
    d_mean = np.mean(diff)
    d_std = np.std(diff, ddof=1)
    cohens_d = d_mean / d_std if d_std > 0 else 0

    return {
        "name_a": name_a,
        "name_b": name_b,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "diff_mean": float(d_mean),
        "t_statistic": float(t_stat),
        "t_pvalue": float(t_pvalue),
        "t_significant": bool(t_pvalue < alpha),
        "wilcoxon_statistic": float(w_stat) if not np.isnan(w_stat) else None,
        "wilcoxon_pvalue": float(w_pvalue) if not np.isnan(w_pvalue) else None,
        "wilcoxon_significant": bool(w_pvalue < alpha) if not np.isnan(w_pvalue) else None,
        "cohens_d": float(cohens_d),
        "alpha": alpha,
    }


def run_ablation_significance_tests(results_dir: str, output_dir: str):
    """
    对模态消融实验进行显著性检验
    对每个域，检验 T+I+F vs Text-only 的差异是否显著
    """
    os.makedirs(output_dir, exist_ok=True)

    # 查找消融实验结果 - 按文件名推断模态
    ablation_results = {}
    for f in os.listdir(results_dir):
        if f.startswith("ablation_pd_gcn_") and f.endswith(".json"):
            path = os.path.join(results_dir, f)
            with open(path, "r") as fp:
                data = json.load(fp)
            # 从文件名提取模态: ablation_pd_gcn_TEXT_supervised.json
            parts = f.replace("ablation_pd_gcn_", "").replace("_supervised.json", "")
            modality = parts
            ablation_results[modality] = data

    if not ablation_results:
        logger.warning(f"未在 {results_dir} 中找到消融实验结果")
        return

    # 提取每次运行的test_acc
    text_accs = []
    trif_accs = []
    for run in ablation_results.get("text", {}).get("runs", []):
        text_accs.append(run["test_acc"])
    for run in ablation_results.get("text+image+formula", {}).get("runs", []):
        trif_accs.append(run["test_acc"])

    if not text_accs or not trif_accs:
        logger.warning(f"缺少Text或T+I+F的多次运行结果, found modalities: {list(ablation_results.keys())}")
        # 生成基于mean±std的模拟报告
        text_data = ablation_results.get("text", {})
        trif_data = ablation_results.get("text+image+formula", {})
        if text_data and trif_data:
            text_mean = text_data.get("test_acc_mean", 0)
            text_std = text_data.get("test_acc_std", 0)
            trif_mean = trif_data.get("test_acc_mean", 0)
            trif_std = trif_data.get("test_acc_std", 0)
            report = []
            report.append("=" * 60)
            report.append("显著性检验报告（基于mean±std估算）")
            report.append("=" * 60)
            report.append(f"\nT+I+F: {trif_mean:.4f} ± {trif_std:.4f}")
            report.append(f"Text:   {text_mean:.4f} ± {text_std:.4f}")
            report.append(f"差值:   {(trif_mean - text_mean)*100:.2f}pp")
            report.append(f"\n注：缺少逐次运行数据，无法进行配对检验")
            report.append(f"std重叠分析：T+I+F的区间 [{trif_mean-trif_std:.4f}, {trif_mean+trif_std:.4f}]")
            report.append(f"              Text的区间   [{text_mean-text_std:.4f}, {text_mean+text_std:.4f}]")
            overlap = max(0, min(trif_mean+trif_std, text_mean+text_std) - max(trif_mean-trif_std, text_mean-text_std))
            report.append(f"重叠区间长度: {overlap:.4f}")

            report_text = "\n".join(report)
            print(report_text)
            with open(os.path.join(output_dir, "significance_tests.txt"), "w") as f:
                f.write(report_text)
        return

    if not text_accs or not trif_accs:
        logger.warning("缺少Text或T+I+F的多次运行结果")
        return

    # Overall显著性检验
    result = paired_significance_test(
        trif_accs, text_accs,
        name_a="T+I+F", name_b="Text"
    )

    report = []
    report.append("=" * 60)
    report.append("显著性检验报告：T+I+F vs Text-only (GCN)")
    report.append("=" * 60)
    report.append(f"\nT+I+F mean: {result['mean_a']:.4f}")
    report.append(f"Text   mean: {result['mean_b']:.4f}")
    report.append(f"差值均值: {result['diff_mean']:.4f} ({result['diff_mean']*100:.2f}pp)")
    report.append(f"\n配对t检验: t={result['t_statistic']:.4f}, p={result['t_pvalue']:.4f}")
    report.append(f"  显著性(alpha=0.05): {'是' if result['t_significant'] else '否'}")
    if result['wilcoxon_pvalue'] is not None:
        report.append(f"\nWilcoxon检验: W={result['wilcoxon_statistic']:.4f}, p={result['wilcoxon_pvalue']:.4f}")
        report.append(f"  显著性(alpha=0.05): {'是' if result['wilcoxon_significant'] else '否'}")
    report.append(f"\nCohen's d (效应量): {result['cohens_d']:.4f}")

    # Per-domain显著性检验（如果有per-domain数据）
    text_per_domain = {}
    trif_per_domain = {}
    for run in ablation_results.get("text", {}).get("runs", []):
        if "per_domain_acc" in run:
            for domain, acc in run["per_domain_acc"].items():
                text_per_domain.setdefault(domain, []).append(acc)
    for run in ablation_results.get("text+image+formula", {}).get("runs", []):
        if "per_domain_acc" in run:
            for domain, acc in run["per_domain_acc"].items():
                trif_per_domain.setdefault(domain, []).append(acc)

    if text_per_domain and trif_per_domain:
        report.append("\n" + "=" * 60)
        report.append("Per-Domain 显著性检验")
        report.append("=" * 60)
        for domain in sorted(text_per_domain.keys()):
            if domain in trif_per_domain:
                r = paired_significance_test(
                    trif_per_domain[domain], text_per_domain[domain],
                    name_a=f"T+I+F({domain})", name_b=f"Text({domain})"
                )
                report.append(f"\n{domain}:")
                report.append(f"  T+I+F={r['mean_a']:.4f}, Text={r['mean_b']:.4f}, "
                             f"Δ={r['diff_mean']*100:.2f}pp")
                report.append(f"  t-test p={r['t_pvalue']:.4f} "
                             f"{'*' if r['t_significant'] else '(n.s.)'}")
                report.append(f"  Cohen's d={r['cohens_d']:.3f}")

    report_text = "\n".join(report)
    print(report_text)

    with open(os.path.join(output_dir, "significance_tests.txt"), "w") as f:
        f.write(report_text)
    with open(os.path.join(output_dir, "significance_tests.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info(f"显著性检验结果保存到 {output_dir}")


def run_fewshot_significance_tests(results_dir: str, output_dir: str):
    """对Few-shot实验进行显著性检验"""
    os.makedirs(output_dir, exist_ok=True)

    report = []
    report.append("=" * 60)
    report.append("Few-shot 显著性检验：T+I+F vs Text-only")
    report.append("=" * 60)

    for model in ["gcn", "gat", "graphsage", "gin", "gt"]:
        for k in [3, 10]:
            text_key = f"nc_{model}_text_supervised_k{k}"
            trif_key = f"nc_{model}_text+image+formula_supervised_k{k}"

            text_path = os.path.join(results_dir, text_key + ".json")
            trif_path = os.path.join(results_dir, trif_key + ".json")

            if not os.path.exists(text_path) or not os.path.exists(trif_path):
                continue

            with open(text_path) as f:
                text_data = json.load(f)
            with open(trif_path) as f:
                trif_data = json.load(f)

            text_accs = [r["test_acc"] for r in text_data.get("runs", [])]
            trif_accs = [r["test_acc"] for r in trif_data.get("runs", [])]

            if len(text_accs) >= 3 and len(trif_accs) >= 3:
                r = paired_significance_test(
                    trif_accs, text_accs,
                    name_a=f"T+I+F({model},{k}-shot)",
                    name_b=f"Text({model},{k}-shot)"
                )
                report.append(f"\n{model} {k}-shot:")
                report.append(f"  T+I+F={r['mean_a']:.4f}, Text={r['mean_b']:.4f}, "
                             f"Δ={r['diff_mean']*100:.2f}pp")
                report.append(f"  t-test p={r['t_pvalue']:.4f} "
                             f"{'*' if r['t_significant'] else '(n.s.)'}")

    report_text = "\n".join(report)
    print(report_text)
    with open(os.path.join(output_dir, "fewshot_significance.txt"), "w") as f:
        f.write(report_text)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="./benchmark/results")
    parser.add_argument("--output_dir", default="./benchmark/analysis/output")
    args = parser.parse_args()

    run_ablation_significance_tests(args.results_dir, args.output_dir)
    run_fewshot_significance_tests(args.results_dir, args.output_dir)
