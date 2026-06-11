"""
结果汇总脚本 - 读取所有JSON结果，生成LaTeX表格和汇总CSV
"""
import os
import json
import glob
import numpy as np
from typing import Dict, List
from loguru import logger


DOMAIN_KEYS = ["cs_si", "cs_lo", "math_lo", "physics_socph"]
DOMAIN_LABELS = ["cs.SI", "cs.LO", "math.LO", "physics.soc-ph"]
MODALITY_ORDER = ["text", "image", "formula", "text+image", "text+formula",
                  "image+formula", "text+image+formula"]
MOD_LABELS = ["Text", "Image", "Formula", "T+I", "T+F", "I+F", "T+I+F"]
MODEL_ORDER = ["mlp", "gcn", "gat", "graphsage", "gin", "gt"]


def load_all_results(results_dir: str) -> Dict:
    """加载所有结果JSON文件"""
    results = {}
    for f in glob.glob(os.path.join(results_dir, "*.json")):
        name = os.path.basename(f).replace(".json", "")
        try:
            with open(f, "r") as fh:
                results[name] = json.load(fh)
        except Exception:
            continue
    return results


def generate_nc_table(results: Dict, output_dir: str):
    """生成节点分类结果LaTeX表格"""
    rows = []
    header = "Model & " + " & ".join(MOD_LABELS) + " \\\\"
    rows.append("\\toprule")
    rows.append(header)
    rows.append("\\midrule")

    for model in MODEL_ORDER:
        acc_vals = []
        f1_vals = []
        for mod in MODALITY_ORDER:
            key = f"nc_{model}_{mod}_supervised"
            r = results.get(key, {})
            acc_mean = r.get("test_acc_mean", 0)
            acc_std = r.get("test_acc_std", 0)
            acc_vals.append(f"{acc_mean:.2f}$\\pm${acc_std:.2f}" if acc_mean > 0 else "-")
            f1_mean = r.get("test_f1_mean", 0)
            f1_std = r.get("test_f1_std", 0)
            f1_vals.append(f"{f1_mean:.2f}$\\pm${f1_std:.2f}" if f1_mean > 0 else "-")

        rows.append(f"{model.upper()} & " + " & ".join(acc_vals) + " \\\\")

    rows.append("\\bottomrule")

    # Save Accuracy table
    acc_table = "\\begin{table}[h]\n\\centering\n\\caption{Node Classification Accuracy (\\%)}\n\\label{tab:nc_acc}\n\\begin{tabular}{l" + "c" * len(MODALITY_ORDER) + "}\n" + "\n".join(rows) + "\n\\end{tabular}\n\\end{table}"

    with open(os.path.join(output_dir, "table_nc_accuracy.tex"), "w") as f:
        f.write(acc_table)

    # Save as readable text
    text_lines = ["Node Classification Results (Accuracy%)"]
    text_lines.append("-" * 120)
    text_lines.append(f"{'Model':<15}" + "".join(f"{l:>15}" for l in MOD_LABELS))
    text_lines.append("-" * 120)

    for model in MODEL_ORDER:
        vals = []
        for mod in MODALITY_ORDER:
            key = f"nc_{model}_{mod}_supervised"
            r = results.get(key, {})
            acc = r.get("test_acc_mean", 0)
            vals.append(f"{acc:.2f}" if acc > 0 else "-")
        text_lines.append(f"{model.upper():<15}" + "".join(f"{v:>15}" for v in vals))

    with open(os.path.join(output_dir, "nc_results_summary.txt"), "w") as f:
        f.write("\n".join(text_lines))

    logger.info(f"节点分类表格已生成: {output_dir}/table_nc_accuracy.tex")


def generate_ablation_table(results: Dict, output_dir: str):
    """生成per-domain模态消融表格（核心创新表）"""
    lines = []
    lines.append("Per-Domain Modality Ablation (GCN, Accuracy%)")
    lines.append("=" * 100)

    # Header
    lines.append(f"{'Domain':<20}" + "".join(f"{l:>12}" for l in MOD_LABELS))
    lines.append("-" * 100)

    for dname, dlabel in zip(DOMAIN_KEYS, DOMAIN_LABELS):
        vals = []
        for mod in MODALITY_ORDER:
            key = f"ablation_pd_gcn_{mod}_supervised"
            r = results.get(key, {})
            pd = r.get("per_domain_acc", {})
            val = pd.get(f"{dname}_acc_mean", 0)
            vals.append(f"{val:.2f}" if val > 0 else "-")
        lines.append(f"{dlabel:<20}" + "".join(f"{v:>12}" for v in vals))

    # Delta rows
    lines.append("-" * 100)
    lines.append("Delta vs Text-only:")
    for dname, dlabel in zip(DOMAIN_KEYS, DOMAIN_LABELS):
        text_key = "ablation_pd_gcn_text_supervised"
        text_r = results.get(text_key, {})
        text_pd = text_r.get("per_domain_acc", {})
        text_val = text_pd.get(f"{dname}_acc_mean", 0)

        deltas = []
        for mod in MODALITY_ORDER:
            key = f"ablation_pd_gcn_{mod}_supervised"
            r = results.get(key, {})
            pd = r.get("per_domain_acc", {})
            val = pd.get(f"{dname}_acc_mean", 0)
            delta = (val - text_val) * 100 if val > 0 and text_val > 0 else 0
            deltas.append(f"{delta:+.1f}" if val > 0 else "-")
        lines.append(f"  {dlabel:<18}" + "".join(f"{d:>12}" for d in deltas))

    with open(os.path.join(output_dir, "per_domain_ablation_summary.txt"), "w") as f:
        f.write("\n".join(lines))

    logger.info(f"模态消融表已生成: {output_dir}/per_domain_ablation_summary.txt")


def generate_lp_table(results: Dict, output_dir: str):
    """生成链接预测结果表格"""
    lp_models = ["gcn", "graphsage", "gat", "gt"]
    lp_mods = ["text", "text+image+formula", "image+formula"]
    metrics = ["mrr", "hits@1", "hits@3"]

    lines = []
    lines.append("Link Prediction Results")
    lines.append("=" * 80)
    lines.append(f"{'Model':<15} {'Modality':<20} {'MRR':>10} {'Hits@1':>10} {'Hits@3':>10}")
    lines.append("-" * 80)

    for model in lp_models:
        for mod in lp_mods:
            key = f"lp_{model}_{mod}"
            r = results.get(key, {})
            if not r:
                continue
            mrr = r.get("mrr_mean", 0)
            h1 = r.get("hits@1_mean", 0)
            h3 = r.get("hits@3_mean", 0)
            lines.append(f"{model.upper():<15} {mod:<20} {mrr:>10.4f} {h1:>10.4f} {h3:>10.4f}")

    with open(os.path.join(output_dir, "lp_results_summary.txt"), "w") as f:
        f.write("\n".join(lines))

    logger.info(f"链接预测表已生成: {output_dir}/lp_results_summary.txt")


def generate_few_shot_table(results: Dict, output_dir: str):
    """生成Few-shot结果表格"""
    fs_models = ["gcn", "gat", "graphsage", "gin", "gt"]
    fs_mods = ["text", "text+image+formula"]

    lines = []
    lines.append("Few-Shot Node Classification (Accuracy%)")
    lines.append("=" * 80)
    lines.append(f"{'Model':<15} {'Modality':<20} {'3-shot':>12} {'10-shot':>12}")
    lines.append("-" * 80)

    for model in fs_models:
        for mod in fs_mods:
            k3_key = f"nc_{model}_{mod}_few_shot_k3"
            k10_key = f"nc_{model}_{mod}_few_shot_k10"
            r3 = results.get(k3_key, {})
            r10 = results.get(k10_key, {})
            a3 = r3.get("test_acc_mean", 0)
            a10 = r10.get("test_acc_mean", 0)
            if a3 > 0 or a10 > 0:
                lines.append(f"{model.upper():<15} {mod:<20} {a3:>12.2f} {a10:>12.2f}")

    with open(os.path.join(output_dir, "few_shot_summary.txt"), "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Few-shot表已生成: {output_dir}/few_shot_summary.txt")


def generate_cross_domain_table(results: Dict, output_dir: str):
    """生成跨域迁移结果表格"""
    for mod_suffix in ["text+image+formula", "text"]:
        lines = []
        lines.append(f"Cross-Domain Transfer (GCN, {mod_suffix})")
        lines.append("=" * 100)
        lines.append(f"{'Source→Target':<20}" + "".join(f"{l:>18}" for l in DOMAIN_LABELS))
        lines.append("-" * 100)

        # Finetune accuracy
        lines.append("Finetune Accuracy%:")
        for src, sl in zip(DOMAIN_KEYS, DOMAIN_LABELS):
            vals = []
            for tgt in DOMAIN_KEYS:
                if src == tgt:
                    vals.append("  -  ")
                    continue
                key = f"cd_{src}_to_{tgt}_gcn_{mod_suffix}"
                r = results.get(key, {})
                acc = r.get("target_finetune_acc_mean", 0)
                vals.append(f"{acc:.2f}" if acc > 0 else "-")
            lines.append(f"  {sl:<18}" + "".join(f"{v:>18}" for v in vals))

        # Scratch baseline
        lines.append("-" * 100)
        lines.append("From-Scratch Baseline%:")
        for src, sl in zip(DOMAIN_KEYS, DOMAIN_LABELS):
            vals = []
            for tgt in DOMAIN_KEYS:
                if src == tgt:
                    vals.append("  -  ")
                    continue
                key = f"cd_{src}_to_{tgt}_gcn_{mod_suffix}"
                r = results.get(key, {})
                acc = r.get("target_scratch_acc_mean", 0)
                vals.append(f"{acc:.2f}" if acc > 0 else "-")
            lines.append(f"  {sl:<18}" + "".join(f"{v:>18}" for v in vals))

        # Transfer gain
        lines.append("-" * 100)
        lines.append("Transfer Gain (finetune - scratch):")
        for src, sl in zip(DOMAIN_KEYS, DOMAIN_LABELS):
            vals = []
            for tgt in DOMAIN_KEYS:
                if src == tgt:
                    vals.append("  -  ")
                    continue
                key = f"cd_{src}_to_{tgt}_gcn_{mod_suffix}"
                r = results.get(key, {})
                gain = r.get("transfer_gain_mean", 0)
                vals.append(f"{gain:+.2f}" if gain != 0 else "-")
            lines.append(f"  {sl:<18}" + "".join(f"{v:>18}" for v in vals))

        fname = f"cross_domain_summary_{mod_suffix.replace('+', '_')}.txt"
        with open(os.path.join(output_dir, fname), "w") as f:
            f.write("\n".join(lines))

        logger.info(f"跨域迁移表已生成: {output_dir}/{fname}")


def main(results_dir: str = "./benchmark/results", output_dir: str = None):
    if output_dir is None:
        output_dir = os.path.join(results_dir, "tables")
    os.makedirs(output_dir, exist_ok=True)

    results = load_all_results(results_dir)
    logger.info(f"加载 {len(results)} 个结果文件")

    generate_nc_table(results, output_dir)
    generate_ablation_table(results, output_dir)
    generate_lp_table(results, output_dir)
    generate_few_shot_table(results, output_dir)
    generate_cross_domain_table(results, output_dir)

    # OpenMAG tables
    generate_openmag_tables(results, output_dir)

    logger.info("所有表格生成完成!")


def generate_openmag_tables(results: Dict, output_dir: str):
    """生成OpenMAG实验结果表格 (Q1-Q10)"""
    tables_dir = os.path.join(output_dir, "openmag")
    os.makedirs(tables_dir, exist_ok=True)

    # Q1: Necessity
    q1 = results.get("q1_necessity", {}).get("results", [])
    if q1:
        lines = ["Q1: Unimodal vs Multimodal Necessity (Accuracy%)", "=" * 80]
        modalities = sorted(set(r["modality"] for r in q1))
        models = sorted(set(r["model"] for r in q1))
        lines.append(f"{'Model':<15}" + "".join(f"{m:>18}" for m in modalities))
        lines.append("-" * 80)
        for model in models:
            vals = []
            for mod in modalities:
                match = [r for r in q1 if r["model"] == model and r["modality"] == mod]
                if match:
                    vals.append(f"{match[0]['acc_mean']:.2f}±{match[0]['acc_std']:.2f}")
                else:
                    vals.append("-")
            lines.append(f"{model.upper():<15}" + "".join(f"{v:>18}" for v in vals))
        _write_table(tables_dir, "q1_necessity.txt", lines)

    # Q5: Effectiveness
    q5 = results.get("q5_effectiveness", {})
    if q5:
        nc = q5.get("node_classification", [])
        lp = q5.get("link_prediction", [])
        cl = q5.get("node_clustering", [])

        lines = ["Q5: Model Effectiveness Comparison", "=" * 100]

        # NC table
        if nc:
            lines.append("\nNode Classification (Accuracy% / F1%)")
            lines.append(f"{'Model':<15} {'Acc':>12} {'F1':>12}")
            lines.append("-" * 50)
            for r in nc:
                lines.append(f"{r['model'].upper():<15} {r['acc_mean']:>12.2f} {r['f1_mean']:>12.2f}")

        # LP table
        if lp:
            lines.append("\nLink Prediction")
            lines.append(f"{'Model':<15} {'MRR':>12} {'Hits@3':>12}")
            lines.append("-" * 50)
            for r in lp:
                mrr = r.get("mrr_mean", 0)
                h3 = r.get("hits@3_mean", 0)
                lines.append(f"{r['model'].upper():<15} {mrr:>12.4f} {h3:>12.4f}")

        # Clustering table
        if cl:
            lines.append("\nNode Clustering")
            lines.append(f"{'Model':<15} {'NMI':>12} {'ARI':>12}")
            lines.append("-" * 50)
            for r in cl:
                lines.append(f"{r['model'].upper():<15} {r['nmi_mean']:>12.4f} {r['ari_mean']:>12.4f}")

        _write_table(tables_dir, "q5_effectiveness.txt", lines)

    # Q3: Finetune
    q3 = results.get("q3_finetune", {}).get("results", [])
    if q3:
        lines = ["Q3: Frozen vs Trainable Encoder", "=" * 60]
        lines.append(f"{'Config':<25} {'Accuracy':>15}")
        lines.append("-" * 50)
        for r in q3:
            lines.append(f"{r['config']:<25} {r['test_acc_mean']:>12.4f}±{r['test_acc_std']:.4f}")
        _write_table(tables_dir, "q3_finetune.txt", lines)

    # Q6: PEFT
    q6 = results.get("q6_peft", {}).get("results", [])
    if q6:
        lines = ["Q6: PEFT Strategy Comparison", "=" * 60]
        lines.append(f"{'Strategy':<25} {'Accuracy':>15}")
        lines.append("-" * 50)
        for r in q6:
            lines.append(f"{r['peft_strategy']:<25} {r['test_acc_mean']:>12.4f}±{r['test_acc_std']:.4f}")
        _write_table(tables_dir, "q6_peft.txt", lines)

    # Q9-Q10: Efficiency
    q910 = results.get("q9_q10_efficiency", {})
    if q910:
        emp = q910.get("empirical_efficiency", [])
        theo = q910.get("theoretical_complexity", [])

        lines = ["Q9: Theoretical Complexity", "=" * 80]
        if theo:
            lines.append(f"{'Model':<20} {'Training':>25} {'Inference':>25} {'Memory':>25}")
            lines.append("-" * 100)
            for r in theo:
                lines.append(f"{r['model']:<20} {r['training']:>25} {r['inference']:>25} {r['memory']:>25}")

        lines.append("\nQ10: Empirical Efficiency")
        if emp:
            lines.append(f"{'Model':<20} {'Time(ms)':>12} {'Mem(MB)':>12} {'Params(M)':>12}")
            lines.append("-" * 60)
            for r in emp:
                if "error" not in r:
                    lines.append(f"{r['model']:<20} {r['time_per_epoch_ms']:>12.1f} "
                                 f"{r['peak_memory_mb']:>12.0f} {r['total_params']/1e6:>12.2f}")

        _write_table(tables_dir, "q9_q10_efficiency.txt", lines)

    # Q7: Robustness & Q8: Edge noise
    for qkey, title in [("q7_robustness", "Q7: Feature/Label Noise Robustness"),
                         ("q8_edge_noise", "Q8: Edge Noise Robustness")]:
        qdata = results.get(qkey, {})
        if qdata:
            lines = [f"{title}", "=" * 80]
            for noise_type, data in qdata.items():
                if isinstance(data, dict) and "results" in data:
                    lines.append(f"\n{noise_type.upper()} NOISE:")
                    lines.append(f"{'Ratio':>10} {'Accuracy':>15}")
                    lines.append("-" * 30)
                    for key in sorted(data["results"].keys()):
                        ratio = key.replace("ratio_", "")
                        r = data["results"][key]
                        lines.append(f"{ratio:>10} {r['mean']:>12.4f}±{r['std']:.4f}")
            _write_table(tables_dir, f"{qkey}.txt", lines)

    logger.info(f"OpenMAG表格已生成: {tables_dir}")


def _write_table(output_dir, filename, lines):
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"  表格: {path}")


if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "./benchmark/results"
    main(results_dir)
