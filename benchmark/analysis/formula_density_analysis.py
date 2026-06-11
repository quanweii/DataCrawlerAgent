"""
Formula密度专项分析 - AMMAG独有的卖点

核心思路：按论文的公式密度分层，量化Formula模态在不同密度区间的贡献
如果Formula密度高的论文 T+F 显著优于 T，这就是一个很有说服力的发现
"""
import os
import json
import argparse
import numpy as np
import torch
from loguru import logger


def load_formula_counts(processed_dir: str) -> dict:
    """加载每篇论文的公式计数"""
    stats_path = os.path.join(processed_dir, "statistics.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            return json.load(f)
    return {}


def compute_formula_density_analysis(
    processed_dir: str,
    results_dir: str,
    output_dir: str,
    device: str = "cuda",
):
    """
    Formula密度专项分析：
    1. 加载每篇论文的公式计数（从markdown统计）
    2. 按公式密度将论文分为低/中/高三组
    3. 在每组上分别评估 Text vs T+F 的性能
    4. 量化 Formula 在不同密度区间的增益
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from benchmark.data.dataset import AMMAGDataset
    from benchmark.models.gnn import get_model
    from benchmark.tasks.node_classification import train_and_eval

    os.makedirs(output_dir, exist_ok=True)

    # 加载数据集
    domains = ["cs_si", "cs_lo", "math_lo", "physics_socph"]

    # Step 1: 收集每篇论文的公式计数
    formula_counts = {}
    md_dir = os.path.join(processed_dir, "..", "output")

    # 从statistics.json读取domain信息
    paper_domains = {}  # paper_id -> domain
    paper_formula_counts = {}  # paper_id -> formula_count

    stats_path = os.path.join(processed_dir, "statistics.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        for domain, ds in stats.get("domain_stats", {}).items():
            for pid in ds.get("paper_ids", []):
                paper_domains[pid] = domain

    # 从markdown文件统计公式数量
    # 或者从已有的embedding文件推断
    # 这里用简化方法：从per-domain statistics估算

    # Step 2: 使用更直接的方法 - 加载完整图，从node特征推断domain
    graph_path = os.path.join(processed_dir, "graph_data.pt")
    if not os.path.exists(graph_path):
        logger.error("找不到graph_data.pt，请先运行预处理")
        return

    data = torch.load(graph_path, map_location="cpu")
    paper_ids = data.get("paper_ids", [])
    labels = data["labels"].numpy()
    domain_labels = data.get("domain_labels", data["labels"]).numpy()

    # Step 3: 加载text和text+formula embedding
    text_emb_path = os.path.join(processed_dir, "text_embeddings.pt")
    tf_emb_path = os.path.join(processed_dir, "multimodal_text+formula_embeddings.pt")

    if not os.path.exists(text_emb_path) or not os.path.exists(tf_emb_path):
        logger.error("找不到text或text+formula embedding，请先运行编码")
        return

    text_emb = torch.load(text_emb_path, map_location="cpu")
    tf_emb = torch.load(tf_emb_path, map_location="cpu")

    # Step 4: 加载formula embedding计算密度指标
    formula_emb_path = os.path.join(processed_dir, "formula_embeddings.pt")
    if os.path.exists(formula_emb_path):
        formula_emb = torch.load(formula_emb_path, map_location="cpu")
        formula_norms = torch.norm(formula_emb["embeddings"], dim=1).numpy()
        # 用formula embedding的L2范数作为公式信息量的代理指标
        # 零向量表示无公式，范数越高表示公式越多/越丰富
    else:
        logger.warning("找不到formula embedding")
        return

    # Step 5: 按formula norm分组
    # 分三组：低（底部33%）、中（33-66%）、高（顶部33%）
    nonzero_mask = formula_norms > 1e-6
    nonzero_norms = formula_norms[nonzero_mask]

    if len(nonzero_norms) == 0:
        logger.error("所有论文的formula embedding为零")
        return

    q33 = np.percentile(nonzero_norms, 33)
    q66 = np.percentile(nonzero_norms, 66)

    logger.info(f"Formula L2范数分布: Q33={q33:.4f}, Q66={q66:.4f}")
    logger.info(f"非零论文数: {nonzero_mask.sum()}, 总论文数: {len(formula_norms)}")

    # Step 6: 对每个domain分别分析
    domain_names = {0: "cs.SI", 1: "cs.LO", 2: "math.LO", 3: "physics.soc-ph"}
    results = {}

    for modality_combo, emb_data in [("text", text_emb), ("text+formula", tf_emb)]:
        emb = emb_data["embeddings"]
        emb_ids = emb_data["ids"]

        for domain_idx in range(4):
            domain_name = domain_names[domain_idx]
            mask = domain_labels == domain_idx
            indices = np.where(mask)[0]

            if len(indices) < 100:
                continue

            # 按formula密度分组
            for group_name, lo, hi in [
                ("low", 0, q33),
                ("medium", q33, q66),
                ("high", q66, float("inf")),
            ]:
                group_mask = (formula_norms >= lo) & (formula_norms < hi) & mask
                group_indices = np.where(group_mask)[0]

                if len(group_indices) < 50:
                    continue

                key = f"{domain_name}_{group_name}_{modality_combo}"
                results[key] = {
                    "domain": domain_name,
                    "formula_density": group_name,
                    "modality": modality_combo,
                    "num_papers": len(group_indices),
                    "avg_formula_norm": float(formula_norms[group_indices].mean()),
                }

    # Step 7: 在分组数据上训练GCN，比较Text vs T+F
    # 为每组创建子数据集并训练
    num_nodes = len(labels)
    num_classes = int(labels.max()) + 1

    all_results = {}
    for density_group in ["low", "medium", "high"]:
        group_mask = np.zeros(num_nodes, dtype=bool)
        if density_group == "low":
            group_mask = (formula_norms > 1e-6) & (formula_norms < q33)
        elif density_group == "medium":
            group_mask = (formula_norms >= q33) & (formula_norms < q66)
        else:
            group_mask = formula_norms >= q66

        group_indices = np.where(group_mask)[0]
        if len(group_indices) < 100:
            continue

        logger.info(f"\n=== Formula密度组: {density_group} ({len(group_indices)}篇) ===")

        for mod_name, emb_path in [
            ("text", text_emb_path),
            ("text+formula", tf_emb_path),
        ]:
            emb_data = torch.load(emb_path, map_location="cpu")
            x = emb_data["embeddings"]

            # 构建PyG Data
            from torch_geometric.data import Data
            graph_data = Data(
                x=x,
                edge_index=data["edge_index"],
                y=data["labels"],
            )

            # 只在group内的节点上评估
            rng = np.random.RandomState(42)
            perm = rng.permutation(group_indices)
            n = len(perm)
            t = int(n * 0.6)
            v = int(n * 0.2)

            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            train_mask[perm[:t]] = True
            val_mask[perm[t:t + v]] = True
            test_mask[perm[t + v:]] = True

            # 训练
            in_dim = x.size(1)
            model = get_model("gcn", in_dim, 256, num_classes, num_layers=2, dropout=0.5)

            result = train_and_eval(
                model, graph_data, train_mask, val_mask, test_mask,
                num_epochs=500, lr=0.01, device=device,
            )

            key = f"{density_group}_{mod_name}"
            all_results[key] = {
                "density_group": density_group,
                "modality": mod_name,
                "num_papers": len(group_indices),
                "test_acc": result["test_acc"],
                "test_f1": result["test_f1"],
                "avg_formula_norm": float(formula_norms[group_mask].mean()),
            }

            logger.info(f"  {mod_name}: acc={result['test_acc']:.4f}, "
                        f"f1={result['test_f1']:.4f}")

    # Step 8: 计算Formula增益（T+F - Text）
    report = []
    report.append("=" * 70)
    report.append("Formula密度专项分析报告")
    report.append("=" * 70)
    report.append(f"\nFormula L2范数分组阈值: 低<{q33:.3f}, 中{q33:.3f}-{q66:.3f}, 高>{q66:.3f}")
    report.append(f"\n{'密度组':<10} {'论文数':>8} {'Text':>8} {'T+F':>8} {'Δ(pp)':>8} {'F-norm':>8}")
    report.append("-" * 55)

    for group in ["low", "medium", "high"]:
        text_key = f"{group}_text"
        tf_key = f"{group}_text+formula"
        if text_key in all_results and tf_key in all_results:
            t_acc = all_results[text_key]["test_acc"]
            f_acc = all_results[tf_key]["test_acc"]
            delta = (f_acc - t_acc) * 100
            n = all_results[text_key]["num_papers"]
            fnorm = all_results[text_key]["avg_formula_norm"]
            report.append(
                f"{group:<10} {n:>8} {t_acc*100:>7.2f}% {f_acc*100:>7.2f}% "
                f"{delta:>+7.2f} {fnorm:>8.4f}"
            )

    report_text = "\n".join(report)
    print(report_text)

    save_path = os.path.join(output_dir, "formula_density_analysis.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "formula_density_report.txt"), "w") as f:
        f.write(report_text)

    logger.info(f"Formula密度分析保存到 {output_dir}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="./benchmark/processed")
    parser.add_argument("--results_dir", default="./benchmark/results")
    parser.add_argument("--output_dir", default="./benchmark/analysis/output")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    compute_formula_density_analysis(
        args.processed_dir, args.results_dir, args.output_dir, device
    )
