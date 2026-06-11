"""
全面实验运行脚本 - 分配到8个GPU并行运行所有实验
用法:
  # 阶段1: 多模态节点分类 + 消融 (GPU 0-5)
  python benchmark/run_all_experiments.py --phase nc_multimodal

  # 阶段2: 链接预测 + Few-shot (GPU 6-7)
  python benchmark/run_all_experiments.py --phase lp_fewshot

  # 阶段3: 跨域迁移 (GPU 0-1)
  python benchmark/run_all_experiments.py --phase cross_domain

  # 阶段4: 子分类分类 (GPU 0-3)
  python benchmark/run_all_experiments.py --phase subcat_nc

  # 阶段5: 汇总
  python benchmark/run_all_experiments.py --phase aggregate
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import sys
import json
import time
import torch
import numpy as np
from loguru import logger
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "benchmark", "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "benchmark", "results")

MODELS = ["mlp", "gcn", "gat", "graphsage", "gin", "gt"]
MODALITIES_7 = ["text", "image", "formula", "text+image", "text+formula",
                "image+formula", "text+image+formula"]
DOMAINS = ["cs_si", "cs_lo", "math_lo", "physics_socph"]

logger.remove()
logger.add(sys.stderr, level="INFO")


def verify_embeddings():
    """验证所有嵌入文件都存在且大小正确"""
    for mod in ["text", "image", "formula"]:
        path = os.path.join(PROCESSED_DIR, f"{mod}_embeddings.pt")
        if not os.path.exists(path):
            logger.error(f"缺少 {mod} embeddings: {path}")
            return False
        data = torch.load(path, map_location="cpu")
        n = len(data["ids"])
        d = data["embeddings"].shape[1]
        logger.info(f"  {mod}: {n} nodes, {d}d")
        if n < 40000:
            logger.error(f"  {mod} embeddings 不完整 (期望~45672)")
            return False
    return True


def get_dataset(modality, label_type="domain"):
    from benchmark.data.dataset import AMMAGDataset
    return AMMAGDataset(root=PROJECT_ROOT, modality=modality, label_type=label_type)


# ============================================================
# Phase 1: 多模态节点分类 (6模型 × 6剩余模态 = 36组, 5 runs each)
# ============================================================

def run_nc_multimodal(models=None, modalities=None, gpu=0):
    """运行多模态节点分类实验"""
    from benchmark.tasks.node_classification import run_node_classification

    if models is None:
        models = MODELS
    if modalities is None:
        modalities = [m for m in MODALITIES_7 if m != "text"]  # text已做

    device = f"cuda:{gpu}"

    for modality in modalities:
        logger.info(f"=== 加载 {modality} 数据集 ===")
        try:
            ds = get_dataset(modality, label_type="domain")
        except Exception as e:
            logger.error(f"加载 {modality} 失败: {e}")
            continue

        for model_name in models:
            # 检查是否已有结果
            result_path = os.path.join(RESULTS_DIR, f"nc_{model_name}_{modality}_supervised.json")
            if os.path.exists(result_path):
                # 检查是否是新的完整结果
                try:
                    old = json.load(open(result_path))
                    if old.get("num_runs", 0) >= 5:
                        logger.info(f"跳过已有结果: {model_name}+{modality}")
                        continue
                except Exception:
                    pass

            logger.info(f"--- NC: {model_name}+{modality} ---")
            try:
                run_node_classification(
                    dataset=ds, model_name=model_name, modality=modality,
                    num_runs=5, num_epochs=500, hidden_dim=256, num_layers=2,
                    dropout=0.5, lr=0.01, device=device,
                    domain_labels=ds._domain_labels,
                    output_dir=RESULTS_DIR,
                )
            except Exception as e:
                logger.error(f"NC实验失败 {model_name}+{modality}: {e}")
                import traceback
                traceback.print_exc()


def run_per_domain_ablation(gpu=0):
    """每域模态消融 (GCN × 7 modalities) - 核心创新实验"""
    from benchmark.tasks.node_classification import run_node_classification

    device = f"cuda:{gpu}"

    for modality in MODALITIES_7:
        result_path = os.path.join(RESULTS_DIR, f"ablation_pd_gcn_{modality}_supervised.json")
        if os.path.exists(result_path):
            try:
                old = json.load(open(result_path))
                if old.get("num_runs", 0) >= 5 and "per_domain_acc" in old:
                    logger.info(f"跳过已有消融结果: gcn+{modality}")
                    continue
            except Exception:
                pass

        logger.info(f"=== 消融: GCN+{modality} ===")
        try:
            ds = get_dataset(modality, label_type="domain")
            run_node_classification(
                dataset=ds, model_name="gcn", modality=modality,
                num_runs=5, num_epochs=500, hidden_dim=256, num_layers=2,
                dropout=0.5, lr=0.01, device=device,
                domain_labels=ds._domain_labels,
                output_dir=RESULTS_DIR, save_prefix="ablation_pd",
            )
        except Exception as e:
            logger.error(f"消融实验失败 gcn+{modality}: {e}")


# ============================================================
# Phase 2: 链接预测 (4模型 × 3模态 = 12组)
# ============================================================

def run_link_prediction(models=None, gpu=0):
    from benchmark.tasks.link_prediction import run_link_prediction

    if models is None:
        models = ["gcn", "graphsage", "gat", "gt"]
    lp_modalities = ["text", "text+image+formula", "image+formula"]
    device = f"cuda:{gpu}"

    for modality in lp_modalities:
        try:
            ds = get_dataset(modality)
        except Exception as e:
            logger.error(f"LP: 加载 {modality} 失败: {e}")
            continue

        for model_name in models:
            result_path = os.path.join(RESULTS_DIR, f"lp_{model_name}_{modality}.json")
            if os.path.exists(result_path):
                try:
                    old = json.load(open(result_path))
                    if old.get("num_runs", 0) >= 5:
                        logger.info(f"跳过已有LP结果: {model_name}+{modality}")
                        continue
                except Exception:
                    pass

            logger.info(f"--- LP: {model_name}+{modality} ---")
            try:
                run_link_prediction(
                    dataset=ds, model_name=model_name, modality=modality,
                    num_runs=5, num_epochs=500, hidden_dim=256, num_layers=2,
                    dropout=0.5, lr=0.01, device=device,
                    output_dir=RESULTS_DIR,
                )
            except Exception as e:
                logger.error(f"LP实验失败 {model_name}+{modality}: {e}")
                import traceback
                traceback.print_exc()


# ============================================================
# Phase 3: Few-shot (5模型 × 2模态 × 2 k = 20组)
# ============================================================

def run_few_shot(models=None, gpu=0):
    from benchmark.tasks.node_classification import run_node_classification

    if models is None:
        models = ["gcn", "gat", "graphsage", "gin", "gt"]
    fs_modalities = ["text", "text+image+formula"]
    ks = [3, 10]
    device = f"cuda:{gpu}"

    for modality in fs_modalities:
        try:
            ds = get_dataset(modality)
        except Exception as e:
            logger.error(f"FS: 加载 {modality} 失败: {e}")
            continue

        for model_name in models:
            for k in ks:
                result_path = os.path.join(RESULTS_DIR,
                    f"nc_{model_name}_{modality}_few_shot_k{k}.json")
                if os.path.exists(result_path):
                    try:
                        old = json.load(open(result_path))
                        if old.get("num_runs", 0) >= 10:
                            logger.info(f"跳过已有FS结果: {model_name}+{modality}+k{k}")
                            continue
                    except Exception:
                        pass

                logger.info(f"--- FS: {model_name}+{modality}+k{k} ---")
                try:
                    run_node_classification(
                        dataset=ds, model_name=model_name, modality=modality,
                        num_runs=10, num_epochs=500, hidden_dim=256, num_layers=2,
                        dropout=0.5, lr=0.01, device=device,
                        split="few_shot", few_shot_k=k,
                        domain_labels=ds._domain_labels,
                        output_dir=RESULTS_DIR,
                    )
                except Exception as e:
                    logger.error(f"FS实验失败 {model_name}+{modality}+k{k}: {e}")


# ============================================================
# Phase 4: 跨域迁移 (12 pairs × 2 modalities)
# ============================================================

def run_cross_domain(gpu=0):
    from benchmark.tasks.cross_domain import run_cross_domain_transfer

    cd_modalities = ["text", "text+image+formula"]
    device = f"cuda:{gpu}"

    for modality in cd_modalities:
        try:
            ds = get_dataset(modality)
        except Exception as e:
            logger.error(f"CD: 加载 {modality} 失败: {e}")
            continue

        for src in DOMAINS:
            for tgt in DOMAINS:
                if src == tgt:
                    continue

                result_path = os.path.join(RESULTS_DIR,
                    f"cd_{src}_to_{tgt}_gcn_{modality}.json")
                if os.path.exists(result_path):
                    try:
                        old = json.load(open(result_path))
                        if old.get("num_runs", 0) >= 3:
                            logger.info(f"跳过已有CD结果: {src}→{tgt}+{modality}")
                            continue
                    except Exception:
                        pass

                logger.info(f"--- CD: {src}→{tgt}, gcn+{modality} ---")
                try:
                    run_cross_domain_transfer(
                        dataset=ds, model_name="gcn", modality=modality,
                        source_domain=src, target_domain=tgt,
                        num_runs=3, num_epochs=500, hidden_dim=256,
                        num_layers=2, dropout=0.5, lr=0.01,
                        device=device, output_dir=RESULTS_DIR,
                    )
                except Exception as e:
                    logger.error(f"CD实验失败 {src}→{tgt}: {e}")


# ============================================================
# Phase 5: 子分类分类 (4域 × 3模型 × 3模态)
# ============================================================

def run_subcat_nc(gpu=0):
    from benchmark.tasks.node_classification import run_node_classification
    from benchmark.data.dataset import AMMAGDataset

    subcat_models = ["gcn", "gat", "graphsage"]
    subcat_mods = ["text", "text+image+formula", "formula"]
    device = f"cuda:{gpu}"

    for domain in DOMAINS:
        for modality in subcat_mods:
            try:
                ds = AMMAGDataset(
                    root=PROJECT_ROOT, modality=modality, label_type="subcat"
                )
                subgraph = ds.get_domain_subgraph(domain)
                n_classes = int(subgraph.y.max().item()) + 1
                logger.info(f"子分类 {domain}: {subgraph.num_nodes} nodes, "
                           f"{subgraph.num_edges} edges, {n_classes} classes")
            except Exception as e:
                logger.error(f"子分类数据加载失败 {domain}+{modality}: {e}")
                continue

            for model_name in subcat_models:
                result_path = os.path.join(RESULTS_DIR,
                    f"subcat_nc_{domain}_{model_name}_{modality}_supervised.json")
                if os.path.exists(result_path):
                    try:
                        old = json.load(open(result_path))
                        if old.get("num_runs", 0) >= 5:
                            continue
                    except Exception:
                        pass

                logger.info(f"--- Subcat: {domain}+{model_name}+{modality} ---")
                try:
                    # 直接在子图上训练
                    from benchmark.models.gnn import get_model
                    from benchmark.tasks.node_classification import train_and_eval

                    all_results = []
                    for run in range(5):
                        seed = run * 42 + 1
                        sub_data = subgraph.to(device)
                        n = sub_data.num_nodes
                        rng = np.random.RandomState(seed)
                        perm = rng.permutation(n)
                        train_size = int(n * 0.6)
                        val_size = int(n * 0.2)

                        train_mask = torch.zeros(n, dtype=torch.bool)
                        val_mask = torch.zeros(n, dtype=torch.bool)
                        test_mask = torch.zeros(n, dtype=torch.bool)
                        train_mask[perm[:train_size]] = True
                        val_mask[perm[train_size:train_size+val_size]] = True
                        test_mask[perm[train_size+val_size:]] = True

                        model = get_model(
                            model_name, sub_data.x.size(1), 256, n_classes,
                            num_layers=2, dropout=0.5
                        )

                        result = train_and_eval(
                            model, sub_data, train_mask, val_mask, test_mask,
                            num_epochs=500, lr=0.01, device=device,
                        )
                        all_results.append(result)
                        logger.info(f"  Run {run+1}/5: acc={result['test_acc']:.4f}")

                    # 汇总
                    metrics = ["test_acc", "test_f1", "val_acc", "training_time"]
                    summary = {}
                    for m in metrics:
                        vals = [r[m] for r in all_results]
                        summary[f"{m}_mean"] = float(np.mean(vals))
                        summary[f"{m}_std"] = float(np.std(vals))
                    summary["num_runs"] = 5
                    summary["model"] = model_name
                    summary["modality"] = modality
                    summary["domain"] = domain
                    summary["task"] = "subcat_nc"
                    summary["runs"] = all_results

                    with open(result_path, "w") as f:
                        json.dump(summary, f, indent=2, default=str)
                    logger.info(f"子分类结果: {domain}+{model_name}+{modality}, "
                               f"acc={summary['test_acc_mean']:.4f}")

                except Exception as e:
                    logger.error(f"子分类实验失败: {e}")
                    import traceback
                    traceback.print_exc()


# ============================================================
# Phase 6: 汇总表格
# ============================================================

def run_aggregate():
    from benchmark.aggregate_results import main as aggregate_main
    aggregate_main(RESULTS_DIR)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True,
        choices=["verify", "nc_multimodal", "per_domain_ablation",
                 "link_prediction", "few_shot", "cross_domain",
                 "subcat_nc", "aggregate"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--modalities", nargs="+", default=None)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.phase == "verify":
        verify_embeddings()
    elif args.phase == "nc_multimodal":
        run_nc_multimodal(args.models, args.modalities, args.gpu)
    elif args.phase == "per_domain_ablation":
        run_per_domain_ablation(args.gpu)
    elif args.phase == "link_prediction":
        run_link_prediction(args.models, args.gpu)
    elif args.phase == "few_shot":
        run_few_shot(args.models, args.gpu)
    elif args.phase == "cross_domain":
        run_cross_domain(args.gpu)
    elif args.phase == "subcat_nc":
        run_subcat_nc(args.gpu)
    elif args.phase == "aggregate":
        run_aggregate()

    logger.info("Done!")
