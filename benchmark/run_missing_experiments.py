"""
运行所有新增实验 - OpenMAG benchmark 补充
按优先级执行：P0 → P1 → P2 → P3
"""
import os
import sys
import json
import time
import torch
import numpy as np
from loguru import logger

# 设置日志
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("./benchmark/logs/missing_experiments.log", level="DEBUG", rotation="100 MB")

RESULTS_DIR = "./benchmark/results"
DOCS_DIR = "/home/dev_qw/DataCrawlerAgent/docs"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs("./benchmark/logs", exist_ok=True)


def get_device(gpu_id=1):
    return f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"


# ===== P0: G2Text =====
def run_g2text_experiments(gpu_id=1):
    logger.info("=" * 60)
    logger.info("P0: G2Text Generation + PEFT Comparison")
    logger.info("=" * 60)

    try:
        from benchmark.tasks.g2text_generation import run_g2text_experiment
        from benchmark.data.dataset import AMMAGDataset

        for modality in ["text", "text+image+formula"]:
            logger.info(f"\n--- G2Text: modality={modality} ---")
            dataset = AMMAGDataset(
                root="./", domains=["cs_si", "cs_lo", "math_lo", "physics_socph"],
                modality=modality, label_type="domain"
            )
            result = run_g2text_experiment(
                dataset=dataset,
                data_dir="./data",
                gnn_backbones=["gcn", "gat", "graphsage"],
                peft_strategies=["none", "lora", "adapter", "ia3"],
                num_runs=3,
                device=get_device(gpu_id),
                output_dir=os.path.join(RESULTS_DIR, "g2text"),
            )
            logger.info(f"G2Text {modality} 完成")
    except Exception as e:
        logger.error(f"G2Text 实验失败: {e}")
        import traceback
        traceback.print_exc()


# ===== P0: G2Image =====
def run_g2image_experiments(gpu_id=2):
    logger.info("=" * 60)
    logger.info("P1: G2Image Generation + Neighbor Ablation")
    logger.info("=" * 60)

    try:
        from benchmark.tasks.g2image_generation import run_g2image_experiment
        from benchmark.data.dataset import AMMAGDataset

        for modality in ["text+image+formula"]:
            logger.info(f"\n--- G2Image: modality={modality} ---")
            dataset = AMMAGDataset(
                root="./", domains=["cs_si", "cs_lo", "math_lo", "physics_socph"],
                modality=modality, label_type="domain"
            )
            result = run_g2image_experiment(
                dataset=dataset,
                data_dir="./data",
                gnn_backbones=["gcn", "gat"],
                neighbor_counts=[0, 2, 4, 6, 8],
                num_runs=3,
                device=get_device(gpu_id),
                output_dir=os.path.join(RESULTS_DIR, "g2image"),
            )
            logger.info(f"G2Image {modality} 完成")
    except Exception as e:
        logger.error(f"G2Image 实验失败: {e}")
        import traceback
        traceback.print_exc()


# ===== P1: Encoder Comparison =====
def run_encoder_comparison(gpu_id=3):
    logger.info("=" * 60)
    logger.info("P1: Multi-Encoder Comparison")
    logger.info("=" * 60)

    try:
        from benchmark.tasks.encoder_comparison import run_encoder_comparison_experiment
        from benchmark.data.dataset import AMMAGDataset

        for modality in ["text"]:
            dataset = AMMAGDataset(
                root="./", domains=["cs_si", "cs_lo", "math_lo", "physics_socph"],
                modality=modality, label_type="domain"
            )
            result = run_encoder_comparison_experiment(
                dataset=dataset,
                data_dir="./data",
                encoder_names=["clip_large", "sentence_bert", "specter", "mathbert"],
                model_name="gcn",
                num_runs=3,
                device=get_device(gpu_id),
                output_dir=os.path.join(RESULTS_DIR, "encoder_comparison"),
            )
    except Exception as e:
        logger.error(f"编码器对比实验失败: {e}")
        import traceback
        traceback.print_exc()


# ===== P2: MLLM Models =====
def run_mllm_model_experiments(gpu_id=4):
    logger.info("=" * 60)
    logger.info("P2: MLLM-Enhanced Models (UniGraph2, Graph4MM, NTSFormer, MLaGA)")
    logger.info("=" * 60)

    try:
        from benchmark.tasks.node_classification import run_node_classification
        from benchmark.data.dataset import AMMAGDataset

        mllm_models = ["unigraph2", "graph4mm", "ntsformer", "mlaga", "graphmae2"]
        modalities = ["text", "text+image+formula"]

        for modality in modalities:
            dataset = AMMAGDataset(
                root="./", domains=["cs_si", "cs_lo", "math_lo", "physics_socph"],
                modality=modality, label_type="domain"
            )
            for model_name in mllm_models:
                logger.info(f"\n--- {model_name} + {modality} ---")
                try:
                    result = run_node_classification(
                        dataset=dataset,
                        model_name=model_name,
                        modality=modality,
                        num_runs=3,
                        device=get_device(gpu_id),
                        output_dir=os.path.join(RESULTS_DIR, "mllm_models"),
                    )
                except Exception as e:
                    logger.error(f"  {model_name} 失败: {e}")
    except Exception as e:
        logger.error(f"MLLM模型实验失败: {e}")
        import traceback
        traceback.print_exc()


# ===== P2: Modality Matching & Alignment =====
def run_modality_tasks(gpu_id=5):
    logger.info("=" * 60)
    logger.info("P2: Modality Matching & Alignment")
    logger.info("=" * 60)

    try:
        from benchmark.tasks.modality_matching import (
            run_modality_matching_experiment,
            run_modality_alignment_experiment,
        )
        from benchmark.data.dataset import AMMAGDataset

        dataset = AMMAGDataset(
            root="./", domains=["cs_si", "cs_lo", "math_lo", "physics_socph"],
            modality="text+image+formula", label_type="domain"
        )

        run_modality_matching_experiment(
            dataset=dataset, num_runs=3, device=get_device(gpu_id),
            output_dir=os.path.join(RESULTS_DIR, "modality_tasks"),
        )

        run_modality_alignment_experiment(
            dataset=dataset,
            output_dir=os.path.join(RESULTS_DIR, "modality_tasks"),
        )
    except Exception as e:
        logger.error(f"模态任务实验失败: {e}")
        import traceback
        traceback.print_exc()


# ===== P3: GraphMAE2 Self-Supervised =====
def run_graphmae2_pretrain(gpu_id=6):
    logger.info("=" * 60)
    logger.info("P3: GraphMAE2 Self-Supervised Pretraining + Linear Eval")
    logger.info("=" * 60)

    try:
        from benchmark.data.dataset import AMMAGDataset
        from benchmark.models.advanced_models import GraphMAE2
        import torch.nn.functional as F

        device = get_device(gpu_id)

        for modality in ["text", "text+image+formula"]:
            dataset = AMMAGDataset(
                root="./", domains=["cs_si", "cs_lo", "math_lo", "physics_socph"],
                modality=modality, label_type="domain"
            )
            data = dataset.data.to(device)
            in_dim = data.x.size(1)
            num_classes = int(data.y.max()) + 1

            # Phase 1: Pretrain
            logger.info(f"\n--- GraphMAE2 Pretrain: {modality} ---")
            model = GraphMAE2(in_dim, 256, num_classes, num_layers=2, dropout=0.5).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            for epoch in range(1, 501):
                model.train()
                optimizer.zero_grad()
                loss = model.pretrain_forward(data.x, data.edge_index)
                loss.backward()
                optimizer.step()

                if epoch % 50 == 0:
                    logger.info(f"  Epoch {epoch}: pretrain_loss={loss.item():.4f}")

            # Phase 2: Linear evaluation (freeze encoder)
            train_mask, val_mask, test_mask = dataset.get_split_masks()
            model.eval()
            with torch.no_grad():
                h = model.encode(data.x, data.edge_index)

            # Train linear classifier on frozen features
            classifier = torch.nn.Linear(256, num_classes).to(device)
            opt_clf = torch.optim.Adam(classifier.parameters(), lr=0.01)

            best_val_acc = 0
            best_test_acc = 0

            for epoch in range(1, 301):
                classifier.train()
                opt_clf.zero_grad()
                out = classifier(h)
                loss = F.cross_entropy(out[train_mask], data.y[train_mask])
                loss.backward()
                opt_clf.step()

                if epoch % 10 == 0:
                    classifier.eval()
                    with torch.no_grad():
                        pred = classifier(h).argmax(dim=1)
                        val_acc = (pred[val_mask] == data.y[val_mask]).float().mean().item()
                        test_acc = (pred[test_mask] == data.y[test_mask]).float().mean().item()
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            best_test_acc = test_acc

            logger.info(f"GraphMAE2 {modality}: linear_eval_acc={best_test_acc:.4f}")

            # Save result
            result = {
                "model": "graphmae2",
                "modality": modality,
                "task": "self_supervised_linear_eval",
                "pretrain_epochs": 500,
                "linear_eval_accuracy": best_test_acc,
            }
            save_path = os.path.join(RESULTS_DIR, "mllm_models", f"graphmae2_{modality}_linear_eval.json")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(result, f, indent=2)

    except Exception as e:
        logger.error(f"GraphMAE2实验失败: {e}")
        import traceback
        traceback.print_exc()


# ===== Re-run fixed Link Prediction =====
def run_fixed_link_prediction(gpu_id=7):
    logger.info("=" * 60)
    logger.info("Fix: Re-run Link Prediction with strict negative sampling")
    logger.info("=" * 60)

    try:
        from benchmark.tasks.link_prediction import run_link_prediction_ablation
        run_link_prediction_ablation(
            root="./",
            domains=["cs_si", "cs_lo", "math_lo", "physics_socph"],
            num_runs=3,
            device=get_device(gpu_id),
            output_dir=os.path.join(RESULTS_DIR, "lp_ablation_fixed_v2"),
        )
    except Exception as e:
        logger.error(f"修复后链接预测实验失败: {e}")
        import traceback
        traceback.print_exc()


def run_fixed_cross_domain_lp(gpu_id=7):
    logger.info("=" * 60)
    logger.info("Fix: Re-run Cross-Domain Link Prediction")
    logger.info("=" * 60)

    try:
        from benchmark.tasks.cross_domain_link_prediction import run_cross_domain_link_prediction
        from benchmark.data.dataset import AMMAGDataset

        for modality in ["text", "text+image+formula", "image+formula"]:
            dataset = AMMAGDataset(
                root="./", domains=["cs_si", "cs_lo", "math_lo", "physics_socph"],
                modality=modality, label_type="domain"
            )
            run_cross_domain_link_prediction(
                dataset=dataset,
                model_names=["gcn", "gat", "graphsage", "gt"],
                modalities=[modality],
                num_runs=3,
                device=get_device(gpu_id),
                output_dir=os.path.join(RESULTS_DIR, "cross_domain_fixed_v2"),
            )
    except Exception as e:
        logger.error(f"修复后跨域LP失败: {e}")
        import traceback
        traceback.print_exc()


# ===== Aggregate all results =====
def aggregate_all_results():
    """收集所有实验结果并汇总到docs目录"""
    logger.info("=" * 60)
    logger.info("汇总所有实验结果")
    logger.info("=" * 60)

    all_results = {}

    # 遍历所有结果目录
    result_dirs = [
        "g2text", "g2image", "encoder_comparison",
        "mllm_models", "modality_tasks",
        "lp_ablation_fixed_v2", "cross_domain_fixed_v2",
    ]

    for subdir in result_dirs:
        dir_path = os.path.join(RESULTS_DIR, subdir)
        if not os.path.exists(dir_path):
            logger.warning(f"结果目录不存在: {dir_path}")
            continue

        for fname in os.listdir(dir_path):
            if fname.endswith(".json"):
                fpath = os.path.join(dir_path, fname)
                try:
                    with open(fpath) as f:
                        data = json.load(f)
                    all_results[f"{subdir}/{fname}"] = data
                except Exception as e:
                    logger.warning(f"读取 {fpath} 失败: {e}")

    # 保存汇总
    summary_path = os.path.join(DOCS_DIR, "all_experiment_results.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"所有实验结果汇总到 {summary_path}")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "g2text", "g2image", "encoder", "mllm",
                                 "modality", "graphmae2", "lp_fix", "cross_lp_fix",
                                 "aggregate"],
                        help="要运行的实验阶段")
    parser.add_argument("--gpu", type=int, default=1, help="GPU ID")
    args = parser.parse_args()

    start_time = time.time()

    if args.phase == "all":
        # 依次运行所有实验（单GPU）
        run_g2text_experiments(args.gpu)
        run_g2image_experiments(args.gpu)
        run_encoder_comparison(args.gpu)
        run_mllm_model_experiments(args.gpu)
        run_modality_tasks(args.gpu)
        run_graphmae2_pretrain(args.gpu)
        run_fixed_link_prediction(args.gpu)
        run_fixed_cross_domain_lp(args.gpu)
        aggregate_all_results()
    elif args.phase == "g2text":
        run_g2text_experiments(args.gpu)
    elif args.phase == "g2image":
        run_g2image_experiments(args.gpu)
    elif args.phase == "encoder":
        run_encoder_comparison(args.gpu)
    elif args.phase == "mllm":
        run_mllm_model_experiments(args.gpu)
    elif args.phase == "modality":
        run_modality_tasks(args.gpu)
    elif args.phase == "graphmae2":
        run_graphmae2_pretrain(args.gpu)
    elif args.phase == "lp_fix":
        run_fixed_link_prediction(args.gpu)
    elif args.phase == "cross_lp_fix":
        run_fixed_cross_domain_lp(args.gpu)
    elif args.phase == "aggregate":
        aggregate_all_results()

    elapsed = time.time() - start_time
    logger.info(f"\n实验总用时: {elapsed/3600:.1f} 小时")
