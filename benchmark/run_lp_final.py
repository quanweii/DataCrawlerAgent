#!/usr/bin/env python3
"""
AMMAG 链接预测 7模态 × 9模型 全量消融（统一配方 + 稳定性守卫）。

统一配方（"统一的lr + 其他措施"）：
  lr=0.01, neg_ratio=10, 同域严格负采样, 无投影(proj_dim=None),
  仅训练边消息传递(无泄露), 梯度裁剪(max_norm=1.0)。

稳定性守卫：
  每个单元格跑 5 个 seed（seed=run*42+1，与 run_link_prediction 一致）。
  若某 seed 的 training_time<30s 视为发散/崩溃（healthy run 通常 50-150s）。
  仅对 healthy runs 求均值±标准差。
  若 clip=True 下 healthy<3，则用 clip=False 重跑 5 seed，取 healthy 更多的方案，
  并在结果中标注 clip_used=False（RevGAT 等可逆架构在高维输入下裁剪致欠拟合）。

用法（每模态一个进程，并行 7 张卡）：
  python benchmark/run_lp_final.py <modality> <gpu> [outdir]
例：
  python benchmark/run_lp_final.py text 0 &
  python benchmark/run_lp_final.py text+image+formula 6 &
"""
import os
import sys
import json
import time

MODALITY = sys.argv[1]
GPU = sys.argv[2] if len(sys.argv) > 2 else "0"
OUTDIR = sys.argv[3] if len(sys.argv) > 3 else "./benchmark/results/lp_ablation_final"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from loguru import logger

from benchmark.data.dataset import AMMAGDataset
from benchmark.models.gnn import get_model
from benchmark.models.link_predictor import DotProductDecoder
from benchmark.tasks.link_prediction import (
    train_and_eval_link_prediction, negative_sampling_strict,
)

MODELS_9 = ["mlp", "gcn", "gat", "graphsage", "gin", "gt", "gatv2", "gcnii", "revgat"]
DOMAINS = ["cs_si", "cs_lo", "math_lo", "physics_socph"]
N_SEEDS = 3              # 与论文/节点分类一致：3次运行（崩溃守卫下可追加 clip=False 的3次）
HEALTHY_MIN_TIME = 30.0   # training_time<30s => 崩溃/发散
MIN_HEALTHY = 2           # healthy<2 时启用 clip=False 重试（覆盖 RevGAT 等可逆架构）


def run_seeds(model_name, data, in_dim, domain_labels, edge_index, num_edges,
              n_seeds=N_SEEDS, clip=True, lr=0.01):
    results = []
    for run in range(n_seeds):
        seed = run * 42 + 1
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(num_edges, generator=g)
        tr = int(num_edges * 0.6)
        va = int(num_edges * 0.2)
        train_edges = edge_index[:, perm[:tr]]
        val_pos = edge_index[:, perm[tr:tr + va]]
        test_pos = edge_index[:, perm[tr + va:]]
        all_pos = torch.cat([train_edges, val_pos, test_pos], dim=1)
        val_neg = negative_sampling_strict(
            all_pos, data.num_nodes, val_pos.size(1) * 10, seed=seed + 1000, labels=domain_labels)
        test_neg = negative_sampling_strict(
            all_pos, data.num_nodes, test_pos.size(1) * 10, seed=seed + 2000, labels=domain_labels)
        enc = get_model(model_name, in_dim, 256, 256, num_layers=2, dropout=0.5, proj_dim=None)
        dec = DotProductDecoder()
        try:
            r = train_and_eval_link_prediction(
                enc, dec, data, train_edges, val_pos, val_neg, test_pos, test_neg,
                num_epochs=500, lr=lr, device="cuda", labels=domain_labels, clip_grad=clip,
            )
            r["seed"] = seed
            r["clip_grad"] = clip
            r["crash"] = r.get("training_time", 0) < HEALTHY_MIN_TIME
            results.append(r)
        except Exception as e:  # NaN / 梯度爆炸等
            logger.error(f"[{model_name}+{MODALITY}] seed={seed} clip={clip} EXCEPTION: {e}")
            results.append({
                "mrr": 0.0, "hits@1": 0.0, "hits@3": 0.0, "hits@10": 0.0,
                "training_time": 0.0, "seed": seed, "clip_grad": clip,
                "crash": True, "error": str(e)[:200],
            })
        logger.info(f"[{model_name}+{MODALITY}] seed={seed} clip={clip} "
                    f"MRR={results[-1].get('mrr', 0):.4f} tt={results[-1].get('training_time', 0):.1f}s "
                    f"crash={results[-1].get('crash')}")
    return results


def aggregate(runs):
    healthy = [r for r in runs if not r.get("crash")]
    use = healthy if healthy else runs  # 全崩则用全部（标注）
    metrics = ["mrr", "hits@1", "hits@3", "hits@10", "training_time"]
    s = {}
    for m in metrics:
        vals = [r[m] for r in use]
        s[f"{m}_mean"] = float(np.mean(vals))
        s[f"{m}_std"] = float(np.std(vals))
    s["num_runs"] = len(runs)
    s["num_healthy"] = len(healthy)
    s["all_crash"] = (len(healthy) == 0)
    s["runs"] = runs
    return s


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    logger.info(f"=== LP final: modality={MODALITY} gpu={GPU} outdir={OUTDIR} ===")

    ds = AMMAGDataset(root=".", domains=DOMAINS, modality=MODALITY, label_type="domain")
    data = ds.data
    in_dim = data.x.size(1)
    domain_labels = ds._domain_labels.cpu()
    edge_index = data.edge_index.cpu()
    num_edges = edge_index.size(1)
    logger.info(f"[{MODALITY}] dataset loaded, in_dim={in_dim}, num_edges={num_edges}")

    for model_name in MODELS_9:
        t0 = time.time()
        # 默认 clip=True
        runs = run_seeds(model_name, data, in_dim, domain_labels, edge_index, num_edges,
                         n_seeds=N_SEEDS, clip=True)
        n_healthy = sum(1 for r in runs if not r.get("crash"))
        clip_used = True

        # 若 clip=True 下 healthy 不足，尝试 clip=False
        if n_healthy < MIN_HEALTHY:
            logger.warning(f"[{model_name}+{MODALITY}] clip=True 仅 {n_healthy}/{N_SEEDS} healthy，"
                           f"尝试 clip=False ...")
            runs_noclip = run_seeds(model_name, data, in_dim, domain_labels, edge_index, num_edges,
                                    n_seeds=N_SEEDS, clip=False)
            n_healthy_nc = sum(1 for r in runs_noclip if not r.get("crash"))
            if n_healthy_nc > n_healthy:
                runs = runs_noclip
                n_healthy = n_healthy_nc
                clip_used = False
                logger.info(f"[{model_name}+{MODALITY}] 采用 clip=False "
                            f"(healthy {n_healthy_nc}/{N_SEEDS} > {n_healthy})")
            else:
                logger.info(f"[{model_name}+{MODALITY}] 保留 clip=True "
                            f"(noclip healthy {n_healthy_nc} <= {n_healthy})")

        summary = aggregate(runs)
        summary.update({
            "model": model_name, "modality": MODALITY, "task": "link_prediction",
            "in_dim": in_dim, "lr": 0.01, "neg_ratio": 10, "proj_dim": None,
            "clip_grad_used": clip_used, "n_seeds": N_SEEDS,
        })
        out = os.path.join(OUTDIR, f"lp_{model_name}_{MODALITY}.json")
        with open(out, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        dt = time.time() - t0
        flag = " [ALL CRASH]" if summary["all_crash"] else ""
        clipf = "" if clip_used else " [clip=OFF]"
        logger.info(f"[{model_name}+{MODALITY}] DONE MRR={summary['mrr_mean']:.4f}±{summary['mrr_std']:.4f}"
                    f" healthy={summary['num_healthy']}/{N_SEEDS}{clipf}{flag} "
                    f"({dt:.0f}s) -> {out}")

    logger.info(f"=== modality {MODALITY} all done ===")


if __name__ == "__main__":
    main()
