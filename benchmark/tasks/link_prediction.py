"""
链接预测任务 - 修复数据泄露，支持7模态消融
"""
import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict
from loguru import logger


def negative_sampling(edge_index, num_nodes, num_neg_samples=None, seed=None,
                      labels=None):
    """负采样：随机生成不存在的边。
    当提供labels时，使用同域负采样（same-domain negative sampling），
    确保负样本与正样本有相同的domain分布，避免模型只学domain匹配。
    """
    if num_neg_samples is None:
        num_neg_samples = edge_index.size(1)

    rng = np.random.RandomState(seed)
    pos_set = set(
        zip(edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy())
    )

    if labels is not None:
        return _same_domain_negative_sampling(
            pos_set, labels, num_neg_samples, rng
        )

    neg_src, neg_dst = [], []
    while len(neg_src) < num_neg_samples:
        s = rng.randint(0, num_nodes)
        d = rng.randint(0, num_nodes)
        if s != d and (s, d) not in pos_set:
            neg_src.append(s)
            neg_dst.append(d)

    return torch.tensor([neg_src, neg_dst], dtype=torch.long)


def negative_sampling_strict(all_pos_edges, num_nodes, num_neg_samples, seed=None,
                             labels=None):
    """严格负采样：排除所有正样本边（训练+验证+测试），防止跨分割泄露。
    当提供labels时，使用同域负采样。
    """
    rng = np.random.RandomState(seed)
    pos_set = set(
        zip(all_pos_edges[0].cpu().numpy(), all_pos_edges[1].cpu().numpy())
    )

    if labels is not None:
        return _same_domain_negative_sampling(
            pos_set, labels, num_neg_samples, rng
        )

    neg_src, neg_dst = [], []
    while len(neg_src) < num_neg_samples:
        s = rng.randint(0, num_nodes)
        d = rng.randint(0, num_nodes)
        if s != d and (s, d) not in pos_set:
            neg_src.append(s)
            neg_dst.append(d)

    return torch.tensor([neg_src, neg_dst], dtype=torch.long)


def _same_domain_negative_sampling(pos_set, labels, num_neg_samples, rng):
    """同域负采样：负样本与正样本具有完全相同的domain pair分布。

    核心思路：统计正样本边中每个domain对(src_domain, dst_domain)的比例，
    然后按相同比例从对应domain对中采样负边。这样正负样本的domain分布一致，
    模型必须学到比domain更细粒度的语义信号才能区分正负样本。
    """
    labels_np = labels.cpu().numpy()

    # 按domain分组节点
    domain_nodes = {}
    for i, d in enumerate(labels_np):
        domain_nodes.setdefault(int(d), []).append(i)

    # 统计正样本的domain pair分布
    domain_pair_counts = {}
    for (s, d) in pos_set:
        pair = (int(labels_np[s]), int(labels_np[d]))
        domain_pair_counts[pair] = domain_pair_counts.get(pair, 0) + 1

    # 构建加权采样列表（domain pair + 累积权重）
    pairs = list(domain_pair_counts.keys())
    counts = np.array([domain_pair_counts[p] for p in pairs], dtype=np.float64)
    cum_weights = np.cumsum(counts)
    total = cum_weights[-1]

    neg_src, neg_dst = [], []
    max_attempts = num_neg_samples * 20
    attempts = 0

    while len(neg_src) < num_neg_samples and attempts < max_attempts:
        attempts += 1
        # 按正样本的domain pair比例加权采样
        r = rng.random() * total
        idx = np.searchsorted(cum_weights, r)
        idx = min(idx, len(pairs) - 1)
        src_domain, dst_domain = pairs[idx]

        src_nodes = domain_nodes[src_domain]
        dst_nodes = domain_nodes[dst_domain]

        s = src_nodes[rng.randint(0, len(src_nodes))]
        d = dst_nodes[rng.randint(0, len(dst_nodes))]

        if s != d and (s, d) not in pos_set:
            neg_src.append(s)
            neg_dst.append(d)

    if len(neg_src) < num_neg_samples:
        logger.warning(f"同域负采样未达到目标: {len(neg_src)}/{num_neg_samples}")

    return torch.tensor([neg_src, neg_dst], dtype=torch.long)


def compute_link_prediction_metrics(pos_scores, neg_scores):
    """计算MRR和Hits@K"""
    num_pos = pos_scores.size(0)
    num_neg_per_pos = neg_scores.size(0) // num_pos if num_pos > 0 else 0

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_10 = 0
    mrr_total = 0.0

    for i in range(num_pos):
        pos_score = pos_scores[i]
        neg_batch = neg_scores[i * num_neg_per_pos:(i + 1) * num_neg_per_pos]
        if num_neg_per_pos == 0:
            neg_batch = neg_scores

        all_s = torch.cat([pos_score.unsqueeze(0), neg_batch])
        rank = (all_s > pos_score).sum().item() + 1

        mrr_total += 1.0 / rank
        if rank <= 1:
            hits_at_1 += 1
        if rank <= 3:
            hits_at_3 += 1
        if rank <= 10:
            hits_at_10 += 1

    n = max(num_pos, 1)
    return {
        "mrr": mrr_total / n,
        "hits@1": hits_at_1 / n,
        "hits@3": hits_at_3 / n,
        "hits@10": hits_at_10 / n,
    }


def train_and_eval_link_prediction(
    encoder: nn.Module,
    decoder: nn.Module,
    data,
    train_edge_index: torch.Tensor,
    val_pos_edges: torch.Tensor,
    val_neg_edges: torch.Tensor,
    test_pos_edges: torch.Tensor,
    test_neg_edges: torch.Tensor,
    num_epochs: int = 500,
    lr: float = 0.01,
    patience: int = 50,
    device: str = "cuda",
    labels: torch.Tensor = None,
    clip_grad: bool = True,
) -> Dict:
    """
    训练并评估链接预测模型

    修复：消息传递只用训练边，避免数据泄露
    修复：同域负采样，避免domain泄露导致任务饱和
    修复v2：梯度裁剪，防止RevGAT/GIN/GT等大参数模型在高维输入下梯度爆炸

    clip_grad：默认True（统一措施）。RevGAT等可逆架构在高维多模态输入下，
    梯度裁剪反而导致欠拟合/数值发散（训练26s即崩溃，MRR~0.37），故允许按需关闭。
    """
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    x = data.x.float().to(device)
    train_edges = train_edge_index.to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=lr, weight_decay=5e-4,
    )

    best_val_mrr = 0
    best_state = None
    patience_counter = 0

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        encoder.train()
        decoder.train()
        optimizer.zero_grad()

        # 修复：消息传递只用训练边
        z = encoder(x, train_edges)
        pos_score = decoder(z, train_edges)

        neg_edges = negative_sampling(
            train_edges, data.num_nodes,
            num_neg_samples=train_edges.size(1), seed=epoch,
            labels=labels,
        ).to(device)
        neg_score = decoder(z, neg_edges)

        pos_loss = -F.logsigmoid(pos_score).mean()
        neg_loss = -F.logsigmoid(-neg_score).mean()
        loss = pos_loss + neg_loss

        loss.backward()
        # 修复v2：梯度裁剪，防止RevGAT/GIN/GT等大参数模型在高维输入下梯度爆炸
        # （clip_grad=False 时跳过：供 RevGAT 等可逆架构按需关闭）
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()),
                max_norm=1.0
            )
        optimizer.step()

        if epoch % 10 == 0:
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                # 修复：验证时消息传递也只用训练边
                z = encoder(x, train_edges)
                val_pos_s = decoder(z, val_pos_edges.to(device))
                val_neg_s = decoder(z, val_neg_edges.to(device))
                val_metrics = compute_link_prediction_metrics(val_pos_s, val_neg_s)

                if val_metrics["mrr"] > best_val_mrr:
                    best_val_mrr = val_metrics["mrr"]
                    best_state = {
                        "encoder": {k: v.cpu().clone() for k, v in encoder.state_dict().items()},
                        "decoder": {k: v.cpu().clone() for k, v in decoder.state_dict().items()},
                    }
                    patience_counter = 0
                else:
                    patience_counter += 10

                if patience_counter >= patience:
                    break

    training_time = time.time() - start_time

    # 测试
    encoder.load_state_dict(best_state["encoder"])
    decoder.load_state_dict(best_state["decoder"])
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        # 修复：测试时消息传递也只用训练边
        z = encoder(x, train_edges)
        test_pos_s = decoder(z, test_pos_edges.to(device))
        test_neg_s = decoder(z, test_neg_edges.to(device))
        test_metrics = compute_link_prediction_metrics(test_pos_s, test_neg_s)

    test_metrics["training_time"] = training_time
    return test_metrics


def run_link_prediction(
    dataset,
    model_name: str,
    modality: str,
    num_runs: int = 5,
    num_epochs: int = 500,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.5,
    lr: float = 0.001,
    device: str = "cuda",
    output_dir: str = "./benchmark/results",
    proj_dim: int = None,
    neg_ratio: int = 10,
    clip_grad: bool = True,
) -> Dict:
    """
    运行链接预测实验
    lr=0.001：与节点分类一致，防止RevGAT/GIN/GT等大参数模型在高维多模态输入下梯度爆炸
    梯度裁剪(max_norm=1.0)：见 train_and_eval_link_prediction，进一步稳定训练

    修复：消息传递阶段严格只用训练边，消除数据泄露
    修复：同域负采样，避免domain泄露导致任务饱和
    修复：neg_ratio控制评估时每个正样本对应的负样本数，解决1:1评估导致饱和
    """
    from benchmark.models.gnn import get_model
    from benchmark.models.link_predictor import DotProductDecoder

    os.makedirs(output_dir, exist_ok=True)
    data = dataset.data
    in_dim = data.x.size(1)

    # 获取domain labels用于同域负采样
    domain_labels = dataset._domain_labels.cpu()

    # 划分边 - 每次run使用不同seed确保可复现
    all_run_results = []
    edge_index = data.edge_index.cpu()
    num_edges = edge_index.size(1)

    for run in range(num_runs):
        seed = run * 42 + 1
        perm = torch.randperm(num_edges, generator=torch.Generator().manual_seed(seed))
        train_size = int(num_edges * 0.6)
        val_size = int(num_edges * 0.2)

        train_edges = edge_index[:, perm[:train_size]]
        val_pos_edges = edge_index[:, perm[train_size:train_size + val_size]]
        test_pos_edges = edge_index[:, perm[train_size + val_size:]]

        # 严格同域负采样：排除所有正样本边 + 保持domain分布一致
        # 评估时使用 neg_ratio 倍负样本，使排序任务更有区分度
        all_pos_edges = torch.cat([train_edges, val_pos_edges, test_pos_edges], dim=1)
        val_neg_edges = negative_sampling_strict(
            all_pos_edges, data.num_nodes,
            val_pos_edges.size(1) * neg_ratio, seed=seed + 1000,
            labels=domain_labels,
        )
        test_neg_edges = negative_sampling_strict(
            all_pos_edges, data.num_nodes,
            test_pos_edges.size(1) * neg_ratio, seed=seed + 2000,
            labels=domain_labels,
        )

        encoder = get_model(model_name, in_dim, hidden_dim, hidden_dim,
                            num_layers=num_layers, dropout=dropout, proj_dim=proj_dim)
        decoder = DotProductDecoder()

        result = train_and_eval_link_prediction(
            encoder, decoder, data,
            train_edges, val_pos_edges, val_neg_edges,
            test_pos_edges, test_neg_edges,
            num_epochs=num_epochs, lr=lr, device=device,
            labels=domain_labels, clip_grad=clip_grad,
        )
        all_run_results.append(result)
        logger.info(f"[Run {run+1}/{num_runs}] {model_name}+{modality}: "
                     f"MRR={result['mrr']:.4f}, Hits@3={result['hits@3']:.4f}")

    # 汇总
    metrics = ["mrr", "hits@1", "hits@3", "hits@10", "training_time"]
    summary = {}
    for m in metrics:
        values = [r[m] for r in all_run_results]
        summary[f"{m}_mean"] = float(np.mean(values))
        summary[f"{m}_std"] = float(np.std(values))

    summary.update({
        "num_runs": num_runs, "model": model_name, "modality": modality,
        "task": "link_prediction", "runs": all_run_results,
    })

    save_path = os.path.join(output_dir, f"lp_{model_name}_{modality}.json")
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"链接预测结果保存到 {save_path}")
    return summary


# ===== 7模态消融实验入口 =====

MODALITY_7 = ["text", "image", "formula", "text+image", "text+formula", "image+formula", "text+image+formula"]
MODELS_4 = ["mlp", "gcn", "gat", "graphsage"]


def run_link_prediction_ablation(
    root: str = "./",
    domains: list = None,
    num_runs: int = 5,
    num_epochs: int = 500,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.5,
    lr: float = 0.01,
    device: str = "cuda",
    output_dir: str = "./benchmark/results/lp_ablation_noproj",
    neg_ratio: int = 10,
):
    """
    7模态 × 4模型 链接预测消融实验
    修复：去掉proj_dim投影层，让GNN直接处理多模态维度
    原因：ProjectedModel为多模态额外增加118万-177万参数（5.5-7.7x），
          导致不公平比较。修复后参数差异为1.75-2.5x（自然的输入维度比例）。
    neg_ratio: 评估时每个正样本对应的负样本数（默认10）
    """
    from benchmark.data.dataset import AMMAGDataset

    domains = domains or ["cs_si", "cs_lo", "math_lo", "physics_socph"]

    results_table = {}

    for modality in MODALITY_7:
        logger.info(f"\n{'='*60}")
        logger.info(f"Loading modality: {modality}")
        logger.info(f"{'='*60}")

        dataset = AMMAGDataset(
            root=root, domains=domains, modality=modality, label_type="domain"
        )

        # 计算拼接后的原始维度
        raw_dim = dataset.data.x.size(1)

        for model_name in MODELS_4:
            logger.info(f"\n--- {model_name} + {modality} (raw_dim={raw_dim}) ---")

            summary = run_link_prediction(
                dataset=dataset,
                model_name=model_name,
                modality=modality,
                num_runs=num_runs,
                num_epochs=num_epochs,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                lr=lr,
                device=device,
                output_dir=output_dir,
                proj_dim=None,
                neg_ratio=neg_ratio,
            )

            key = f"{model_name}_{modality}"
            results_table[key] = {
                "model": model_name,
                "modality": modality,
                "raw_dim": raw_dim,
                "mrr_mean": summary["mrr_mean"],
                "mrr_std": summary["mrr_std"],
                "hits@1_mean": summary["hits@1_mean"],
                "hits@3_mean": summary["hits@3_mean"],
                "hits@10_mean": summary["hits@10_mean"],
            }

            logger.info(f"  => MRR={summary['mrr_mean']:.4f}±{summary['mrr_std']:.4f}, "
                         f"H@10={summary['hits@10_mean']:.4f}")

    # 保存汇总表
    summary_path = os.path.join(output_dir, "lp_7modality_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results_table, f, indent=2, default=str)
    logger.info(f"\n汇总结果保存到 {summary_path}")

    # 打印表格
    logger.info(f"\n{'='*80}")
    logger.info("链接预测 7模态消融结果汇总 (MRR)")
    logger.info(f"{'='*80}")
    header = f"{'模型':<14}" + "".join(f"{m:<12}" for m in MODALITY_7)
    logger.info(header)
    for model_name in MODELS_4:
        row = f"{model_name:<14}"
        for modality in MODALITY_7:
            key = f"{model_name}_{modality}"
            if key in results_table:
                r = results_table[key]
                row += f"{r['mrr_mean']:.4f}±{r['mrr_std']:.3f}  "
            else:
                row += f"{'N/A':<12}"
        logger.info(row)

    return results_table
