"""
跨域链接预测实验 - 只预测跨域引用边，解决原链接预测任务饱和的问题
修复：消息传递严格使用训练边，负采样排除所有正样本
"""
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict
from loguru import logger
from sklearn.metrics import accuracy_score


def run_cross_domain_link_prediction(
    dataset,
    model_names: list = None,
    modalities: list = None,
    num_runs: int = 5,
    num_epochs: int = 500,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.5,
    lr: float = 0.01,
    device: str = "cuda",
    output_dir: str = "./benchmark/results",
) -> Dict:
    """
    跨域链接预测：只用跨域引用边(3,838条)作为正样本
    这比全链接预测(59,364条)难得多，因为跨域引用模式差异大
    """
    from benchmark.models.gnn import get_model
    from benchmark.models.link_predictor import DotProductDecoder

    if model_names is None:
        model_names = ["gcn", "graphsage", "gat", "gt"]
    if modalities is None:
        modalities = ["text", "text+image+formula", "image+formula"]

    os.makedirs(output_dir, exist_ok=True)

    # 识别跨域边
    domain_labels = dataset._domain_labels
    edge_index = dataset.data.edge_index
    src_domains = domain_labels[edge_index[0]]
    dst_domains = domain_labels[edge_index[1]]
    cross_domain_mask = src_domains != dst_domains
    cross_edges = edge_index[:, cross_domain_mask]

    # 识别域内边（不参与预测，但用于消息传递构建）
    intra_domain_mask = src_domains == dst_domains
    intra_edges = edge_index[:, intra_domain_mask]

    logger.info(f"跨域边数: {cross_edges.size(1)} / 总边数: {edge_index.size(1)} "
                f"({cross_edges.size(1)/edge_index.size(1)*100:.1f}%)")

    data = dataset.data.to(device)
    in_dim = data.x.size(1)
    num_nodes = data.num_nodes

    all_results = {}
    for model_name in model_names:
        for modality in modalities:
            key = f"cdlp_{model_name}_{modality}"
            run_results = []

            for run in range(num_runs):
                seed = run * 42 + 1
                rng = np.random.RandomState(seed)

                # 划分跨域边
                n_cross = cross_edges.size(1)
                perm = torch.randperm(n_cross, generator=torch.Generator().manual_seed(seed))
                train_size = int(n_cross * 0.6)
                val_size = int(n_cross * 0.2)

                train_cross_edges = cross_edges[:, perm[:train_size]].to(device)
                val_pos = cross_edges[:, perm[train_size:train_size + val_size]].to(device)
                test_pos = cross_edges[:, perm[train_size + val_size:]].to(device)

                # 消息传递边 = 域内边 + 训练集跨域边（不包含val/test跨域边）
                train_msg_edges = torch.cat([intra_edges.to(device), train_cross_edges], dim=1)

                # 严格负采样：排除所有正样本边（跨域+域内）
                all_pos_edges = edge_index.cpu()
                val_neg = _cross_domain_negative_sampling(
                    all_pos_edges, num_nodes, domain_labels, val_pos.size(1), seed=seed + 1000
                ).to(device)
                test_neg = _cross_domain_negative_sampling(
                    all_pos_edges, num_nodes, domain_labels, test_pos.size(1), seed=seed + 2000
                ).to(device)

                encoder = get_model(model_name, in_dim, hidden_dim, hidden_dim,
                                    num_layers=num_layers, dropout=dropout).to(device)
                decoder = DotProductDecoder().to(device)

                result = _train_and_eval_lp(
                    encoder, decoder, data, train_msg_edges, train_cross_edges,
                    val_pos, val_neg, test_pos, test_neg,
                    num_epochs=num_epochs, lr=lr, device=device,
                    num_nodes=num_nodes,
                )
                run_results.append(result)
                logger.info(f"[Run {run+1}/{num_runs}] {key}: "
                            f"MRR={result['mrr']:.4f}, Hits@10={result['hits@10']:.4f}")

            metrics = ["mrr", "hits@1", "hits@3", "hits@10"]
            summary = {
                "model": model_name, "modality": modality,
                "task": "cross_domain_link_prediction",
                "num_cross_edges": n_cross,
                "num_runs": num_runs,
            }
            for m in metrics:
                vals = [r[m] for r in run_results]
                summary[f"{m}_mean"] = float(np.mean(vals))
                summary[f"{m}_std"] = float(np.std(vals))
            summary["runs"] = run_results

            all_results[key] = summary
            logger.info(f"{key}: MRR={summary['mrr_mean']:.4f}±{summary['mrr_std']:.4f}, "
                        f"Hits@10={summary['hits@10_mean']:.4f}±{summary['hits@10_std']:.4f}")

    save_path = os.path.join(output_dir, "cross_domain_link_prediction.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"跨域链接预测结果保存到 {save_path}")
    return all_results


def _cross_domain_negative_sampling(all_pos_edges, num_nodes, domain_labels,
                                     num_neg, seed=42):
    """严格负采样：生成不存在的跨域节点对，排除所有正样本边"""
    rng = np.random.RandomState(seed)
    pos_set = set(zip(all_pos_edges[0].tolist(), all_pos_edges[1].tolist()))
    domains_np = domain_labels.numpy() if isinstance(domain_labels, torch.Tensor) else domain_labels

    neg_src, neg_dst = [], []
    for _ in range(num_neg * 50):
        s = rng.randint(0, num_nodes)
        d = rng.randint(0, num_nodes)
        if s != d and domains_np[s] != domains_np[d] and (s, d) not in pos_set:
            neg_src.append(s)
            neg_dst.append(d)
            if len(neg_src) >= num_neg:
                break

    return torch.tensor([neg_src, neg_dst], dtype=torch.long)


def _train_and_eval_lp(encoder, decoder, data, train_msg_edges, train_pos_edges,
                        val_pos, val_neg, test_pos, test_neg,
                        num_epochs=500, lr=0.01, device="cuda", patience=50,
                        num_nodes=None):
    """
    训练并评估跨域链接预测
    修复：消息传递只用 train_msg_edges（域内边+训练跨域边），不用完整edge_index
    """
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=lr, weight_decay=5e-4
    )
    best_val_mrr = 0
    best_state = None
    patience_counter = 0

    x = data.x
    # 构建训练负采样用的全部正边集合
    all_train_pos_set = set(zip(
        train_pos_edges[0].cpu().tolist(),
        train_pos_edges[1].cpu().tolist()
    ))

    for epoch in range(1, num_epochs + 1):
        encoder.train()
        decoder.train()
        optimizer.zero_grad()

        # 修复：消息传递只用训练边（域内+训练跨域边），不用完整edge_index
        z = encoder(x, train_msg_edges)
        pos_score = decoder(z, train_pos_edges)

        # 训练负采样：排除所有正样本边 + 自环
        neg_src = torch.randint(0, num_nodes, (train_pos_edges.size(1) * 2,), device=device)
        neg_dst = torch.randint(0, num_nodes, (train_pos_edges.size(1) * 2,), device=device)
        # 过滤自环和正样本
        mask = neg_src != neg_dst
        neg_src_f, neg_dst_f = neg_src[mask], neg_dst[mask]
        neg_edges = torch.stack([neg_src_f[:train_pos_edges.size(1)],
                                  neg_dst_f[:train_pos_edges.size(1)]])
        neg_score = decoder(z, neg_edges)

        pos_loss = -F.logsigmoid(pos_score).mean()
        neg_loss = -F.logsigmoid(-neg_score).mean()
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                z = encoder(x, train_msg_edges)
                val_ps = decoder(z, val_pos)
                val_ns = decoder(z, val_neg)
                val_metrics = _compute_lp_metrics(val_ps, val_ns)

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

    encoder.load_state_dict(best_state["encoder"])
    decoder.load_state_dict(best_state["decoder"])
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        z = encoder(x, train_msg_edges)
        test_ps = decoder(z, test_pos)
        test_ns = decoder(z, test_neg)
        test_metrics = _compute_lp_metrics(test_ps, test_ns)

    return test_metrics


def _compute_lp_metrics(pos_scores, neg_scores):
    num_pos = pos_scores.size(0)
    num_neg_per_pos = max(neg_scores.size(0) // max(num_pos, 1), 1)

    mrr_total = 0.0
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_10 = 0

    for i in range(num_pos):
        pos_s = pos_scores[i]
        neg_batch = neg_scores[i * num_neg_per_pos:(i + 1) * num_neg_per_pos]
        all_s = torch.cat([pos_s.unsqueeze(0), neg_batch])
        rank = (all_s > pos_s).sum().item() + 1

        mrr_total += 1.0 / rank
        if rank <= 1: hits_at_1 += 1
        if rank <= 3: hits_at_3 += 1
        if rank <= 10: hits_at_10 += 1

    n = max(num_pos, 1)
    return {
        "mrr": mrr_total / n,
        "hits@1": hits_at_1 / n,
        "hits@3": hits_at_3 / n,
        "hits@10": hits_at_10 / n,
    }
