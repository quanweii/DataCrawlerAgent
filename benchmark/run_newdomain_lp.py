"""
单域去饱和链路预测 runner（steam/tmdb/wikiart）—— 论文核心卖点表。

卖点故事
--------
高同质性图上，随机负采样让 LP 退化为「同标签就预测有边」的平凡任务 → AUC≈1，
GNN 什么都学不到（wikiart 尤甚）。本 runner 用【同一模型】在
  random       均匀随机非边
  hard         同标签约束非边（消除「标签一致」捷径）
  hard_degree  同标签 + 同对数度数桶非边（再消除「度数」捷径）
三档测试负样本上评估，直接呈现「AUC 随负样本变难而下降」的去饱和曲线。

为什么不复用 run_link_prediction / train_and_eval_link_prediction
----------------------------------------------------------------
- run_link_prediction 的同域负采样在单域退化为随机 → 重新饱和。
- train_and_eval_link_prediction 一次只评一种负样本集，且每 epoch 重采样，
  无法干净地呈现「同一模型在三档负样本上的降级」。故这里写自包含训练循环。

实现要点
- 消息传递图 = train_pos（无向；仅训练正边，val/test 边不参与消息传递，无泄漏）。
- 训练负样本每 epoch 重采样（向量化 GPU 采样 + torch.isin 批量查重，排除全部正边，
  杜绝采到 held-out 正边；任意图规模都是毫秒级）。
- 模型选择：val_pos vs val_neg(random) 的 AUC，早停。
- 评估：加载最优模型，节点嵌入 z 算一次，对 test_pos vs 三档负样本分别算 AUC/AP/MRR/Hits@1。
- 崩溃检测：仅靠 NaN/Inf loss 抛异常（真正的发散），不再用训练时长误判「快速早停」为崩溃。

用法:
  python benchmark/run_newdomain_lp.py --domain steam --gpu 0 \\
      --modalities text+image --models mlp,gcn,gat,graphsage,gin \\
      --seeds 3 --epochs 300
"""
# ---- CLI 在 import torch 前解析 ----
import os
import argparse

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--domain", required=True, choices=["steam", "tmdb", "wikiart", "arxiv", "spotify"])
_pre.add_argument("--gpu", type=int, default=None)
_pre.add_argument("--modalities", default="text+image")
_pre.add_argument("--models", default="gcn")
_pre.add_argument("--seeds", type=int, default=10)
_pre.add_argument("--epochs", type=int, default=300)
_pre.add_argument("--hidden", type=int, default=256)
_pre.add_argument("--layers", type=int, default=2)
_pre.add_argument("--dropout", type=float, default=0.5)
_pre.add_argument("--lr", type=float, default=0.001)
_pre.add_argument("--patience", type=int, default=40)
_pre.add_argument("--out-suffix", default="",
                  help="输出文件名后缀(如 _advanced)，避免覆盖头条 summary")
_pre.add_argument("--proj-dim", type=int, default=768,
                  help="learnable Linear 投影维度(OpenMAG 公平模态对比);0=不投影")
_args, _ = _pre.parse_known_args()
if _args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_args.gpu)

import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

logger.add(sys.stderr, level="INFO")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.data.domain_loader import SingleDomainDataset, load_domain_graph, load_lp_edges
from benchmark.models.gnn import get_model, model_lr
from benchmark.models.link_predictor import DotProductDecoder
from benchmark.tasks.link_prediction import compute_link_prediction_metrics

NEG_SETS = ["random", "hard", "hard_degree"]


def _auc_ap(pos_scores: np.ndarray, neg_scores: np.ndarray):
    """AUC / Average-Precision。"""
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    s = np.concatenate([pos_scores, neg_scores])
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


@torch.no_grad()
def _hitsk_metrics(z, decoder, test_pos, forbidden_keys, num_nodes, device,
                   n_neg=1000, chunk=512):
    """标准 LP 排序指标：每个测试正边 (u,v)，采 n_neg 个随机过滤负（与 u 配对，过滤 train/val/test 正边），
    排 v 的分。向量化 + 分块（内存有界）。返回 hits@1/3/10 + mrr（1000 随机过滤负，标准协议）。
    file-based 负样本(ratio=1)使 hits@3/10 退化；此函数补充有意义的标准 Hits@k。"""
    z = z.to(device); test_pos = test_pos.to(device)
    T = test_pos.size(1)
    g = torch.Generator(device=device).manual_seed(20260711)
    h1 = h3 = h10 = 0.0; mrr = 0.0; counted = 0
    for i in range(0, T, chunk):
        src = test_pos[0, i:i + chunk]           # [b]
        b = src.size(0)
        pos_s = decoder(z, test_pos[:, i:i + chunk])             # [b]
        cand = torch.randint(num_nodes, (b, n_neg), generator=g, device=device)  # [b,n_neg]
        keys = src.unsqueeze(1) * num_nodes + cand
        bad = (cand == src.unsqueeze(1)) | torch.isin(keys, forbidden_keys)
        neg_edges = torch.stack([src.unsqueeze(1).expand(b, n_neg).reshape(-1), cand.reshape(-1)], 0)
        neg_s = decoder(z, neg_edges).view(b, n_neg).masked_fill(bad, float("-inf"))
        rank = (neg_s > pos_s.unsqueeze(1)).long().sum(1) + 1     # [b]
        h1 += (rank <= 1).sum().item(); h3 += (rank <= 3).sum().item(); h10 += (rank <= 10).sum().item()
        mrr += (1.0 / rank.float()).sum().item(); counted += b
    n = max(counted, 1)
    return {"hits@1": h1 / n, "hits@3": h3 / n, "hits@10": h10 / n, "mrr": mrr / n}


def _make_undirected(edge_index: torch.Tensor) -> torch.Tensor:
    """加反向边 + 去自环（用于消息传递）。"""
    mask = edge_index[0] != edge_index[1]
    ei = edge_index[:, mask]
    rev = torch.stack([ei[1], ei[0]], dim=0)
    return torch.cat([ei, rev], dim=1)


def _resample_train_neg(num_nodes: int, num_samples: int,
                        forbidden_keys: torch.Tensor,
                        generator: torch.Generator) -> torch.Tensor:
    """向量化随机负采样（GPU 批量采样 + torch.isin 批量查重）。

    forbidden_keys: 1D LongTensor (s*num_nodes+d)，含全部正边双向 key，已在 device 上。
    generator: 同 device 的 torch.Generator，保证可复现。

    旧版用 Python while 循环逐个采样 + set 查重：steam(43k 边)就 60s/seed，
    wikiart(416k 边)会几十小时/seed。向量化后任意规模都是毫秒级。
    """
    device = forbidden_keys.device
    out_s, out_d = [], []
    need = num_samples
    for _ in range(20):  # 碰撞率极低，几轮即填满
        if need <= 0:
            break
        n_try = need * 2 + 16  # 过采样 2x 覆盖拒绝
        s = torch.randint(num_nodes, (n_try,), generator=generator, device=device)
        d = torch.randint(num_nodes, (n_try,), generator=generator, device=device)
        keys = s * num_nodes + d
        bad = (s == d) | torch.isin(keys, forbidden_keys)
        good = ~bad
        out_s.append(s[good]); out_d.append(d[good])
        need -= int(good.sum().item())
    s = torch.cat(out_s)[:num_samples]
    d = torch.cat(out_d)[:num_samples]
    return torch.stack([s, d], dim=0)


def train_one(x, msg_edges, train_pos, val_pos, val_neg,
              model_name: str, num_nodes: int, args, device):
    """训练一个 encoder+decoder。

    返回 (encoder, decoder, best_val_auc, train_time)。崩溃（NaN/异常）抛出。
    """
    in_dim = x.size(1)
    encoder = get_model(model_name, in_dim, args.hidden, args.hidden,
                        num_layers=args.layers, dropout=args.dropout,
                        proj_dim=args.proj_dim if args.proj_dim else None).to(device)
    decoder = DotProductDecoder().to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=model_lr(model_name, default=args.lr), weight_decay=5e-4)

    msg_edges = msg_edges.to(device)
    x = x.to(device)
    train_pos_d = train_pos.to(device)
    val_pos_d = val_pos.to(device)
    val_neg_d = val_neg.to(device)
    neg_gen = torch.Generator(device=device)  # 负采样随机源（可复现）

    best_val_auc, best_state, patience = -1.0, None, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        encoder.train(); decoder.train()
        optimizer.zero_grad()
        z = encoder(x, msg_edges)
        pos_s = decoder(z, train_pos_d)
        neg_gen.manual_seed(epoch * 7 + 1)
        neg_ei = _resample_train_neg(
            num_nodes, train_pos_d.size(1), args._forbidden_keys, neg_gen)
        neg_s = decoder(z, neg_ei)
        loss = -F.logsigmoid(pos_s).mean() - F.logsigmoid(-neg_s).mean()
        loss.backward()
        # 梯度裁剪，防 GIN/GT 等高维输入梯度爆炸（与 arxiv LP 一致）
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()), max_norm=1.0)
        optimizer.step()

        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(f"训练发散 (loss={loss.item()})")

        if epoch % 5 == 0 or epoch == args.epochs:
            encoder.eval(); decoder.eval()
            with torch.no_grad():
                z = encoder(x, msg_edges)
                vp = decoder(z, val_pos_d)
                vn = decoder(z, val_neg_d)
                auc, _ = _auc_ap(vp.cpu().numpy(), vn.cpu().numpy())
                if auc > best_val_auc:
                    best_val_auc = auc
                    best_state = (
                        {k: v.cpu().clone() for k, v in encoder.state_dict().items()},
                        {k: v.cpu().clone() for k, v in decoder.state_dict().items()},
                    )
                    patience = 0
                else:
                    patience += 5
                if patience >= args.patience:
                    break

    train_time = time.time() - t0
    encoder.load_state_dict(best_state[0])
    decoder.load_state_dict(best_state[1])
    return encoder, decoder, best_val_auc, train_time


@torch.no_grad()
def evaluate(encoder, decoder, x, msg_edges, test_pos, neg_sets_d, device,
             num_nodes=None, forbidden_keys=None):
    """同一模型，对三档负样本计算 AUC/AP + 标准 Hits@k（1000 随机过滤负）。"""
    encoder.eval(); decoder.eval()
    z = encoder(x.to(device), msg_edges.to(device))
    tp_s = decoder(z, test_pos.to(device)).cpu().numpy()
    out = {}
    for name, neg_ei in neg_sets_d.items():
        tn_s = decoder(z, neg_ei.to(device)).cpu().numpy()
        auc, ap = _auc_ap(tp_s, tn_s)
        m = compute_link_prediction_metrics(
            torch.from_numpy(tp_s), torch.from_numpy(tn_s))
        out[name] = {"auc": auc, "ap": ap, "mrr": m["mrr"],
                     "hits@1": m["hits@1"], "hits@3": m["hits@3"], "hits@10": m["hits@10"]}
    # 标准 Hits@k/MRR（1000 随机过滤负；file-based ratio=1 使 hits@3/10 退化，故单独算）
    if num_nodes is not None and forbidden_keys is not None:
        out["hitsk"] = _hitsk_metrics(z, decoder, test_pos, forbidden_keys, num_nodes, device)
    return out


def main():
    domain = _args.domain
    modalities = [m.strip() for m in _args.modalities.split(",") if m.strip()]
    models = [m.strip() for m in _args.models.split(",") if m.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = PROJECT_ROOT / "benchmark" / "results" / "newdomain_lp"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 图 + LP 边（边分割固定 seed=42，与 lp_hard_negative_sampler 一致）
    graph = load_domain_graph(domain)
    num_nodes = graph["num_nodes"]
    train_pos = load_lp_edges(domain, graph, "train_pos")
    msg_edges = _make_undirected(train_pos)
    val_pos = load_lp_edges(domain, graph, "val_pos")
    val_neg = load_lp_edges(domain, graph, "val_neg")          # random
    test_pos = load_lp_edges(domain, graph, "test_pos")
    neg_sets = {
        "random":      load_lp_edges(domain, graph, "test_neg"),
        "hard":        load_lp_edges(domain, graph, "test_neg_hard"),
        "hard_degree": load_lp_edges(domain, graph, "test_neg_hard_degree"),
    }
    # 训练负采样禁区 = 全部正边(train+val+test)的双向 key（向量化查重用，杜绝采到 held-out 正边）
    all_pos = torch.cat([train_pos, val_pos, test_pos], dim=1)
    fwd = all_pos[0] * num_nodes + all_pos[1]
    rev = all_pos[1] * num_nodes + all_pos[0]
    forbidden_keys = torch.cat([fwd, rev]).unique().to(device)
    _args._forbidden_keys = forbidden_keys
    logger.info(f"[{domain}] forbidden(正边双向 key)={forbidden_keys.size(0)} "
                f"msg_edges={msg_edges.size(1)} train_pos={train_pos.size(1)} "
                f"test_pos={test_pos.size(1)}")

    table = {}
    for modality in modalities:
        logger.info(f"\n{'=' * 60}\n[{domain}] LP modality={modality}\n{'=' * 60}")
        ds = SingleDomainDataset(domain, modality=modality)
        x = ds.data.x

        for model in models:
            per_seed = []
            healthy = 0
            for seed in range(_args.seeds):
                torch.manual_seed(seed)
                try:
                    enc, dec, val_auc, ttime = train_one(
                        x, msg_edges, train_pos, val_pos, val_neg,
                        model, num_nodes, _args, device)
                    row = evaluate(enc, dec, x, msg_edges, test_pos, neg_sets, device,
                                   num_nodes=num_nodes, forbidden_keys=_args._forbidden_keys)
                    row["val_auc"], row["time"] = val_auc, ttime
                    per_seed.append(row)
                    healthy += 1
                    logger.info(f"  [{model}+{modality} seed{seed}] "
                                f"random AUC={row['random']['auc']:.3f} "
                                f"hard AUC={row['hard']['auc']:.3f} "
                                f"hard_deg AUC={row['hard_degree']['auc']:.3f} "
                                f"({ttime:.1f}s)")
                except Exception as e:  # 崩溃保护：单 seed 失败（如 NaN 发散）不影响其余
                    logger.error(f"  [{model}+{modality} seed{seed}] 崩溃: {e}")
                    continue
            if healthy == 0:
                logger.error(f"  [{model}+{modality}] 全部 seed 崩溃，跳过")
                continue

            # 多 seed 平均
            agg = {}
            for name in NEG_SETS:
                agg[name] = {}
                for met in ["auc", "ap", "mrr", "hits@1", "hits@3", "hits@10"]:
                    vals = [r[name][met] for r in per_seed]
                    agg[name][f"{met}_mean"] = float(np.mean(vals))
                    agg[name][f"{met}_std"] = float(np.std(vals))
            hk_vals = [r["hitsk"] for r in per_seed if r.get("hitsk")]
            if hk_vals:
                agg["hitsk_random_filtered"] = {}
                for met in ["hits@1", "hits@3", "hits@10", "mrr"]:
                    vals = [h[met] for h in hk_vals]
                    agg["hitsk_random_filtered"][f"{met}_mean"] = float(np.mean(vals))
                    agg["hitsk_random_filtered"][f"{met}_std"] = float(np.std(vals))
            agg["val_auc"] = float(np.mean([r["val_auc"] for r in per_seed]))
            agg["time"] = float(np.mean([r["time"] for r in per_seed]))
            agg["healthy_seeds"] = healthy
            table[f"{model}+{modality}"] = agg

    sp = out_dir / f"lp_{domain}_summary{_args.out_suffix}.json"
    with open(sp, "w") as f:
        json.dump(table, f, indent=2)
    logger.info(f"保存到 {sp}")
    _print_table(domain, table, models, modalities)


def _print_table(domain, table, models, modalities):
    logger.info(f"\n{'=' * 78}")
    logger.info(f"[{domain}] 链路预测去饱和结果 (AUC: random / hard / hard_degree)")
    logger.info(f"{'=' * 78}")
    for modality in modalities:
        logger.info(f"-- modality={modality} --")
        logger.info(f"{'model':<14}{'random':<12}{'hard':<12}{'hard_degree':<14}{'Δ(random-hard)':<16}")
        for model in models:
            k = f"{model}+{modality}"
            if k not in table:
                logger.info(f"{model:<14}{'N/A':<12}{'N/A':<12}{'N/A':<14}")
                continue
            r = table[k]
            ra = r["random"]["auc_mean"]
            ha = r["hard"]["auc_mean"]
            hda = r["hard_degree"]["auc_mean"]
            logger.info(f"{model:<14}{ra:<12.3f}{ha:<12.3f}{hda:<14.3f}{(ra - ha):<16.3f}")


if __name__ == "__main__":
    main()
