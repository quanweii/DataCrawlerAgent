"""
LP 结构调控(steering)——E1 的 LP 半边。

故事
----
"智能体画哪种边"不仅驱动 NC 的 GNN/MLP 翻转，也驱动 LP 的【标签捷径饱和严重度】。
对每个关系子图(同节点同特征只筛边)： positives=该关系的边，按 80/10/10 切；
用同一 GCN 在 random / hard(同标签) / hard_degree(同标签+同度数桶) 三档测试负样本上评 AUC。
预期：同质性越高(尤其 shared_style=1.0 这种"边即同标签")→ random AUC 越接近 1(被捷径
"解出")→ 去饱和 Δ 越大。这把"现有 homophilic-MAG LP 高分虚高"坐实为【构建期引入的病】：
换个关系(构建决策)就能让饱和度天差地别。

自包含(不复用 run_newdomain_lp 的 CSV 流程)：直接在张量上向量化负采样。
用法:
  python benchmark/run_steering_lp.py --gpu 0 --model gcn --seeds 3 --epochs 300
"""
# ---- CLI 在 import torch 前解析 ----
import os
import argparse

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--domains", default="steam,tmdb,wikiart")
_pre.add_argument("--gpu", type=int, default=None)
_pre.add_argument("--model", default="gcn")
_pre.add_argument("--modalities", default="text+image")
_pre.add_argument("--seeds", type=int, default=3)
_pre.add_argument("--epochs", type=int, default=300)
_pre.add_argument("--hidden", type=int, default=256)
_pre.add_argument("--layers", type=int, default=2)
_pre.add_argument("--dropout", type=float, default=0.5)
_pre.add_argument("--lr", type=float, default=0.001)
_pre.add_argument("--patience", type=int, default=40)
_pre.add_argument("--seed_split", type=int, default=42)
_args, _ = _pre.parse_known_args()
if _args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_args.gpu)

import sys
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

logger.add(sys.stderr, level="INFO")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.data.domain_loader import SingleDomainDataset, load_domain_graph
from benchmark.models.gnn import get_model
from benchmark.models.link_predictor import DotProductDecoder


# ---------- 边工具 ----------
def _unique_pairs(edge_index):
    """edge_index[2,2E](无向) → 唯一无向对 [2,E]（src<dst）。"""
    s, d = edge_index[0], edge_index[1]
    mask = s < d
    keys = s[mask].long() * 10_000_000 + d[mask].long()  # N<1e7 时唯一
    _, idx = np.unique(keys.numpy(), return_index=True)
    return torch.stack([s[mask][idx], d[mask][idx]], dim=0)


def _make_undirected(pairs):
    return torch.cat([pairs, pairs[[1, 0]]], dim=1)


def _forbidden_keys(pairs, N, device):
    s, d = pairs[0], pairs[1]
    keys = torch.cat([s * N + d, d * N + s]).unique().to(device)
    return keys


# ---------- 负采样 ----------
def _sample_random_neg(N, k, forbidden_keys, rng, device):
    fk = forbidden_keys
    out_s, out_d = [], []
    need = k
    for _ in range(30):
        if need <= 0:
            break
        m = need * 2 + 16
        s = torch.randint(N, (m,), generator=None)
        d = torch.randint(N, (m,), generator=None)
        keys = s * N + d
        good = ~((s == d) | torch.isin(keys, fk.cpu()))
        gs, gd = s[good], d[good]
        out_s.append(gs); out_d.append(gd)
        need -= int(good.sum().item())
    out_s = torch.cat(out_s)[:k]; out_d = torch.cat(out_d)[:k]
    return torch.stack([out_s, out_d], dim=0).to(device)


def _build_groups(label_of, secondary=None):
    """按 label(或 (label,bucket)) 分组 → (group_flat, group_of, offsets)。
    group_of[i] = 节点 i 所属组的 id；group_flat = 各组成员拼接；offsets[l]=组 l 在 flat 里的起点。
    """
    N = len(label_of)
    if secondary is None:
        key = label_of.astype(np.int64)
    else:
        key = label_of.astype(np.int64) * 100 + secondary.astype(np.int64)
    uniq, inv = np.unique(key, return_inverse=True)  # inv[i]=组id
    order = np.argsort(inv, kind="stable")
    sizes = np.bincount(inv)
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    group_flat = order.astype(np.int64)
    return group_flat, inv.astype(np.int64), offsets


def _sample_grouped_neg(N, k, group_flat, group_of, offsets, forbidden_keys, rng):
    """同组(同 label / 同 label+bucket)非边采样。"""
    fk = forbidden_keys.cpu()
    out_s, out_d = [], []
    need = k
    for _ in range(40):
        if need <= 0:
            break
        m = need * 2 + 16
        s = rng.randint(0, N, size=m)
        g = group_of[s]
        sizes = offsets[g + 1] - offsets[g]
        sizes = np.maximum(sizes, 1)
        ri = rng.randint(0, 1 << 30, size=m) % sizes
        d = group_flat[offsets[g] + ri]
        s_t = torch.from_numpy(s); d_t = torch.from_numpy(d)
        keys = s_t * N + d_t
        good = ~((s_t == d_t) | torch.isin(keys, fk))
        out_s.append(s_t[good].numpy()); out_d.append(d_t[good].numpy())
        need -= int(good.sum().item())
    out_s = np.concatenate(out_s)[:k]; out_d = np.concatenate(out_d)[:k]
    return torch.tensor(np.stack([out_s, out_d]), dtype=torch.long)


def _degree_buckets(edge_index, N, n_buckets=10):
    deg = np.zeros(N, dtype=np.int64)
    np.add.at(deg, edge_index[0].numpy(), 1)
    if deg.max() <= 1:
        return np.zeros(N, dtype=np.int64)
    maxd = max(deg.max(), 2)
    idx = np.minimum(n_buckets - 1,
                     (np.log2(np.maximum(deg, 1)) / math.log2(maxd) * (n_buckets - 1)).astype(np.int64))
    return idx


# ---------- 训练/评估 ----------
def _auc(pos, neg):
    from sklearn.metrics import roc_auc_score
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.concatenate([pos, neg])
    return float(roc_auc_score(y, s))


def train_lp(x, msg_edges, train_pos, val_pos, val_neg, model_name, N, args, device, seed):
    torch.manual_seed(seed)
    in_dim = x.size(1)
    enc = get_model(model_name, in_dim, args.hidden, args.hidden,
                    num_layers=args.layers, dropout=args.dropout).to(device)
    dec = DotProductDecoder().to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()),
                           lr=args.lr, weight_decay=5e-4)
    x = x.to(device); msg_edges = msg_edges.to(device)
    train_pos = train_pos.to(device); val_pos = val_pos.to(device); val_neg = val_neg.to(device)
    forbidden = args._forbidden
    best_auc, best_state, pat = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        enc.train(); dec.train(); opt.zero_grad()
        z = enc(x, msg_edges)
        pos_s = dec(z, train_pos)
        # 每 epoch 重采样训练负(random, 向量化)
        neg_ei = _sample_random_neg(N, train_pos.size(1), forbidden, None, device)
        neg_s = dec(z, neg_ei)
        loss = -F.logsigmoid(pos_s).mean() - F.logsigmoid(-neg_s).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 1.0)
        opt.step()
        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError("发散")
        if epoch % 5 == 0 or epoch == args.epochs:
            enc.eval(); dec.eval()
            with torch.no_grad():
                z = enc(x, msg_edges)
                vp = dec(z, val_pos).cpu().numpy(); vn = dec(z, val_neg).cpu().numpy()
                auc = _auc(vp, vn)
                if auc > best_auc:
                    best_auc = auc
                    best_state = ({k: v.cpu().clone() for k, v in enc.state_dict().items()},
                                  {k: v.cpu().clone() for k, v in dec.state_dict().items()})
                    pat = 0
                else:
                    pat += 5
                if pat >= args.patience:
                    break
    enc.load_state_dict(best_state[0]); dec.load_state_dict(best_state[1])
    return enc, dec


@torch.no_grad()
def eval_lp(enc, dec, x, msg_edges, test_pos, neg_sets, device):
    enc.eval(); dec.eval()
    z = enc(x.to(device), msg_edges.to(device))
    tp = dec(z, test_pos.to(device)).cpu().numpy()
    out = {}
    for name, neg in neg_sets.items():
        tn = dec(z, neg.to(device)).cpu().numpy()
        out[name] = _auc(tp, tn)
    return out


def run_one_relation(domain, relation, label, args, device):
    g = load_domain_graph(domain, relation=relation)
    N = g["num_nodes"]
    y = g["y"].numpy()
    pairs = _unique_pairs(g["edge_index"])           # [2,E] 无向唯一
    rng = np.random.RandomState(args.seed_split)
    perm = rng.permutation(pairs.size(1))
    n = pairs.size(1)
    n_tr = int(n * 0.8); n_va = int(n * 0.1)
    tr = pairs[:, perm[:n_tr]]; va = pairs[:, perm[n_tr:n_tr + n_va]]; te = pairs[:, perm[n_tr + n_va:]]
    if te.size(1) == 0:
        logger.warning(f"[{domain}/{label}] 测试边为 0，跳过"); return None
    msg = _make_undirected(tr)
    all_pos = torch.cat([tr, va, te], dim=1)
    args._forbidden = _forbidden_keys(all_pos, N, device)

    # 测试负样本(固定，每档 |te|)
    k = te.size(1)
    label_of = y.astype(np.int64)
    gflat, gof, goff = _build_groups(label_of)
    buck = _degree_buckets(msg, N)
    gflat2, gof2, goff2 = _build_groups(label_of, secondary=buck)
    rng2 = np.random.RandomState(123)
    neg_sets = {
        "random": _sample_random_neg(N, k, args._forbidden, None, device),
        "hard": _sample_grouped_neg(N, k, gflat, gof, goff, args._forbidden, rng2),
        "hard_degree": _sample_grouped_neg(N, k, gflat2, gof2, goff2, args._forbidden, rng2),
    }
    val_neg = _sample_random_neg(N, va.size(1), args._forbidden, None, device)

    ds = SingleDomainDataset(domain, modality=args.modalities, relation=relation)
    x = ds.data.x

    per_seed = []
    for seed in range(args.seeds):
        try:
            enc, dec = train_lp(x, msg, tr, va, val_neg, args.model, N, args, device, seed)
            r = eval_lp(enc, dec, x, msg, te, neg_sets, device)
            per_seed.append(r)
        except Exception as e:
            logger.error(f"  [{domain}/{label} seed{seed}] 崩溃: {e}")
    if not per_seed:
        return None
    agg = {name: float(np.mean([r[name] for r in per_seed])) for name in ["random", "hard", "hard_degree"]}
    agg["homophily"] = g["homophily"]; agg["edges"] = n; agg["test_edges"] = te.size(1)
    return agg


def main():
    domains = [d.strip() for d in _args.domains.split(",") if d.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = PROJECT_ROOT / "benchmark" / "results" / "steering"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_s = {}
    for domain in domains:
        logger.info(f"\n{'#' * 70}\n# LP STEERING domain={domain} (model={_args.model})\n{'#' * 70}")
        full = load_domain_graph(domain)
        rels = [("all", None)] + [(r, r) for r in sorted(full["relations"].keys())]
        dom = {}
        for label, rel in rels:
            r = run_one_relation(domain, rel, label, _args, device)
            if r is None:
                continue
            dom[label] = r
            logger.info(f"  [{label:<16}] homo={r['homophily']:.3f} "
                        f"rand={r['random']:.3f} hard={r['hard']:.3f} "
                        f"hard_deg={r['hard_degree']:.3f} Δ(r-hd)={r['random']-r['hard_degree']:.3f}")
        order = sorted(dom.keys(), key=lambda k: dom[k]["homophily"])
        logger.info(f"\n[{domain}] LP STEERING (按同质性升序)")
        logger.info(f"{'relation':<18}{'homo':<8}{'random':<9}{'hard':<9}{'hard_deg':<11}{'Δ(r-hd)':<10}")
        for k in order:
            r = dom[k]
            logger.info(f"{k:<18}{r['homophily']:<8.3f}{r['random']:<9.3f}{r['hard']:<9.3f}"
                        f"{r['hard_degree']:<11.3f}{r['random']-r['hard_degree']:<10.3f}")
        all_s[domain] = dom
        with open(out_dir / f"steering_lp_{domain}.json", "w") as f:
            json.dump(dom, f, indent=2)
    with open(out_dir / "steering_lp_all.json", "w") as f:
        json.dump(all_s, f, indent=2)
    logger.info(f"\n✅ LP steering 完成 → {out_dir}/steering_lp_all.json")


if __name__ == "__main__":
    main()
