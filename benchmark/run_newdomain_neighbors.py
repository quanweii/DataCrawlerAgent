"""
单域邻居数消融 (Q2) — steam / tmdb / wikiart。

回答：图结构（每节点邻居数 K）对新域节点分类的作用，以及曲线形状是否随
同质性 (homophily) 变化。3 个域同质性差异极大：
    steam    homo 0.526  (近中性,  8 类)
    tmdb     homo 0.237  (强异质, 18 类) ← GNN 在此应当塌缩
    wikiart  homo 0.894  (强同质, 64 类) ← GNN 在此应大幅收益

实验设计
--------
- K ∈ {0, 2, 4, 6, 8}：每个节点最多保留 K 条**去重无向**邻边
    （对端方向同步保留，保持无向；不足 K 条则全保留）。
- K=0：没有边 → SAGEConv 退化为纯线性变换（无消息传递）。
  为保证 "K=0 ≡ MLP" 这条 sanity 成立，K=0 直接走 MLP 模型，
  其余 K 走 GraphSAGE。
- 模型：graphsage（text）与 graphsage（text+image），按 CLIP 嵌入。
- 每个 (domain, modality, K) 跑 runs=3，分层 60/20/20 split，seed = run*42+1。

SANITY 检查（必须在结果中复现，否则 run 不可信）：
  - K=0 test_acc ≈ MLP 全图 acc（run_newdomain_nc.py 的 nc_*_mlp_*_supervised.json）。
  - 大 K（这里 8 已经接近全图，因为多数节点度数 ≤ 8）test_acc ≈
    GraphSAGE 全图 acc（nc_*_graphsage_*_supervised.json）。

用法:
  python benchmark/run_newdomain_neighbors.py --domain steam --gpu 6 \\
      --modalities text,text+image --ks 0,2,4,6,8 --runs 3 --epochs 300

  python benchmark/run_newdomain_neighbors.py --domain all --gpu 6   # 跑全部 3 域
"""
# ---- CLI 在 import torch 前解析（控制 CUDA 可见性）----
import os
import argparse

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--domain", required=True,
                  help="steam | tmdb | wikiart | all")
_pre.add_argument("--gpu", type=int, default=None)
_pre.add_argument("--modalities", default="text,text+image")
_pre.add_argument("--ks", default="0,2,4,6,8",
                  help="逗号分隔的 K 列表，K=0 自动走 MLP")
_pre.add_argument("--runs", type=int, default=10)
_pre.add_argument("--epochs", type=int, default=300)
_pre.add_argument("--hidden", type=int, default=256)
_pre.add_argument("--layers", type=int, default=2)
_pre.add_argument("--dropout", type=float, default=0.5)
_pre.add_argument("--lr", type=float, default=0.01)
_pre.add_argument("--patience", type=int, default=50)
_pre.add_argument("--proj-dim", type=int, default=768, help="learnable Linear 投影;0=不投影")
_args, _ = _pre.parse_known_args()
if _args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_args.gpu)

import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score

logger.add(sys.stderr, level="INFO")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.data.domain_loader import SingleDomainDataset
from benchmark.models.gnn import get_model


# ===================== 邻居截断 =====================

def truncate_neighbors(edge_index: torch.Tensor, max_neighbors: int,
                       seed: int = 0) -> torch.Tensor:
    """对每个节点最多保留 max_neighbors 条**去重无向**邻边（含对端方向）。

    语义：edge_index 视为无向（run_newdomain_nc / domain_loader 已经加了对端）。
    我们对节点 u 的邻接集合去重后，随机不重复采样最多 K 条边 (u, v)，同时
    保留 (v, u)，使结果仍是无向图。这样每节点的实际入边数 ≤ K（加上被对端
    选中的额外入边），与 "每节点 K 邻居" 的 GraphSAGE 采样语义一致。

    Args:
        edge_index: [2, E] long tensor，假定无向（含双向）
        max_neighbors: 每节点最多保留的邻居数（去重后）
        seed: 随机种子，保证可复现

    Returns:
        [2, E'] long tensor，仍是无向图。
    """
    if max_neighbors <= 0:
        return torch.zeros((2, 0), dtype=torch.long)

    rng = np.random.RandomState(seed)
    num_nodes = int(edge_index.max().item()) + 1

    # 构建去重无向邻接表：对每条无向边只在一个节点上登记一次会更省，
    # 但为了"每节点视角一致"我们按有向边分别登记（u 看到 v，v 也看到 u），
    # 然后按节点 u 采样后，把选中边 (u,v) 与 (v,u) 都加入。
    # 注意：edge_index 已是无向，所以 (u,v) 和 (v,u) 都在表里。
    # 我们以无向边集合为单位去重，避免重复。
    undirected = set()
    # 同时按节点收集无向邻居
    nbr_sets = [set() for _ in range(num_nodes)]
    for i in range(edge_index.size(1)):
        a = int(edge_index[0, i].item())
        b = int(edge_index[1, i].item())
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        undirected.add(key)
        nbr_sets[a].add(b)
        nbr_sets[b].add(a)

    # 按节点采样：对每个节点，如果邻居数 > K，随机保留 K 个；记录保留的无向边
    kept = set()
    # 决定每个节点"保留"哪些邻居（无向边 (u,v) 只要被 u 或 v 选中就保留，
    # 这会让度数高的节点实际保留的边可能 > K，更接近 GraphSAGE 全邻居意义；
    # 但为确保 "≤K" 严格成立，我们改成：边 (u,v) 仅当 **双方都保留了对方** 才保留。
    # 这样每个节点的实际度数 ≤ K（因为它的保留集 ≤ K，对方保留与否只减不增）。
    node_keep = [None] * num_nodes
    for u in range(num_nodes):
        nbrs = sorted(nbr_sets[u])
        if len(nbrs) <= max_neighbors:
            node_keep[u] = set(nbrs)
        else:
            chosen = rng.choice(len(nbrs), size=max_neighbors, replace=False)
            node_keep[u] = set(nbrs[c] for c in chosen)

    for (a, b) in undirected:
        if b in node_keep[a] and a in node_keep[b]:
            kept.add((a, b))

    if not kept:
        return torch.zeros((2, 0), dtype=torch.long)

    src, dst = [], []
    for (a, b) in kept:
        src += [a, b]
        dst += [b, a]
    return torch.tensor([src, dst], dtype=torch.long)


# ===================== 单次训练 =====================

def train_and_eval(model, x, edge_index, y,
                   train_mask, val_mask, test_mask,
                   num_epochs, lr, patience, device):
    model = model.to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)
    y = y.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_test_acc = 0.0
    best_test_f1 = 0.0
    p_counter = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % 10 == 0 or epoch == num_epochs:
            model.eval()
            with torch.no_grad():
                out = model(x, edge_index)
                pred = out.argmax(dim=-1)
                val_acc = accuracy_score(y[val_mask].cpu(), pred[val_mask].cpu())
                test_acc = accuracy_score(y[test_mask].cpu(), pred[test_mask].cpu())
                test_f1 = f1_score(y[test_mask].cpu(), pred[test_mask].cpu(),
                                   average="macro")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_test_f1 = test_f1
                    p_counter = 0
                else:
                    p_counter += 10
                if p_counter >= patience:
                    break

    return {
        "test_acc": float(best_test_acc),
        "test_f1": float(best_test_f1),
        "val_acc": float(best_val_acc),
    }


def run_one_cell(ds, modality, k, num_runs, num_epochs,
                 hidden_dim, num_layers, dropout, lr, patience, device):
    """跑一个 (domain, modality, K) 单元，返回 mean/std。"""
    data = ds.data
    x = data.x
    y = data.y
    in_dim = x.size(1)
    out_dim = int(y.max().item()) + 1

    accs, f1s = [], []
    for run in range(num_runs):
        seed = run * 42 + 1
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_mask, val_mask, test_mask = ds.get_split_masks(
            split="supervised", seed=seed
        )

        # K=0 → 直接用 MLP（无消息传递），保证 sanity: K=0 ≡ MLP
        if k == 0:
            model = get_model("mlp", in_dim, hidden_dim, out_dim,
                              num_layers=num_layers, dropout=dropout,
                              proj_dim=_args.proj_dim if _args.proj_dim else None)
            edge_index_k = torch.zeros((2, 0), dtype=torch.long)
        else:
            model = get_model("graphsage", in_dim, hidden_dim, out_dim,
                              num_layers=num_layers, dropout=dropout,
                              proj_dim=_args.proj_dim if _args.proj_dim else None)
            edge_index_k = truncate_neighbors(data.edge_index, k, seed=seed)

        res = train_and_eval(model, x, edge_index_k, y,
                             train_mask, val_mask, test_mask,
                             num_epochs=num_epochs, lr=lr,
                             patience=patience, device=device)
        accs.append(res["test_acc"])
        f1s.append(res["test_f1"])
        logger.info(f"    [run {run+1}/{num_runs}] K={k} {modality}: "
                    f"acc={res['test_acc']:.4f} f1={res['test_f1']:.4f}")

    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "acc_runs": accs,
        "f1_runs": f1s,
    }


# ===================== 主流程 =====================

def run_domain(domain, modalities, ks, out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"\n{'#' * 70}\n# DOMAIN={domain}  device={device}\n{'#' * 70}")

    summary = {
        "domain": domain,
        "homophily_hint": {"steam": 0.526, "tmdb": 0.237, "wikiart": 0.894}.get(domain),
        "ks": ks,
        "modalities": modalities,
        "params": {
            "runs": _args.runs, "epochs": _args.epochs, "hidden": _args.hidden,
            "layers": _args.layers, "dropout": _args.dropout, "lr": _args.lr,
        },
        "cells": {},  # f"{modality}|k{K}" -> {...}
    }

    for modality in modalities:
        logger.info(f"\n{'=' * 60}\n[{domain}] modality={modality}  加载数据...\n{'=' * 60}")
        ds = SingleDomainDataset(domain, modality=modality)
        logger.info(f"[{domain}/{modality}] num_nodes={ds.data.num_nodes} "
                    f"edges={ds.data.edge_index.size(1)} in_dim={ds.data.x.size(1)} "
                    f"classes={int(ds.data.y.max().item())+1}")
        avg_deg = ds.data.edge_index.size(1) / max(1, ds.data.num_nodes)
        logger.info(f"[{domain}/{modality}] avg_degree(no self-loop, both dirs)={avg_deg:.2f}")

        for k in ks:
            logger.info(f"--- {domain} / {modality} / K={k} ---")
            t0 = time.time()
            cell = run_one_cell(
                ds, modality, k,
                num_runs=_args.runs, num_epochs=_args.epochs,
                hidden_dim=_args.hidden, num_layers=_args.layers,
                dropout=_args.dropout, lr=_args.lr,
                patience=_args.patience, device=device,
            )
            cell["elapsed_s"] = round(time.time() - t0, 1)
            summary["cells"][f"{modality}|k{k}"] = cell
            logger.info(f"  >> K={k} {modality}: acc={cell['acc_mean']:.4f}±"
                        f"{cell['acc_std']:.4f} f1={cell['f1_mean']:.4f}±"
                        f"{cell['f1_std']:.4f} ({cell['elapsed_s']:.0f}s)")

            # 增量保存（崩了也保住已有结果）
            sp = out_dir / f"neighbors_{domain}.json"
            with open(sp, "w") as f:
                json.dump(summary, f, indent=2)

    sp = out_dir / f"neighbors_{domain}.json"
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"[{domain}] 完整结果保存到 {sp}")
    _print_domain_table(domain, summary, modalities, ks)
    return summary


def _print_domain_table(domain, summary, modalities, ks):
    logger.info(f"\n{'=' * 72}\n[{domain}] 邻居数消融 (test_acc / macro-F1)\n{'=' * 72}")
    header = f"{'modality':<14}" + "".join(f"K={k:<5}".ljust(20) for k in ks)
    logger.info(header)
    for modality in modalities:
        row = f"{modality:<14}"
        for k in ks:
            c = summary["cells"].get(f"{modality}|k{k}")
            if c:
                row += f"{c['acc_mean']:.3f}/{c['f1_mean']:.3f}    "
            else:
                row += f"{'N/A':<20}"
        logger.info(row)
    # 趋势
    logger.info(f"\n[{domain}] ΔAcc(K=8 − K=0):")
    for modality in modalities:
        c0 = summary["cells"].get(f"{modality}|k0")
        c8 = summary["cells"].get(f"{modality}|k8")
        if c0 and c8:
            d = c8["acc_mean"] - c0["acc_mean"]
            logger.info(f"  {modality:<14}  {d:+.4f}  "
                        f"(K=0 {c0['acc_mean']:.4f} → K=8 {c8['acc_mean']:.4f})")


def main():
    if _args.domain == "all":
        domains = ["steam", "tmdb", "wikiart"]
    else:
        domains = [_args.domain]

    modalities = [m.strip() for m in _args.modalities.split(",") if m.strip()]
    ks = [int(k) for k in _args.ks.split(",") if k.strip() != ""]
    ks = sorted(set(ks))
    assert 0 in ks, "K 列表必须包含 0（用于 MLP sanity baseline）"

    out_dir = PROJECT_ROOT / "benchmark" / "results" / "newdomain_neighbors"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"配置: domains={domains} modalities={modalities} ks={ks} "
                f"runs={_args.runs} epochs={_args.epochs} gpu={_args.gpu}")

    all_summary = {}
    for domain in domains:
        all_summary[domain] = run_domain(domain, modalities, ks, out_dir)

    # 汇总跨域对比
    cross_path = out_dir / "neighbors_summary_all.json"
    with open(cross_path, "w") as f:
        json.dump(all_summary, f, indent=2)
    logger.info(f"\n跨域汇总保存到 {cross_path}")

    _print_cross_domain(all_summary, modalities, ks)


def _print_cross_domain(all_summary, modalities, ks):
    logger.info(f"\n{'#' * 72}\n# 跨域对比：同质性 vs 邻居收益\n{'#' * 72}")
    homo = {"steam": 0.526, "tmdb": 0.237, "wikiart": 0.894}
    for modality in modalities:
        logger.info(f"\n--- modality={modality} ---")
        header = f"{'domain(homo)':<22}" + "".join(f"K={k}".ljust(14) for k in ks) + "Δ(K8-K0)"
        logger.info(header)
        for domain in all_summary:
            summ = all_summary[domain]
            row = f"{domain}(h={homo.get(domain,'?')})".ljust(22)
            accs = []
            for k in ks:
                c = summ["cells"].get(f"{modality}|k{k}")
                if c:
                    row += f"{c['acc_mean']:.3f}±{c['acc_std']:.3f}  "[:14]
                    accs.append(c["acc_mean"])
                else:
                    row += f"{'N/A':<14}"
            if len(accs) == len(ks):
                row += f"{accs[-1]-accs[0]:+.3f}"
            logger.info(row)


if __name__ == "__main__":
    main()
