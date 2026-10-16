"""
跨域迁移实验 — {steam, tmdb, wikiart} 三域之间的 GCN 特征提取器迁移。

协议（最干净的可行方案）
------------------------
所有域共享 768-d CLIP text 嵌入 → 特征空间天然对齐（无需投影/对齐层）。

对每个有向对 (source A -> target B)，共 12 对（3 域排列 A!=B）+ 3 对对角 A->A（SANITY）：

  Phase 1 (source training):
    在 A 上训练一个 2 层 GCN（text 特征 + A 的标签）。
      GCN.convs[0] : Linear-style GCNConv(768 -> hidden)   ← 特征提取器
      GCN.convs[1] : GCNConv(hidden -> |classes_A|)         ← A 的分类头
    用 A 的 60/20/20 分层切分训练 + early-stop on A val。

  Phase 2 (target linear probe, FROZEN extractor):
    构造一个新 GCN：convs[0] 从 A 复制并 **冻结**（requires_grad=False），
    convs[1] 换成新的 GCNConv(hidden -> |classes_B|) 作为 B 的线性头。
    在 B 的 60% train 上 **只训练 convs[1]**，early-stop on B val，eval on B 的 20% test。
    报告 B test acc + macro-F1。

SANITY
  对角 A->A：Phase 1 在 A 训练，Phase 2 把同一个 convs[0] 冻结后只在 A 上重训一个新头。
  这应该 **大致** 复现单域 2 层 GCN NC 的 acc（因为分类头被重训、conv1 冻结，
  数值会略低于端到端可微的 from-scratch GCN —— 这是预期且诚实的）。

输出
  benchmark/results/newdomain_crossdomain/transfer_matrix.json
  含 transfer_matrix[A][B] = {acc, f1, ...}, diagonal, sanity_baseline, domain_meta。

用法
  python benchmark/run_newdomain_crossdomain.py --gpu 7 --runs 3 --epochs 300
"""
# ---- CLI 在 import torch 前解析（控制 CUDA 可见性）----
import os
import argparse

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", type=int, default=7)
_pre.add_argument("--domains", default="steam,tmdb,wikiart")
_pre.add_argument("--modality", default="text")
_pre.add_argument("--runs", type=int, default=3)
_pre.add_argument("--epochs", type=int, default=300)
_pre.add_argument("--hidden", type=int, default=256)
_pre.add_argument("--dropout", type=float, default=0.5)
_pre.add_argument("--lr", type=float, default=0.01)
_args, _ = _pre.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_args.gpu)

import sys
import json
import time
import copy
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.data.domain_loader import SingleDomainDataset
from torch_geometric.nn import GCNConv


# ===================== 模型：2 层 GCN，可冻结第一层 =====================

class GCN2(nn.Module):
    """2 层 GCN。freeze_conv0=True 时 convs[0] 不更新（迁移特征提取器）。"""

    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5, freeze_conv0=False):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList([
            GCNConv(in_dim, hidden_dim),
            GCNConv(hidden_dim, out_dim),
        ])
        if freeze_conv0:
            for p in self.convs[0].parameters():
                p.requires_grad_(False)

    def forward(self, x, edge_index):
        x = self.convs[0](x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[1](x, edge_index)
        return x


# ===================== 训练 / 评估 =====================

def _train_gcn(model, data, train_mask, val_mask, num_epochs, lr, device, patience=80):
    """训练 GCN，early-stop on val acc，返回 best_state（cpu 拷贝）。"""
    model = model.to(device)
    data = data.to(device)
    # 只优化 requires_grad=True 的参数
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    best_val = -1.0
    best_state = None
    patience_cnt = 0
    y = data.y

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == num_epochs:
            model.eval()
            with torch.no_grad():
                pred = model(data.x, data.edge_index).argmax(dim=-1).cpu()
            val_acc = accuracy_score(y[val_mask].cpu(), pred[val_mask].cpu())
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 10
                if patience_cnt >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def _eval(model, data, test_mask, device):
    model.eval()
    data = data.to(device)
    model = model.to(device)
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=-1).cpu()
    y = data.y.cpu()
    return {
        "acc": float(accuracy_score(y[test_mask], pred[test_mask])),
        "f1": float(f1_score(y[test_mask], pred[test_mask], average="macro")),
    }


# ===================== 单次 transfer: A -> B =====================

def run_one_transfer(ds_src, ds_tgt, num_classes_src, num_classes_tgt,
                     hidden, dropout, lr, epochs, seed, device):
    """Phase1: train GCN on A; Phase2: freeze convs[0], train only head on B."""
    in_dim = ds_src.data.x.size(1)
    assert in_dim == ds_tgt.data.x.size(1), "feature dims must match (both 768-d CLIP text)"

    rng = np.random.RandomState(seed)

    # ---- Phase 1: source training (full GCN, both layers trainable) ----
    tr_s, va_s, te_s = ds_src.get_split_masks(split="supervised", seed=seed)
    src_model = GCN2(in_dim, hidden, num_classes_src, dropout=dropout, freeze_conv0=False)
    src_model, src_val = _train_gcn(src_model, ds_src.data, tr_s, va_s, epochs, lr, device)
    src_eval = _eval(src_model, ds_src.data, te_s, device)

    # ---- Phase 2: target linear probe with FROZEN convs[0] ----
    # 复制 convs[0] 权重，新建 convs[1]（B 的头），冻结 convs[0]
    tgt_model = GCN2(in_dim, hidden, num_classes_tgt, dropout=dropout, freeze_conv0=True)
    # 复制 convs[0]
    src_conv0_state = src_model.convs[0].state_dict()
    tgt_model.convs[0].load_state_dict(src_conv0_state)
    # 确认冻结生效
    for p in tgt_model.convs[0].parameters():
        p.requires_grad_(False)

    tr_t, va_t, te_t = ds_tgt.get_split_masks(split="supervised", seed=seed)
    tgt_model, tgt_val = _train_gcn(tgt_model, ds_tgt.data, tr_t, va_t, epochs, lr, device)
    tgt_eval = _eval(tgt_model, ds_tgt.data, te_t, device)

    # ---- 对照: from-scratch target GCN（不冻结，全重训）= 单域 GCN 基线 ----
    scratch_model = GCN2(in_dim, hidden, num_classes_tgt, dropout=dropout, freeze_conv0=False)
    scratch_model, scratch_val = _train_gcn(scratch_model, ds_tgt.data, tr_t, va_t, epochs, lr, device)
    scratch_eval = _eval(scratch_model, ds_tgt.data, te_t, device)

    return {
        "source_test_acc": src_eval["acc"],
        "source_test_f1": src_eval["f1"],
        "transfer_test_acc": tgt_eval["acc"],
        "transfer_test_f1": tgt_eval["f1"],
        "scratch_test_acc": scratch_eval["acc"],   # target from-scratch GCN（= 单域 GCN 基线）
        "scratch_test_f1": scratch_eval["f1"],
        "transfer_gain": tgt_eval["acc"] - scratch_eval["acc"],
        "source_val_acc": src_val,
        "target_val_acc": tgt_val,
        "seed": seed,
    }


# ===================== 主流程 =====================

def main():
    domains = [d.strip() for d in _args.domains.split(",") if d.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device={device}  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}  "
                f"torch.cuda.device_count()={torch.cuda.device_count() if torch.cuda.is_available() else 0}")

    # 缓存数据集 & 类别数 & 同质性
    datasets, n_classes, domain_meta = {}, {}, {}
    for d in domains:
        ds = SingleDomainDataset(d, modality=_args.modality)
        datasets[d] = ds
        nc = int(ds.data.y.max().item()) + 1
        n_classes[d] = nc
        domain_meta[d] = {
            "num_nodes": ds.graph["num_nodes"],
            "num_edges": ds.graph["edge_index"].size(1) // 2,
            "num_classes": nc,
            "homophily": float(ds.graph["homophily"]),
            "in_dim": int(ds.data.x.size(1)),
            "modality": _args.modality,
        }
        logger.info(f"[{d}] nodes={ds.graph['num_nodes']} classes={nc} "
                    f"homophily={ds.graph['homophily']:.3f} in_dim={ds.data.x.size(1)}")

    out_dir = PROJECT_ROOT / "benchmark" / "results" / "newdomain_crossdomain"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 所有有向对（含对角，做 sanity）
    pairs = [(a, b) for a in domains for b in domains]

    results = {}  # results[f"{a}->{b}"] = {...mean/std...}
    raw_runs = {}
    t0 = time.time()
    for a, b in pairs:
        key = f"{a}->{b}"
        diag = " (DIAGONAL/sanity)" if a == b else ""
        logger.info(f"\n{'='*60}\nTransfer {key}{diag}\n{'='*60}")
        run_results = []
        for run in range(_args.runs):
            seed = run * 42 + 1
            r = run_one_transfer(
                datasets[a], datasets[b], n_classes[a], n_classes[b],
                hidden=_args.hidden, dropout=_args.dropout, lr=_args.lr,
                epochs=_args.epochs, seed=seed, device=device,
            )
            run_results.append(r)
            logger.info(f"  [run {run+1}/{_args.runs}] src_acc={r['source_test_acc']:.4f} "
                        f"transfer_acc={r['transfer_test_acc']:.4f} "
                        f"scratch_acc={r['scratch_test_acc']:.4f} "
                        f"gain={r['transfer_gain']:+.4f} "
                        f"transfer_f1={r['transfer_test_f1']:.4f}")

        # 汇总
        agg = {}
        for m in ["source_test_acc", "source_test_f1", "transfer_test_acc", "transfer_test_f1",
                  "scratch_test_acc", "scratch_test_f1", "transfer_gain",
                  "source_val_acc", "target_val_acc"]:
            vals = [r[m] for r in run_results]
            agg[f"{m}_mean"] = float(np.mean(vals))
            agg[f"{m}_std"] = float(np.std(vals))
        agg["num_runs"] = _args.runs
        agg["runs"] = run_results
        results[key] = agg
        raw_runs[key] = run_results
        logger.info(f"  >>> {key} transfer={agg['transfer_test_acc_mean']:.4f}±{agg['transfer_test_acc_std']:.4f} "
                    f"(scratch={agg['scratch_test_acc_mean']:.4f}, gain={agg['transfer_gain_mean']:+.4f})")

    # 组装 transfer matrix (acc)
    matrix = {}
    for a in domains:
        matrix[a] = {}
        for b in domains:
            r = results[f"{a}->{b}"]
            matrix[a][b] = {
                "transfer_acc": round(r["transfer_test_acc_mean"], 4),
                "transfer_acc_std": round(r["transfer_test_acc_std"], 4),
                "transfer_f1": round(r["transfer_test_f1_mean"], 4),
                "scratch_acc": round(r["scratch_test_acc_mean"], 4),
                "source_acc": round(r["source_test_acc_mean"], 4),
                "transfer_gain": round(r["transfer_gain_mean"], 4),
            }

    # sanity: 对角 transfer acc vs scratch (单域 GCN)
    sanity = {}
    for d in domains:
        r = results[f"{d}->{d}"]
        sanity[d] = {
            "diagonal_transfer_acc": round(r["transfer_test_acc_mean"], 4),
            "scratch_acc": round(r["scratch_test_acc_mean"], 4),
            "note": "diagonal = freeze convs[0] then retrain head on same domain; "
                    "scratch = full from-scratch 2-layer GCN (= single-domain GCN NC baseline). "
                    "Expect diagonal <= scratch (head-only retrain).",
        }

    output = {
        "task": "cross_domain_gcn_transfer",
        "protocol": ("2-layer GCN trained on source A (768-d CLIP text feats + A labels); "
                     "convs[0] (feature extractor) FROZEN and copied to target B; "
                     "only new linear head (convs[1], GCNConv hidden->|B classes|) trained on B 60/20/20; "
                     "eval on B 20% test. scratch = full from-scratch GCN on B."),
        "modality": _args.modality,
        "hidden": _args.hidden,
        "dropout": _args.dropout,
        "lr": _args.lr,
        "epochs": _args.epochs,
        "num_runs": _args.runs,
        "device": device,
        "domains": domains,
        "domain_meta": domain_meta,
        "transfer_matrix": matrix,
        "sanity_diagonal_vs_scratch": sanity,
        "per_pair_detail": results,
        "elapsed_sec": round(time.time() - t0, 1),
    }

    out_path = out_dir / "transfer_matrix.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\n保存到 {out_path}")

    # 打印最终矩阵
    _print_matrix(matrix, domains)


def _print_matrix(matrix, domains):
    src_tgt = "source\\target"
    logger.info(f"\n{'='*70}\nGCN 跨域迁移矩阵 (transfer test_acc; 对角=冻结提取器后同域重训头)\n{'='*70}")
    header = f"{src_tgt:<16}" + "".join(f"{d:<14}" for d in domains)
    logger.info(header)
    for a in domains:
        row = f"{a:<16}"
        for b in domains:
            v = matrix[a][b]["transfer_acc"]
            row += f"{v:<14.4f}"
        logger.info(row)
    logger.info(f"\n{'-'*70}\nscratch (from-scratch GCN, = 单域 GCN 基线) test_acc:")
    for b in domains:
        s = matrix[b][b]["scratch_acc"]
        logger.info(f"  {b}: scratch={s:.4f}")
    # transfer gain 矩阵
    logger.info(f"\n{'-'*70}\ntransfer_gain = transfer_acc - target_scratch_acc (负=迁移有害):")
    hdr = f"{src_tgt:<16}" + "".join(f"{d:<14}" for d in domains)
    logger.info(hdr)
    for a in domains:
        row = f"{a:<16}"
        for b in domains:
            v = matrix[a][b]["transfer_gain"]
            row += f"{v:+.4f}".ljust(14)
        logger.info(row)


if __name__ == "__main__":
    main()
