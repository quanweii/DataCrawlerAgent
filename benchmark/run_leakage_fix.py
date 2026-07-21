#!/usr/bin/env python3
"""
TRACK C — wikiart NC label-leakage CORRECTION.

两件事:
1) LABEL-PROPAGATION CEILING BASELINE (steam / tmdb / wikiart)
   只用 train-mask 标签 + 图邻接（无特征），做迭代式邻居多数投票（label
   propagation）。这是"图结构单独能偷多少 label"的上界。
   切分严格复现 benchmark/data/domain_loader.py get_split_masks（seed = run*42+1，
   run0→seed1，与 NC runner 一致）。
   期望锚点: wikiart ~0.92, tmdb ~0.22, steam 介于二者之间。

2) E3 SUPPRESSION (wikiart)
   重建 edge_index：移除 shared_style 边（即 relation='same_artist'），
   重跑 NC (mlp,gcn,gat,graphsage,gcnii x text,image, runs=3, gpu 0)，
   与"全图(含 shared_style)"原结果对比 → 量化 definitional-leakage 贡献。

输出:
   benchmark/results/leakage_fix/lp_ceiling.json
   benchmark/results/leakage_fix/wikiart_e3.json

用法:
   python benchmark/run_leakage_fix.py --gpu 0
   python benchmark/run_leakage_fix.py --gpu 0 --skip-e3   # 只跑 LP ceiling
"""
# ---- CLI 在 import torch 前解析（控制 CUDA 可见性）----
import os
import argparse

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", type=int, default=0)
_pre.add_argument("--runs", type=int, default=3)
_pre.add_argument("--epochs", type=int, default=300)
_pre.add_argument("--lp-iters", type=int, default=50, help="label-propagation 迭代轮数")
_pre.add_argument("--skip-e3", action="store_true", help="跳过 E3 suppression")
_pre.add_argument("--skip-lp", action="store_true", help="跳过 LP ceiling")
_args, _ = _pre.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_args.gpu)

import sys
import json
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score

logger.remove()
logger.add(sys.stderr, level="INFO")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.data.domain_loader import SingleDomainDataset
from benchmark.tasks.node_classification import run_node_classification

OUT_DIR = PROJECT_ROOT / "benchmark" / "results" / "leakage_fix"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOMAINS = ["steam", "tmdb", "wikiart"]
E3_MODELS = ["mlp", "gcn", "gat", "graphsage", "gcnii"]
E3_MODALITIES = ["text", "image"]


# =====================================================================
# 1) LABEL-PROPAGATION CEILING BASELINE
# =====================================================================

def label_propagation_predict(data, train_mask, num_iters=50):
    """只用 train-mask 标签 + 邻接结构预测所有节点（含 test）。

    迭代式邻居多数投票（等价于硬标签传播）:
      - 每轮：每个非锚点节点取邻居标签的众数投票；train 节点标签钉死。
      - 多轮直到收敛或达到 num_iters。
    无特征参与 —— 这正是"图结构单独"的信息上界。

    Returns: pred LongTensor [N]
    """
    N = data.num_nodes
    y = data.y.clone()
    ei = data.edge_index
    # 邻接表（无向）
    nbr = [[] for _ in range(N)]
    src, dst = ei[0].tolist(), ei[1].tolist()
    for s, d in zip(src, dst):
        nbr[s].append(d)

    # 初始化：训练节点用真标签，其余用 -1（未知）
    pred = torch.full((N,), -1, dtype=torch.long)
    pred[train_mask] = y[train_mask]

    is_anchor = train_mask.clone()
    for it in range(num_iters):
        changed = 0
        new_pred = pred.clone()
        order = np.arange(N)
        # 同步更新（用上一轮 pred 投票），避免顺序偏差
        for i in order:
            if is_anchor[i]:
                continue
            # 邻居中已知标签的投票
            votes = {}
            for d in nbr[i]:
                lb = pred[d].item()
                if lb >= 0:
                    votes[lb] = votes.get(lb, 0) + 1
            if votes:
                # 平票取标签值最小者（确定性）
                best = max(votes.items(), key=lambda kv: (kv[1], -kv[0]))[0]
                if new_pred[i].item() != best:
                    new_pred[i] = best
                    changed += 1
        pred = new_pred
        if changed == 0:
            logger.info(f"  LP iter {it+1}: converged (0 changes)")
            break

    # 未被传播到的节点（孤立 / 邻居全未知）→ 回退到全局多数类
    unknown = pred < 0
    if unknown.any():
        global_major = torch.bincount(y[train_mask]).argmax().item()
        pred[unknown] = global_major
        logger.info(f"  LP: {int(unknown.sum())} nodes unreachable, set to global-majority "
                    f"class {global_major}")
    return pred


def run_lp_ceiling():
    logger.info("=" * 70 + "\nLABEL-PROPAGATION CEILING BASELINE (no features)\n" + "=" * 70)
    results = {}
    for domain in DOMAINS:
        logger.info(f"\n--- {domain} LP ceiling ---")
        ds = SingleDomainDataset(domain, modality="text")  # 特征不参与，text 仅占位
        n = ds.data.num_nodes

        accs, f1s = [], []
        # 复现 NC runner 的 seed 协议: seed = run*42+1
        for run in range(_args.runs):
            seed = run * 42 + 1
            tr, va, te = ds.get_split_masks("supervised", seed=seed)
            pred = label_propagation_predict(ds.data, tr, num_iters=_args.lp_iters)
            y = ds.data.y
            acc = accuracy_score(y[te].cpu().numpy(), pred[te].cpu().numpy())
            f1 = f1_score(y[te].cpu().numpy(), pred[te].cpu().numpy(), average="macro")
            accs.append(acc); f1s.append(f1)
            logger.info(f"  [{domain}] run{run} seed={seed}: acc={acc:.4f} macroF1={f1:.4f} "
                        f"(train={int(tr.sum())} test={int(te.sum())})")

        results[domain] = {
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
            "acc_per_run": [float(a) for a in accs],
            "f1_per_run": [float(f) for f in f1s],
            "num_runs": _args.runs,
            "method": "iterative neighbor majority vote (hard label propagation), "
                      "train-labels pinned, no features",
            "homophily": float(ds.graph["homophily"]),
            "num_classes": int(len(ds.graph["label_names"])),
            "num_nodes": int(n),
            "split_seed_protocol": "seed = run*42+1 (matches run_newdomain_nc / NC runner)",
        }
        logger.info(f"  >>> {domain}: acc={np.mean(accs):.4f}±{np.std(accs):.4f}  "
                    f"macroF1={np.mean(f1s):.4f}±{np.std(f1s):.4f}")

    out = OUT_DIR / "lp_ceiling.json"
    payload = {
        "task": "label-propagation ceiling (structure-only, no features)",
        "domains": results,
        "note": ("This is the upper bound on how much TEST accuracy any feature-less "
                 "method can extract purely from graph structure + train labels. "
                 "A high value signals strong structural/label leakage in the edges."),
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"保存 LP ceiling → {out}")
    return results


# =====================================================================
# 2) E3 SUPPRESSION (wikiart): remove shared_style edges, re-run NC
# =====================================================================

def run_e3_suppression():
    logger.info("\n" + "=" * 70 +
                "\nE3 SUPPRESSION (wikiart): edge_index WITHOUT shared_style\n" +
                "=" * 70)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 先记录原始（全图，含 shared_style）数字 —— 直接读已存在的 NC 结果文件
    orig_dir = PROJECT_ROOT / "benchmark" / "results" / "newdomain_nc"
    orig = {}
    for model in E3_MODELS:
        for mod in E3_MODALITIES:
            p = orig_dir / f"nc_wikiart_{model}_{mod}_supervised.json"
            if p.exists():
                with open(p) as f:
                    s = json.load(f)
                orig[f"{model}+{mod}"] = {
                    "acc": s["test_acc_mean"], "acc_std": s["test_acc_std"],
                    "f1": s["test_f1_mean"], "f1_std": s["test_f1_std"],
                }
            else:
                logger.warning(f"原始 NC 结果缺失: {p}")

    # E3: 移除 shared_style → 只保留 same_artist 边
    suppressed = {}
    suppression_meta = {}
    for mod in E3_MODALITIES:
        logger.info(f"\n--- wikiart E3 modality={mod} (shared_style removed) ---")
        ds = SingleDomainDataset("wikiart", modality=mod, relation="same_artist")
        suppression_meta[f"{mod}_homophily"] = float(ds.graph["homophily"])
        suppression_meta[f"{mod}_num_edges"] = int(ds.graph["edge_index"].size(1) // 2)
        suppression_meta[f"{mod}_relation"] = ds.graph["relation"]
        logger.info(f"  [suppressed] homophily={ds.graph['homophily']:.3f} "
                    f"unique_edges={ds.graph['edge_index'].size(1)//2} "
                    f"(full graph homophily=0.894, edges=520570 includes shared_style)")
        for model in E3_MODELS:
            logger.info(f"  --- {model} + {mod} (no shared_style) ---")
            summary = run_node_classification(
                ds, model_name=model, modality=mod,
                num_runs=_args.runs, num_epochs=_args.epochs,
                hidden_dim=256, num_layers=2, dropout=0.5, lr=0.01,
                device=device, split="supervised", domain_labels=None,
                output_dir=str(OUT_DIR), save_prefix="nc_wikiart_e3",
            )
            suppressed[f"{model}+{mod}"] = {
                "acc": summary["test_acc_mean"], "acc_std": summary["test_acc_std"],
                "f1": summary["test_f1_mean"], "f1_std": summary["test_f1_std"],
            }

    # 组装 before/after 对比表 + leakage 贡献
    comparison = {}
    for k in suppressed:
        o = orig.get(k, {})
        s = suppressed[k]
        delta_acc = s["acc"] - o.get("acc", float("nan"))
        comparison[k] = {
            "acc_with_shared_style": o.get("acc"),
            "acc_without_shared_style": s["acc"],
            "delta_acc": float(delta_acc),
            "f1_with_shared_style": o.get("f1"),
            "f1_without_shared_style": s["f1"],
            "delta_f1": float(s["f1"] - o.get("f1", float("nan"))),
            "leakage_contribution_acc": float(-delta_acc),  # shared_style 提供了多少 acc
        }

    payload = {
        "task": "E3: wikiart NC with shared_style edges removed (same_artist only)",
        "original_source": "benchmark/results/newdomain_nc/nc_wikiart_*.json",
        "full_graph": {"homophily": 0.894, "num_edges": 520570,
                       "relations": {"same_artist": 443554, "shared_style": 77016}},
        "suppressed_graph_meta": suppression_meta,
        "models": E3_MODELS, "modalities": E3_MODALITIES,
        "runs": _args.runs, "epochs": _args.epochs,
        "original": orig,
        "suppressed": suppressed,
        "comparison": comparison,
        "note": ("shared_style edges connect artworks sharing a *style*, and the wikiart "
                 "NC label IS the style (64 classes). These edges are therefore "
                 "definitionally tied to the prediction target — removing them isolates "
                 "the leakage contribution. delta_acc<0 means shared_style was inflating "
                 "accuracy via label leakage."),
    }
    out = OUT_DIR / "wikiart_e3.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"保存 E3 → {out}")
    _print_e3_table(orig, suppressed, comparison)
    return payload


def _print_e3_table(orig, suppressed, comparison):
    logger.info("\n" + "=" * 78)
    logger.info("WIKIART E3: acc before (with shared_style) vs after (without)  [delta]")
    logger.info("=" * 78)
    logger.info(f"{'cell':<20}{'before':<12}{'after':<12}{'delta':<10}{'leakage':<10}")
    for k in suppressed:
        c = comparison[k]
        before = c["acc_with_shared_style"]
        after = c["acc_without_shared_style"]
        d = c["delta_acc"]
        leak = c["leakage_contribution_acc"]
        bs = f"{before:.3f}" if before is not None else "N/A"
        logger.info(f"{k:<20}{bs:<12}{after:<12}{d:+.3f}     {leak:+.3f}")


# =====================================================================
def main():
    logger.info(f"GPU visible: {os.environ.get('CUDA_VISIBLE_DEVICES')}  "
                f"runs={_args.runs} epochs={_args.epochs} lp_iters={_args.lp_iters}")
    if not _args.skip_lp:
        run_lp_ceiling()
    if not _args.skip_e3:
        run_e3_suppression()
    logger.info("ALL DONE.")


if __name__ == "__main__":
    main()
