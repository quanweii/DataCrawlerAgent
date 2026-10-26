"""
单域节点分类 HEADLINE runner（steam/tmdb/wikiart/arxiv）—— leak-free 协议为默认头条。

为什么 leak-free 是头条
-----------------------
transductive NC 里 GNN 对全图做消息传递;若 train/test 节点间有边相连,train 标签会
经 message-passing / label-propagation 流到 test → test 指标被「结构泄露」抬高
(wikiart LP-ceiling 无特征就到 0.915 = 铁证)。旧随机切分路径(跨切分边不剪)有此泄露;
本 runner 默认走 leak-free,把"修好"做进头条,而不只是"披露"。

leak-free 协议(默认)
  1. stratified 60/20/20 切分 (固定 split_seed=1;所有模型共用同一切分 → 方差只来自初始化)。
  2. 删所有跨 {train,val,test} 的边 → 三者不相连 → label propagation 无 train→test 路径。
  3. LP-ceiling 双报(原始图 vs 剪枝图):剪枝后应跌到≈多数类基线 = 泄露已消除铁证。
  4. 剪枝图上跑各模型(固定切分,变 init seed)→ test_acc + macro-F1 + std。

可选 --legacy-random: 旧随机切分路径(每 run 重抽切分、跨切分边不剪 = 有结构泄露),
仅作对照基线,与 leak-free 并列报告(论文"保守双报")。

用法(leak-free 头条):
  python benchmark/run_newdomain_nc.py --domain wikiart --gpu 6 \\
      --modalities text+image --models mlp,gcn,gat,graphsage,gin,gcnii --init-seeds 5
"""
# ---- CLI 在 import torch 前解析(控制 CUDA 可见性)----
import os
import argparse

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--domain", required=True, choices=["steam", "tmdb", "wikiart", "arxiv", "spotify"])
_pre.add_argument("--gpu", type=int, default=None)
_pre.add_argument("--modalities", default="text+image")
_pre.add_argument("--models", default="mlp,gcn,gat,graphsage,gin,gcnii")
_pre.add_argument("--epochs", type=int, default=300)
_pre.add_argument("--hidden", type=int, default=256)
_pre.add_argument("--layers", type=int, default=2)
_pre.add_argument("--dropout", type=float, default=0.5)
_pre.add_argument("--lr", type=float, default=0.01)
# leak-free 头条参数
_pre.add_argument("--init-seeds", type=int, default=5,
                  help="leak-free 路径的初始化 seed 数(固定切分下变初始化)")
_pre.add_argument("--split-seed", type=int, default=1,
                  help="leak-free 路径的固定分层切分 seed(所有模型共用)")
# 对照基线
_pre.add_argument("--legacy-random", action="store_true",
                  help="走旧随机切分路径(跨切分边不剪=有泄露),仅作对照;默认 leak-free")
_pre.add_argument("--runs", type=int, default=3,
                  help="legacy-random 路径每模型重复次数(per-run 重抽切分)")
_pre.add_argument("--out-suffix", default="",
                  help="输出文件名后缀(如 _perseed)，避免覆盖头条 summary")
_pre.add_argument("--proj-dim", type=int, default=768,
                  help="learnable Linear 投影维度(OpenMAG 公平模态对比);0=不投影")
_pre.add_argument("--protocols", default="all",
                  help="all 或逗号分隔: transductive,split_disjoint,temporal,group_inductive")
_args, _ = _pre.parse_known_args()
if _args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_args.gpu)

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from loguru import logger

logger.add(sys.stderr, level="INFO")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.data.domain_loader import SingleDomainDataset
from benchmark.data.splits import (stratified_split, prune_cross_split_edges,
                                   label_propagation_ceiling)
from benchmark.data.protocols import (PROTOCOLS, generate_protocol_views,
                                      protocol_excluded_relations,
                                      save_protocol_suite)
from benchmark.models.gnn import get_model, model_lr
from benchmark.tasks.node_classification import train_and_eval, run_node_classification


# ===================== leak-free 头条路径 =====================

def run_leak_free(domain, modality, models, device, out_dir):
    """固定分层切分 + 跨切分边剪枝 + LP-ceiling 双报 + 多 init-seed。"""
    ds = SingleDomainDataset(domain, modality=modality)
    y = ds.data.y
    ei_full = ds.data.edge_index
    n = ds.data.num_nodes

    # 1. 固定分层切分(所有模型共用)
    masks = stratified_split(y, ratios=(0.6, 0.2, 0.2), seed=_args.split_seed)
    tr, va, te = masks

    # 2. 跨切分边剪枝
    ei_pruned = prune_cross_split_edges(ei_full, masks)
    orig, kept = ei_full.size(1), ei_pruned.size(1)
    logger.info(f"\n[{domain}/{modality}] leak-free: 切分 seed={_args.split_seed} "
                f"边 {orig} -> {kept} (删 {orig-kept}, {100*(orig-kept)/max(orig,1):.1f}%)")

    # 3. LP-ceiling 双报(证明泄露已消除)
    lp_orig = label_propagation_ceiling(y, ei_full, tr, te)
    lp_pruned = label_propagation_ceiling(y, ei_pruned, tr, te)
    logger.info(f"[{domain}] LP-ceiling(无特征): 原始图 acc={lp_orig['acc']:.3f} "
                f"-> 剪枝图 acc={lp_pruned['acc']:.3f} "
                f"(多数类={lp_pruned['majority_acc']:.3f})  "
                f"→ 泄露{'已消除' if lp_pruned['acc'] <= lp_pruned['majority_acc'] + 0.05 else '仍残留'}")

    # 4. 剪枝图上跑各模型(固定切分,变 init seed)
    data = Data(x=ds.data.x, edge_index=ei_pruned, y=y)
    data.num_nodes = n
    in_dim, out_dim = data.x.size(1), int(y.max().item()) + 1

    table = {}
    for model in models:
        logger.info(f"--- {model} + {modality} (leak-free, init_seeds={_args.init_seeds}) ---")
        accs, f1s = [], []
        for seed in range(_args.init_seeds):
            torch.manual_seed(seed)
            m = get_model(model, in_dim, _args.hidden, out_dim,
                          num_layers=_args.layers, dropout=_args.dropout,
                          proj_dim=_args.proj_dim if _args.proj_dim else None)
            r = train_and_eval(m, data, tr, va, te,
                               num_epochs=_args.epochs, lr=model_lr(model), device=device)
            accs.append(r["test_acc"]); f1s.append(r["test_f1"])
            logger.info(f"  [seed {seed}] acc={r['test_acc']:.4f} f1={r['test_f1']:.4f}")
        key = f"{model}+{modality}"
        table[key] = {
            "acc": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "f1": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
            "accs": [float(a) for a in accs], "f1s": [float(f) for f in f1s],
        }
        logger.info(f"  ⇒ {model}: acc={np.mean(accs):.3f}±{np.std(accs):.3f} "
                    f"macro-F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}")

    summary = {
        "domain": domain, "modality": modality, "protocol": "leak_free_pruned",
        "split_seed": _args.split_seed, "init_seeds": _args.init_seeds,
        "edges_original": orig, "edges_pruned": kept, "edges_removed": orig - kept,
        "lp_ceiling_original": lp_orig, "lp_ceiling_pruned": lp_pruned,
        "table": table,
    }
    sp = out_dir / f"nc_{domain}_summary{_args.out_suffix}.json"
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"保存到 {sp}")
    return summary


# ===================== legacy 随机切分对照基线 =====================

def run_legacy_random(domain, modality, models, device, out_dir):
    """旧路径:run_node_classification 内部 per-run 重抽分层切分,跨切分边不剪(有泄露)。"""
    logger.info(f"\n[{domain}/{modality}] legacy-random(有结构泄露,仅对照)")
    ds = SingleDomainDataset(domain, modality=modality)
    table = {}
    for model in models:
        logger.info(f"--- {model} + {modality} (legacy-random, runs={_args.runs}) ---")
        summary = run_node_classification(
            ds, model_name=model, modality=modality,
            num_runs=_args.runs, num_epochs=_args.epochs,
            hidden_dim=_args.hidden, num_layers=_args.layers,
            dropout=_args.dropout, lr=_args.lr, device=device,
            split="supervised", domain_labels=None,
            output_dir=str(out_dir), save_prefix=f"nc_{domain}",
        )
        key = f"{model}+{modality}"
        table[key] = {
            "acc": summary["test_acc_mean"], "acc_std": summary["test_acc_std"],
            "f1": summary["test_f1_mean"], "f1_std": summary["test_f1_std"],
        }
    out = {"domain": domain, "modality": modality, "protocol": "legacy_random_leaky",
           "runs": _args.runs, "table": table}
    sp = out_dir / f"nc_{domain}_random_summary.json"
    with open(sp, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"保存到 {sp}")
    return out


def run_protocol_matrix(domain, modalities, models, protocols, device, out_dir):
    """Run the same encoder/fusion/model grid under four explicit graph views.

    Frozen per-node embeddings are encoded once and reused across views.  This
    is intentional: a split changes information exposure, not node content.
    Any future trainable preprocessing must be fitted on train_mask only.
    """
    protocol_out = out_dir / "protocol_matrix"
    view_out = PROJECT_ROOT / "benchmark" / "results" / "protocol_views"
    protocol_out.mkdir(parents=True, exist_ok=True)
    combined = {
        "domain": domain,
        "protocols": protocols,
        "modalities": modalities,
        "models": models,
        "split_seed": _args.split_seed,
        "init_seeds": _args.init_seeds,
        "representation_policy": (
            "frozen per-node encoder outputs reused across protocol views; "
            "label-derived fields removed before encoding; mean fusion has no fitted state"
        ),
        "results": {},
    }
    suite = None
    for modality in modalities:
        excluded = protocol_excluded_relations(domain)
        ds = SingleDomainDataset(domain, modality=modality,
                                 exclude_relations=excluded)
        if suite is None:
            suite = generate_protocol_views(
                domain, ds.graph["node_ids"], ds.data.y, ds.data.edge_index,
                seed=_args.split_seed, excluded_relations=excluded)
            save_protocol_suite(domain, suite, view_out)
            combined["protocol_manifest"] = suite["manifest"]
        for protocol in protocols:
            view = suite["views"][protocol]
            tr, va, te = view["masks"]
            edge_index = view["edge_index"]
            data = Data(x=ds.data.x, edge_index=edge_index, y=ds.data.y)
            data.num_nodes = ds.data.num_nodes
            lp = label_propagation_ceiling(ds.data.y, edge_index, tr, te)
            cells = {}
            logger.info(f"\n[{domain}/{modality}/{protocol}] nodes="
                        f"{int(tr.sum())}/{int(va.sum())}/{int(te.sum())}, "
                        f"directed_edges={edge_index.size(1)}")
            for model in models:
                accs, f1s, vals, best_epochs = [], [], [], []
                for seed in range(_args.init_seeds):
                    torch.manual_seed(seed)
                    m = get_model(
                        model, data.x.size(1), _args.hidden,
                        int(data.y.max().item()) + 1,
                        num_layers=_args.layers, dropout=_args.dropout,
                        proj_dim=_args.proj_dim if _args.proj_dim else None)
                    result = train_and_eval(
                        m, data, tr, va, te, num_epochs=_args.epochs,
                        lr=model_lr(model), device=device)
                    accs.append(float(result["test_acc"]))
                    f1s.append(float(result["test_f1"]))
                    vals.append(float(result["val_acc"]))
                    best_epochs.append(int(result["best_epoch"]))
                cells[model] = {
                    "acc": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                    "f1": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
                    "accs": accs, "f1s": f1s,
                    "val_acc": float(np.mean(vals)), "val_acc_std": float(np.std(vals)),
                    "val_accs": vals, "best_epochs": best_epochs,
                }
                logger.info(f"  {model}: acc={np.mean(accs):.4f}±{np.std(accs):.4f}, "
                            f"f1={np.mean(f1s):.4f}±{np.std(f1s):.4f}")
            key = f"{protocol}/{modality}"
            combined["results"][key] = {
                "protocol": protocol, "modality": modality,
                "split_sizes": [int(tr.sum()), int(va.sum()), int(te.sum())],
                "directed_edges": int(edge_index.size(1)),
                "topology_only_predictability": lp,
                "interpretation": "diagnostic_only_not_a_leakage_verdict",
                "models": cells,
            }
            cell_path = protocol_out / f"nc_{domain}_{protocol}_{modality.replace('+', '_')}.json"
            cell_path.write_text(json.dumps(combined["results"][key], indent=2), encoding="utf-8")
    matrix_path = protocol_out / f"nc_{domain}_protocol_matrix.json"
    matrix_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    logger.info(f"protocol matrix saved to {matrix_path}")
    return combined


def _print_table(domain, summary, models, modalities):
    proto = summary.get("protocol", "?")
    table = summary.get("table", summary)
    logger.info(f"\n{'=' * 70}\n[{domain}] NC ({proto})  test_acc / macro-F1\n{'=' * 70}")
    header = f"{'model':<14}" + "".join(f"{m + ' (acc/f1)':<26}" for m in modalities)
    logger.info(header)
    for model in models:
        row = f"{model:<14}"
        for modality in modalities:
            k = f"{model}+{modality}"
            r = table.get(k)
            if r:
                row += f"{r['acc']:.3f}/{r['f1']:.3f}±{r['f1_std']:.3f}       "
            else:
                row += f"{'N/A':<26}"
        logger.info(row)


def main():
    domain = _args.domain
    modalities = [m.strip() for m in _args.modalities.split(",") if m.strip()]
    models = [m.strip() for m in _args.models.split(",") if m.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = PROJECT_ROOT / "benchmark" / "results" / "newdomain_nc"
    out_dir.mkdir(parents=True, exist_ok=True)

    if _args.legacy_random:
        merged = None
        for modality in modalities:
            last = run_legacy_random(domain, modality, models, device, out_dir)
            if merged is None:
                merged = last
            else:
                merged["table"].update(last["table"])
        _print_table(domain, merged, models, modalities)
    else:
        protocols = list(PROTOCOLS) if _args.protocols == "all" else [
            p.strip() for p in _args.protocols.split(",") if p.strip()]
        unknown = sorted(set(protocols) - set(PROTOCOLS))
        if unknown:
            raise ValueError(f"unknown protocols: {unknown}; choose from {PROTOCOLS}")
        run_protocol_matrix(domain, modalities, models, protocols, device, out_dir)


if __name__ == "__main__":
    main()
