"""
结构调控(steering)实验 —— 论文统一闭环的承重实验(E1)。

故事
----
多智能体"画哪种边"是图同质性的旋钮。本 runner 在【同一域、同一批节点、同一批特征、
同一个 NC 切分、同一个模型初始化种子】下，只换边（按 attributes.relation 筛边，
外加 all=全图基线），重跑 NC，验证：同质性随关系变 → GNN/MLP gap 可预测翻转。
- MLP 不用边 → 跨关系必须**恒定**(对照基线，证明差异只来自边)。
- GNN 用边 → 随同质性升降。

为什么不用 run_node_classification：它在 get_model() 前不设 torch 种子，模型初始化
用了未受控 RNG → MLP 跨关系漂移 ~0.04，破坏对照。这里自带【每个 run 前 manual_seed】
的 NC 循环，保证 (model,run) 初始化在关系间一致 → MLP 严格恒定、GNN 差异纯来自边。

这正是把"多智能体=旋钮"从口号变证明：手工脚本边写死、结构不可控，做不出这种
"改建图决策→改下游"的因果对照。

用法:
  python benchmark/run_steering.py --gpu 0 \\
      --domains steam,tmdb,wikiart --modalities text \\
      --models mlp,gcn,gat,graphsage,gcnii --runs 3 --epochs 300
"""
# ---- CLI 在 import torch 前解析 ----
import os
import argparse

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--domains", default="steam,tmdb,wikiart")
_pre.add_argument("--gpu", type=int, default=None)
_pre.add_argument("--modalities", default="text")
_pre.add_argument("--models", default="mlp,gcn,gat,graphsage,gcnii")
_pre.add_argument("--runs", type=int, default=3)
_pre.add_argument("--epochs", type=int, default=300)
_pre.add_argument("--hidden", type=int, default=256)
_pre.add_argument("--layers", type=int, default=2)
_pre.add_argument("--dropout", type=float, default=0.5)
_pre.add_argument("--lr", type=float, default=0.01)
_pre.add_argument("--proj-dim", type=int, default=768, help="learnable Linear 投影;0=不投影")
_args, _ = _pre.parse_known_args()
if _args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_args.gpu)

import sys
import json
from pathlib import Path

import numpy as np
import torch
from loguru import logger

logger.add(sys.stderr, level="INFO")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.data.domain_loader import SingleDomainDataset, load_domain_graph
from benchmark.models.gnn import get_model
from benchmark.tasks.node_classification import train_and_eval


def nc_one_model(ds, model_name, modality, args, device):
    """自带受控种子的 NC：每个 run 前 manual_seed(run) → 初始化在关系间一致。"""
    data = ds.data.to(device)
    in_dim = data.x.size(1)
    out_dim = int(data.y.max().item()) + 1
    f1s, accs = [], []
    for run in range(args.runs):
        torch.manual_seed(run)  # 关键：固定模型初始化 + 训练 RNG
        train_mask, val_mask, test_mask = ds.get_split_masks(
            "supervised", seed=run * 42 + 1)  # split 种子与主表一致
        model = get_model(model_name, in_dim, args.hidden, out_dim,
                          num_layers=args.layers, dropout=args.dropout,
                          proj_dim=args.proj_dim if args.proj_dim else None)
        res = train_and_eval(model, data, train_mask, val_mask, test_mask,
                             num_epochs=args.epochs, lr=args.lr, device=device)
        f1s.append(res["test_f1"]); accs.append(res["test_acc"])
    return float(np.mean(f1s)), float(np.mean(accs)), float(np.std(f1s))


def main():
    domains = [d.strip() for d in _args.domains.split(",") if d.strip()]
    modalities = [m.strip() for m in _args.modalities.split(",") if m.strip()]
    models = [m.strip() for m in _args.models.split(",") if m.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = PROJECT_ROOT / "benchmark" / "results" / "steering"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summary = {}
    for domain in domains:
        logger.info(f"\n{'#' * 70}\n# STEERING domain={domain}\n{'#' * 70}")
        full = load_domain_graph(domain)
        rels = [("all", None)] + [(r, r) for r in sorted(full["relations"].keys())]

        dom_table = {}
        for label, rel in rels:
            logger.info(f"\n=== [{domain}] relation={label} ===")
            ds = SingleDomainDataset(domain, modality=_args.modalities, relation=rel)
            row = {
                "homophily": ds.graph["homophily"],
                "covered_nodes": ds.graph["covered"],
                "num_nodes": ds.graph["num_nodes"],
                "edges_unique": ds.graph["edge_index"].size(1) // 2,
            }
            for modality in modalities:
                for model in models:
                    f1m, accm, f1s = nc_one_model(ds, model, modality, _args, device)
                    row[f"{model}+{modality}_f1"] = f1m
                    row[f"{model}+{modality}_acc"] = accm
                    row[f"{model}+{modality}_f1std"] = f1s
            dom_table[label] = row
            logger.info(f"  [{label}] homo={row['homophily']:.3f} "
                        f"covered={row['covered_nodes']}/{row['num_nodes']} "
                        f"edges={row['edges_unique']}")
            for model in models:
                k = f"{model}+{modalities[0]}_f1"
                logger.info(f"    {model:<11} f1={row[k]:.3f}")

        # 按同质性排序展示
        order = sorted(dom_table.keys(), key=lambda t: dom_table[t]["homophily"])
        m0 = modalities[0]
        logger.info(f"\n{'=' * 80}\n[{domain}] STEERING NC (按同质性升序)  modality={m0}\n{'=' * 80}")
        head = f"{'relation':<18}{'homo':<8}{'cover':<9}" + \
               "".join(f"{m:<11}" for m in models) + "gap(MLP-bestGNN)"
        logger.info(head)
        for t in order:
            r = dom_table[t]
            cells = "".join(f"{r.get(f'{m}+{m0}_f1', 0):<11.3f}" for m in models)
            mlp = r.get(f"mlp+{m0}_f1", 0)
            gnn_models = [m for m in models if m != "mlp"]
            gnn_best = max((r.get(f"{m}+{m0}_f1", 0) for m in gnn_models), default=mlp)
            logger.info(f"{t:<18}{r['homophily']:<8.3f}{r['covered_nodes']:<9}"
                        f"{cells}{mlp - gnn_best:+.3f}")
        # MLP 恒定性自检
        mlp_vals = [dom_table[t].get(f"mlp+{m0}_f1", 0) for t in dom_table]
        logger.info(f"  [自检] MLP 跨关系 f1: min={min(mlp_vals):.3f} max={max(mlp_vals):.3f} "
                    f"spread={max(mlp_vals)-min(mlp_vals):.4f} (应≈0，证明差异只来自边)")
        all_summary[domain] = dom_table
        with open(out_dir / f"steering_{domain}.json", "w") as f:
            json.dump(dom_table, f, indent=2)

    with open(out_dir / "steering_all.json", "w") as f:
        json.dump(all_summary, f, indent=2)
    logger.info(f"\n✅ steering 全部完成 → {out_dir}/steering_all.json")


if __name__ == "__main__":
    main()
