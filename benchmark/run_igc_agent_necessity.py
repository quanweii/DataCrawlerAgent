#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IGC agent-necessity 实验 driver（承重：回答"为何必须多智能体"）。

比较 5 个 constructor 在 4 个 spec 上的关系子集选择 → 测量 G(S) 性质：
  - SchemaRule (公平脚本基线) / naive keyword / oracle (暴搜上界)
  - single_shot_llm (消融：无 verifier)
  - closed_loop_multi_agent (头条：planner↔critic 闭环)
头条 = S1b(盲重命名) 下：闭环把 LP-ceiling 压到≈多数类基线，SchemaRule 因名字匿名而崩塌，
single-shot 高方差。可选 --nc-eval 给头条加 NC-F1 列（证明去泄露未毁可学习性）。

用法
  # 1. smoke（确定性 + 1 域）
  python benchmark/run_igc_agent_necessity.py --smoke --gpu 0
  # 2. 全跑（含 LLM + NC）
  python benchmark/run_igc_agent_necessity.py --domains steam,tmdb,wikiart \
      --seeds-llm 5 --seeds-closed 3 --nc-eval --gpu 0
"""
import os
import argparse

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--domains", default="steam,tmdb,wikiart")
_pre.add_argument("--gpu", type=int, default=None)
_pre.add_argument("--seeds-llm", type=int, default=5)
_pre.add_argument("--seeds-closed", type=int, default=3)
_pre.add_argument("--nc-eval", action="store_true")
_pre.add_argument("--nc-models", default="mlp,gcn")
_pre.add_argument("--nc-seeds", type=int, default=3)
_pre.add_argument("--nc-epochs", type=int, default=200)
_pre.add_argument("--split-seed", type=int, default=1)
_pre.add_argument("--smoke", action="store_true")
_pre.add_argument("--no-llm", action="store_true", help="只跑确定性 constructor")
_args, _ = _pre.parse_known_args()
if _args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_args.gpu)

import json
import sys
import numpy as np
from pathlib import Path

import torch
from loguru import logger

logger.add(sys.stderr, level="INFO")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.igc.graph_subset import load_domain_base, measure_subset, subset_edge_index
from benchmark.igc.specs import SPECS, blind_map, inventory_for_prompt
from benchmark.igc.constructors import (naive_keyword, schema_rule, oracle,
                                        greedy_removal, single_shot_llm,
                                        closed_loop_multi_agent)

_LLM_WORKERS = 3   # 并发 LLM 调用数（保守，避免端点限流）

def _par(fn, items):
    """线程池并行 map（LLM 调用独立、client 线程安全）。保序。"""
    from concurrent.futures import ThreadPoolExecutor
    items = list(items)
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=_LLM_WORKERS) as ex:
        return list(ex.map(fn, items))


def measure_policy(base, real_set):
    """对 constructor 选出的真实关系集合，测量 G(S)。确定性。"""
    return measure_subset(base, real_set, split_seed=_args.split_seed)


def nc_f1_for_subset(base, real_set, models, device):
    """leak-free NC on 构造图：剪枝切分 + GCN/MLP。证明去泄露未毁可学习性。"""
    from benchmark.data.splits import stratified_split, prune_cross_split_edges
    from benchmark.models.gnn import get_model
    from benchmark.tasks.node_classification import train_and_eval
    from torch_geometric.data import Data
    ei, _, _ = subset_edge_index(base, real_set)
    out_dim = int(base["y"].max().item()) + 1
    in_dim = base["x"].size(1)
    res = {}
    for m in models:
        f1s = []
        for s in range(_args.nc_seeds):
            masks = stratified_split(base["y"], seed=s * 42 + 1)
            ei_p = prune_cross_split_edges(ei, masks)
            dp = Data(x=base["x"], edge_index=ei_p, y=base["y"])
            dp.num_nodes = base["num_nodes"]
            torch.manual_seed(s)
            model = get_model(m, in_dim, 256, out_dim, num_layers=2, dropout=0.5)
            r = train_and_eval(model, dp.to(device), masks[0], masks[1], masks[2],
                               num_epochs=_args.nc_epochs, lr=0.01, device=device)
            f1s.append(r["test_f1"])
        res[m] = float(np.mean(f1s))
    return res


def run_domain(domain, device, out_dir):
    from benchmark.igc.constructors import _make_client, _load_env
    _load_env()
    base = load_domain_base(domain, modality="text+image")
    all_rels = set(base["all_relations"])
    logger.info(f"\n{'#' * 70}\n# IGC domain={domain}  rels={base['all_relations']}\n{'#' * 70}")

    client = None if _args.no_llm else _make_client()
    dom_results = {"specs": {}}

    for sid, spec in SPECS.items():
        if _args.smoke and sid not in ("S1a", "S1b"):
            continue
        inventory = inventory_for_prompt(base, spec)
        bmap = blind_map(base["all_relations"]) if spec["blind"] else {}
        logger.info(f"\n=== [{domain}/{sid}] blind={spec['blind']} target={spec['target']} ===")
        spec_res = {"blind": spec["blind"], "target": spec["target"]}

        # 全关系基线（无过滤）的测量
        all_meas = measure_policy(base, all_rels)
        spec_res["all_relations_baseline"] = _strip(all_meas)

        constructors = {}

        # 确定性
        for name, fn in [("schema_rule", lambda: schema_rule(spec, inventory, bmap)),
                         ("naive_keyword", lambda: naive_keyword(spec, inventory, bmap)),
                         ("greedy_removal", lambda: greedy_removal(base, spec, split_seed=_args.split_seed))]:
            rs = fn()
            constructors[name] = {"real_set": sorted(rs), **_strip(measure_policy(base, rs))}

        # oracle（暴搜；S5 跳过）
        if sid != "S5":
            orc = oracle(base, spec, split_seed=_args.split_seed)
            constructors["oracle"] = {"real_set": sorted(orc), **_strip(measure_policy(base, orc))}

        # LLM 构造器
        if not _args.no_llm:
            # single-shot (N seeds)
            n_ss = _args.seeds_llm if not _args.smoke else 1
            def _ss(s):
                rs, _ = single_shot_llm(base, spec, inventory, bmap, client=client, seed=s)
                m = measure_policy(base, rs)
                return m["lp_ceiling"], m["homophily"]
            ss_res = _par(_ss, range(n_ss))
            ss_lp = [r[0] for r in ss_res]; ss_homos = [r[1] for r in ss_res]
            constructors["single_shot_llm"] = {
                "lp_ceiling_mean": float(np.mean(ss_lp)), "lp_ceiling_std": float(np.std(ss_lp)),
                "homophily_mean": float(np.mean(ss_homos)), "n_seeds": len(ss_lp),
            }
            # closed-loop (N seeds) — 仅对 S1a/S1b/S2
            if sid in ("S1a", "S1b", "S2"):
                n_cl = _args.seeds_closed if not _args.smoke else 1
                def _cl(s):
                    rs, trace = closed_loop_multi_agent(
                        base, spec, inventory, bmap, client=client, seed=s,
                        split_seed=_args.split_seed)
                    m = measure_policy(base, rs)
                    return m["lp_ceiling"], m["homophily"], len(trace), trace
                cl_res = _par(_cl, range(n_cl))
                cl_lp = [r[0] for r in cl_res]; cl_homos = [r[1] for r in cl_res]
                cl_rounds = [r[2] for r in cl_res]; traces = [r[3] for r in cl_res]
                constructors["closed_loop"] = {
                    "lp_ceiling_mean": float(np.mean(cl_lp)), "lp_ceiling_std": float(np.std(cl_lp)),
                    "homophily_mean": float(np.mean(cl_homos)),
                    "rounds_mean": float(np.mean(cl_rounds)), "n_seeds": len(cl_lp),
                    "example_trace": traces[0] if traces else None,
                }

        spec_res["constructors"] = constructors
        dom_results["specs"][sid] = spec_res
        _log_spec_table(domain, sid, spec_res)

    # 头条 NC-F1：S1b 闭环 vs 全关系
    if _args.nc_eval and not _args.no_llm and "S1b" in dom_results["specs"]:
        inv = inventory_for_prompt(base, SPECS["S1b"])
        bm = blind_map(base["all_relations"])
        cl_rs, _ = closed_loop_multi_agent(base, SPECS["S1b"], inv, bm, client=client,
                                           split_seed=_args.split_seed)
        models = [m.strip() for m in _args.nc_models.split(",") if m.strip()]
        dom_results["nc_f1"] = {
            "all_relations": nc_f1_for_subset(base, all_rels, models, device),
            "closed_loop_S1b": nc_f1_for_subset(base, cl_rs, models, device),
        }
        logger.info(f"[{domain}] NC-F1(all vs closed-loop S1b): {dom_results['nc_f1']}")

    with open(out_dir / f"igc_{domain}.json", "w") as f:
        json.dump(dom_results, f, indent=2, ensure_ascii=False)
    return dom_results


def _strip(meas):
    return {k: meas[k] for k in ("lp_ceiling", "majority_acc", "homophily", "n_edges")}


def _log_spec_table(domain, sid, spec_res):
    base_lp = spec_res["all_relations_baseline"]["lp_ceiling"]
    maj = spec_res["all_relations_baseline"]["majority_acc"]
    logger.info(f"  [{sid}] all-rel LPceil={base_lp:.3f} (maj={maj:.3f})")
    for name, c in spec_res["constructors"].items():
        if "lp_ceiling" in c:
            logger.info(f"    {name:<20} LPceil={c['lp_ceiling']:.3f} homo={c['homophily']:.3f} "
                        f"rels={c.get('real_set')}")
        else:
            logger.info(f"    {name:<20} LPceil={c['lp_ceiling_mean']:.3f}±{c['lp_ceiling_std']:.3f} "
                        f"homo={c['homophily_mean']:.3f} (n={c['n_seeds']})")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    domains = [d.strip() for d in _args.domains.split(",") if d.strip()]
    out_dir = PROJECT_ROOT / "benchmark" / "results" / "igc"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_dom = {}
    for dom in domains:
        all_dom[dom] = run_domain(dom, device, out_dir)
    with open(out_dir / "igc_all.json", "w") as f:
        json.dump(all_dom, f, indent=2, ensure_ascii=False)
    logger.info(f"\n✅ IGC 完成 → {out_dir}/igc_all.json")
    _print_headline(all_dom)


def _print_headline(all_dom):
    logger.info(f"\n{'=' * 80}\nHEADLINE: S1b (blind) LP-ceiling per constructor  "
                f"(lower=better; majority baseline in parens)\n{'=' * 80}")
    for dom, d in all_dom.items():
        s1b = d.get("specs", {}).get("S1b")
        if not s1b:
            continue
        maj = s1b["all_relations_baseline"]["majority_acc"]
        row = f"{dom:<10} (maj {maj:.3f}) | "
        row += f"all={s1b['all_relations_baseline']['lp_ceiling']:.3f} | "
        for name in ("schema_rule", "naive_keyword", "single_shot_llm", "closed_loop",
                     "greedy_removal", "oracle"):
            c = s1b["constructors"].get(name, {})
            if "lp_ceiling" in c:
                row += f"{name}={c['lp_ceiling']:.3f} | "
            elif "lp_ceiling_mean" in c:
                row += f"{name}={c['lp_ceiling_mean']:.3f}±{c['lp_ceiling_std']:.3f} | "
        logger.info(row)


if __name__ == "__main__":
    main()
