#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exp1 (arxiv desaturation) — 把 arxiv NC label 从 4 子域(饱和, 全模型 0.89-0.90)
细化为 (子域 × subcat) = 26 类全局一致细 label，同节点同边只换 label，重跑 NC。

细 label 源：benchmark/processed/graph_data.pt 自带 labels(4) + subcat_labels(7) +
paper_ids；(子域, subcat) 组合 = 全局唯一（subcat 单独混子域，配对后不混）。
不碰 5GB arxiv-metadata-cleaned.json。

输出 nc_arxiv_summary_fine26.json（不覆盖原 saturated 结果）。
"""
import os, sys, json
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------- Phase 1: 重建变体 JSON（同节点/边，只换 label）----------
d = torch.load(PROJECT_ROOT / "benchmark/processed/graph_data.pt",
               map_location="cpu", weights_only=False)
lab = d["labels"]; sub = d["subcat_labels"]; pids = d["paper_ids"]
combo = (lab * 7 + sub)
uniq = sorted(combo.unique().tolist())
remap = {int(u): i for i, u in enumerate(uniq)}
pid2fine = {pid: remap[int(combo[i])] for i, pid in enumerate(pids)}
print(f"[exp1-arxiv] finer label: {len(uniq)} classes (from 4 subdomains)")

src = PROJECT_ROOT / "data/arxiv_datasets/arxiv_multimodal_graph_agent.json"
g = json.load(open(src, encoding="utf-8"))
nodes = g["nodes"]
missing = reassigned = 0
for n in nodes:
    fi = pid2fine.get(n["id"])
    if fi is None:
        missing += 1
        continue
    n["attributes"]["label_index"] = fi
    n["attributes"]["label"] = f"finecat_{fi}"
    reassigned += 1
print(f"[exp1-arxiv] reassigned {reassigned}/{len(nodes)} nodes (missing {missing})")
dst = PROJECT_ROOT / "data/arxiv_datasets/arxiv_multimodal_graph_agent_fine26.json"
with open(dst, "w", encoding="utf-8") as f:
    json.dump(g, f)
print(f"[exp1-arxiv] wrote {dst}")

# ---------- Phase 2: 在变体图上跑 leak-free NC ----------
sys.argv = ["run_exp1", "--domain", "arxiv", "--gpu", "0",
            "--init-seeds", "5", "--out-suffix", "_fine26",
            "--modalities", "text+image"]
sys.path.insert(0, str(PROJECT_ROOT))
import benchmark.data.domain_loader as dl
dl.DOMAIN_GRAPH["arxiv"] = str(dst)          # 指向变体图（features 仍复用 processed/arxiv）
import benchmark.run_newdomain_nc as nc
device = "cuda" if torch.cuda.is_available() else "cpu"
out_dir = PROJECT_ROOT / "benchmark" / "results" / "newdomain_nc"
models = ["mlp", "gcn", "gat", "graphsage", "gin", "gcnii"]
nc.run_leak_free("arxiv", "text+image", models, device, out_dir)
print("[exp1-arxiv] done → benchmark/results/newdomain_nc/nc_arxiv_summary_fine26.json")
