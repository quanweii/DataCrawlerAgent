#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E1 本地 npz 版 —— 绕过被墙的 PyG 镜像, 直接从 git clone 到的 shchur/gnn-benchmark npz
加载公开 benchmark, 跑论文同一套 leak-profile (profile_data, 无改动).

数据源(npz 已在 /tmp/gnn-bench/data/npz/, 经 git clone --filter=blob:none --sparse 取得):
  amazon_electronics_computers.npz   构造边 co-purchase  ← 最像 MAG
  amazon_electronics_photo.npz       构造边 co-purchase  ← 最像 MAG
  ms_academic_cs.npz                 构造边 co-authorship ← 最像 MAG
  ms_academic_phy.npz                构造边 co-authorship ← 最像 MAG
  cora.npz / citeseer.npz / pubmed.npz  引用边(自然, 对照)

shchur npz = scipy.sparse CSR 序列化(adj_* / attr_*) + labels + class_names.
"""
import sys, json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from benchmark.run_public_leakage_profile import profile_data, OUT, SPLIT_SEED

NPZ = Path("/tmp/gnn-bench/data/npz")
DATASETS = [
    ("amazon_computers", "co-purchase (constructed)", "amazon_electronics_computers.npz"),
    ("amazon_photo",     "co-purchase (constructed)", "amazon_electronics_photo.npz"),
    ("coauthor_cs",      "co-authorship (constructed)", "ms_academic_cs.npz"),
    ("coauthor_physics", "co-authorship (constructed)", "ms_academic_phy.npz"),
    ("cora",             "citation (natural)",        "cora.npz"),
    ("citeseer",         "citation (natural)",        "citeseer.npz"),
    ("pubmed",           "citation (natural)",        "pubmed.npz"),
]

class G:  # 极简 Data 替身(profile_data 只用 .y 和 .edge_index)
    def __init__(self, y, ei): self.y, self.edge_index = y, ei

def load_shchur(npz_path):
    f = np.load(npz_path, allow_pickle=True)
    adj = sp.csr_matrix((f["adj_data"], f["adj_indices"], f["adj_indptr"]),
                        shape=tuple(f["adj_shape"]))
    adj = adj + adj.T  # 对称化(shchur 存上三角)
    adj.setdiag(0); adj.eliminate_zeros()
    coo = adj.tocoo()
    ei = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    y = torch.as_tensor(f["labels"]).long()
    return G(y, ei)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, failed = [], []
    for name, kind, fn in DATASETS:
        p = NPZ / fn
        if not p.exists():
            failed.append({"name": name, "kind": kind, "error": f"npz missing: {p}"}); continue
        try:
            data = load_shchur(p)
            r = profile_data(name, data, kind)
            rows.append(r)
            print(f"  OK  {name:<18} homo={r['homophily']:.2f} lp_raw={r['lp_raw']} "
                  f"pruned={r['lp_pruned']} maj={r['majority']} gap={r['leak_gap']:+.3f}", flush=True)
        except Exception as e:
            failed.append({"name": name, "kind": kind, "error": str(e)[:160]})
            print(f"  ERR {name}: {str(e)[:120]}", flush=True)
    out = OUT / "leakage_profile_public_localnpz.json"
    json.dump({"rows": rows, "failed": failed, "split_seed": SPLIT_SEED,
               "measure": "benchmark/data/splits.py (label_propagation_ceiling, stratified 60/20/20 seed=1, cross-split edge pruning)",
               "source": "shchur/gnn-benchmark npz (git clone --filter=blob:none --sparse)"},
              open(out, "w"), indent=2, ensure_ascii=False)
    print(f"\n[done] rows={len(rows)} failed={len(failed)} -> {out}")

if __name__ == "__main__":
    main()
