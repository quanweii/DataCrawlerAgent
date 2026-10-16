#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E7 — steering 合成受控扫描: 回应 effective-n≈3(只 3 真实域)的统计软肋。
合成 SBM 图, 固定节点特征(→ MLP 恒定=对照), 只旋同质性(边) → 跑 GCN vs MLP。
若合成点落在与 12 真实点相同的 gap-vs-homophily 趋势上 → 机制非 3-域巧合。
复用 run_steering 同款 get_model/train_and_eval + 受控种子协议。
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--N", type=int, default=1000)
ap.add_argument("--C", type=int, default=10)
ap.add_argument("--feat", type=int, default=8)
ap.add_argument("--avg_deg", type=int, default=10)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--epochs", type=int, default=200)
a = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark.models.gnn import get_model
from benchmark.tasks.node_classification import train_and_eval

HOMO_LEVELS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
SIGNAL = 0.5   # 弱信号
NOISE = 1.6

def make_features(N, C, D, seed=0):
    """固定特征矩阵(所有 homophily 共用 → MLP 恒定). 类条件高斯(弱信号)."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(C, D) * SIGNAL
    y = rng.randint(0, C, N)
    X = centers[y] + rng.randn(N, D) * NOISE
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def sbm_edges(y, homophily, avg_deg, seed=0):
    """实现目标同质性的 SBM 边。返回 edge_index + 实际同质性."""
    rng = np.random.RandomState(seed)
    N = len(y); C = int(y.max())+1
    n_edges = N * avg_deg // 2
    src, dst = [], []
    same = 0
    for _ in range(n_edges):
        i = rng.randint(N)
        if rng.rand() < homophily:
            cands = np.where(y.numpy() == y[i].item())[0]
        else:
            cands = np.where(y.numpy() != y[i].item())[0]
        if len(cands) == 0: continue
        j = cands[rng.randint(len(cands))]
        if j == i: continue
        src += [i, j]; dst += [j, i]
        same += int(y[i] == y[j])
    ei = torch.tensor([src, dst], dtype=torch.long)
    mask = ei[0] != ei[1]
    ei = ei[:, mask]
    h_real = (y[ei[0]] == y[ei[1]]).float().mean().item()
    return ei, h_real

def nc_model(X, y, ei, model_name, out_dim, seed):
    """受控种子 NC(与 run_steering 一致: manual_seed → MLP 跨 homophily 恒定)."""
    n = X.size(0)
    masks = stratified(y, seed=seed)
    data = Data(x=X, edge_index=ei, y=y); data.num_nodes = n
    torch.manual_seed(seed)
    m = get_model(model_name, X.size(1), 256, out_dim, num_layers=2, dropout=0.5, proj_dim=128)
    r = train_and_eval(m, data.to(DEV), *[mm.to(DEV) for mm in masks],
                       num_epochs=a.epochs, lr=0.01, device=DEV)
    return r["test_f1"]

def stratified(y, ratios=(0.6,0.2,0.2), seed=1):
    rng = np.random.RandomState(seed); n=len(y); tr=va=te=np.zeros(n,bool)
    for c in np.unique(y.numpy()):
        idx = np.where(y.numpy()==c)[0]; rng.shuffle(idx)
        i6=int(len(idx)*0.6); i8=int(len(idx)*0.8)
        tr[idx[:i6]]=True; va[idx[i6:i8]]=True; te[idx[i8:]]=True
    return torch.tensor(tr), torch.tensor(va), torch.tensor(te)

def main():
    X, y = make_features(a.N, a.C, a.feat)
    out_dim = a.C
    rows = []
    print(f"[E7] synthetic SBM sweep: N={a.N} C={a.C} feat={a.feat} seeds={a.seeds}", flush=True)
    # MLP 跨所有 homophily 必须恒定(同特征同 split) —— 先测一次确认
    ei0, _ = sbm_edges(y, 0.5, a.avg_deg, seed=0)
    mlp_ref = np.mean([nc_model(X, y, ei0, "mlp", out_dim, s) for s in range(a.seeds)])
    for h_target in HOMO_LEVELS:
        gcn_f1s, mlp_f1s = [], []
        for s in range(a.seeds):
            ei, h_real = sbm_edges(y, h_target, a.avg_deg, seed=s*100)
            torch.manual_seed(s)
            g = nc_model(X, y, ei, "gcn", out_dim, s)
            # MLP 用同一个 ei(边不影响 MLP, 但保持协议一致)
            m = nc_model(X, y, ei, "mlp", out_dim, s)
            gcn_f1s.append(g); mlp_f1s.append(m)
        gap = float(np.mean(gcn_f1s) - np.mean(mlp_f1s))
        rows.append({"h_target": h_target, "h_real": round(h_real,3),
                     "gcn_f1": round(float(np.mean(gcn_f1s)),3),
                     "mlp_f1": round(float(np.mean(mlp_f1s)),3),
                     "gap": round(gap,3)})
        print(f"  h={h_target:.2f}(real {h_real:.2f}) GCN={np.mean(gcn_f1s):.3f} "
              f"MLP={np.mean(mlp_f1s):.3f} gap={gap:+.3f}", flush=True)
    # 相关
    hr = [r["h_real"] for r in rows]; gaps = [r["gap"] for r in rows]
    r = float(np.corrcoef(hr, gaps)[0,1])
    print(f"\n=== synthetic: Pearson(gap, homophily) = {r:.3f} over {len(rows)} controlled levels ===")
    print(f"MLP constancy check: ref={mlp_ref:.3f}, per-level mlp={[row['mlp_f1'] for row in rows]}")
    json.dump({"rows": rows, "pearson_gap_homophily": round(r,4), "n_levels": len(rows),
               "mlp_reference": round(float(mlp_ref),3), "config": vars(a)},
              open(ROOT/"benchmark/results/e7_synthetic_sweep.json","w"), indent=2)
    print("saved -> benchmark/results/e7_synthetic_sweep.json")

if __name__ == "__main__":
    main()
