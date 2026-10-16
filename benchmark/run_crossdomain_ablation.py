#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨域迁移 ablation（T2⑧）—— 回答审稿人「负迁移是不是冻错层/只用 text 的 artifact？」

在 run_newdomain_crossdomain.py（单协议：冻结 convs[0]）基础上加：
  - 协议维度:  linear_probe (冻 convs[0]、只训头 = 原) + full_finetune (两层 init 自源域、全微调)
  - 模态维度:  text (768) 与 text+image (1536)，4 域维度都对齐
  - 第 4 域:    arxiv（text+image 时退化为 text+image 1536d，与其它 3 域一致）

若 full_finetune / text+image 下仍无正迁移 → 负迁移是真发现，非 artifact。
输出 benchmark/results/newdomain_crossdomain/transfer_ablation.json

用法
  python benchmark/run_crossdomain_ablation.py --gpu 6 --runs 3 --epochs 300
"""
import os
import sys
import argparse
import json
import time
from pathlib import Path

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", type=int, default=6)
_pre.add_argument("--domains", default="steam,tmdb,wikiart,arxiv")
_pre.add_argument("--modalities", default="text,text+image")
_pre.add_argument("--protocols", default="linear_probe,full_finetune")
_pre.add_argument("--runs", type=int, default=3)
_pre.add_argument("--epochs", type=int, default=300)
_pre.add_argument("--hidden", type=int, default=256)
_pre.add_argument("--dropout", type=float, default=0.5)
_pre.add_argument("--lr", type=float, default=0.01)
_args, _ = _pre.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_args.gpu)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from loguru import logger

logger.remove(); logger.add(sys.stderr, level="INFO")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.data.domain_loader import SingleDomainDataset
from torch_geometric.nn import GCNConv


class GCN2(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList([GCNConv(in_dim, hidden_dim), GCNConv(hidden_dim, out_dim)])

    def forward(self, x, edge_index):
        x = F.dropout(F.relu(self.convs[0](x, edge_index)), p=self.dropout, training=self.training)
        return self.convs[1](x, edge_index)


def train_gcn(model, data, tr, va, epochs, lr, device, freeze_conv0=False, patience=80):
    model = model.to(device); data = data.to(device)
    if freeze_conv0:
        for p in model.convs[0].parameters():
            p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=5e-4)
    crit = nn.CrossEntropyLoss()
    best, bestst, pat = -1.0, None, 0
    y = data.y
    for ep in range(1, epochs + 1):
        model.train(); opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = crit(out[tr], y[tr]); loss.backward(); opt.step()
        if ep % 10 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                pred = model(data.x, data.edge_index).argmax(1).cpu()
            vac = accuracy_score(y[va].cpu(), pred[va].cpu())
            if vac > best:
                best = vac; bestst = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; pat = 0
            else:
                pat += 10
                if pat >= patience:
                    break
    if bestst:
        model.load_state_dict(bestst)
    return model, best


def evaluate(model, data, te, device):
    model.eval(); model.to(device); data = data.to(device)
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(1).cpu()
    y = data.y.cpu()
    return {"acc": float(accuracy_score(y[te], pred[te])),
            "f1": float(f1_score(y[te], pred[te], average="macro", zero_division=0))}


def main():
    domains = [d.strip() for d in _args.domains.split(",")]
    modalities = [m.strip() for m in _args.modalities.split(",")]
    protocols = [p.strip() for p in _args.protocols.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = PROJECT_ROOT / "benchmark" / "results" / "newdomain_crossdomain"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 缓存 (domain, modality) -> dataset
    DS = {}
    def get_ds(d, mod):
        key = (d, mod)
        if key not in DS:
            DS[key] = SingleDomainDataset(d, modality=mod)
        return DS[key]

    results = {}
    t0 = time.time()
    for mod in modalities:
        results[mod] = {}
        # 预读各类别数 + 校验维度一致
        nc = {}
        in_dim = None
        for d in domains:
            ds = get_ds(d, mod)
            nc[d] = int(ds.data.y.max().item()) + 1
            if in_dim is None:
                in_dim = ds.data.x.size(1)
            assert ds.data.x.size(1) == in_dim, f"{d} in_dim {ds.data.x.size(1)} != {in_dim}"
        logger.info(f"[mod={mod}] in_dim={in_dim} domains={domains} classes={nc}")

        for proto in protocols:
            results[mod][proto] = {}
            pairs = [(a, b) for a in domains for b in domains]
            for a, b in pairs:
                key = f"{a}->{b}"
                dsa, dsb = get_ds(a, mod), get_ds(b, mod)
                gains, taccs, saccs = [], [], []
                for run in range(_args.runs):
                    seed = run * 42 + 1
                    # Phase1: 源域全训（两协议共用）
                    tr_s, va_s, te_s = dsa.get_split_masks("supervised", seed=seed)
                    src = GCN2(in_dim, _args.hidden, nc[a], _args.dropout)
                    src, _ = train_gcn(src, dsa.data, tr_s, va_s, _args.epochs, _args.lr, device)
                    # 目标域 from-scratch（对照，两协议共用）
                    tr_t, va_t, te_t = dsb.get_split_masks("supervised", seed=seed)
                    scratch = GCN2(in_dim, _args.hidden, nc[b], _args.dropout)
                    scratch, _ = train_gcn(scratch, dsb.data, tr_t, va_t, _args.epochs, _args.lr, device)
                    se = evaluate(scratch, dsb.data, te_t, device)
                    # Phase2: 迁移
                    tgt = GCN2(in_dim, _args.hidden, nc[b], _args.dropout)
                    tgt.convs[0].load_state_dict(src.convs[0].state_dict())  # 拷源域 conv0
                    if proto == "full_finetune":
                        # 两层都可训（conv0 用源域 init，conv1 新建），全微调
                        tgt, _ = train_gcn(tgt, dsb.data, tr_t, va_t, _args.epochs, _args.lr, device,
                                           freeze_conv0=False)
                    else:  # linear_probe：冻 conv0，只训头
                        tgt, _ = train_gcn(tgt, dsb.data, tr_t, va_t, _args.epochs, _args.lr, device,
                                           freeze_conv0=True)
                    te = evaluate(tgt, dsb.data, te_t, device)
                    taccs.append(te["acc"]); saccs.append(se["acc"]); gains.append(te["acc"] - se["acc"])
                results[mod][proto][key] = {
                    "transfer_acc": float(np.mean(taccs)), "transfer_acc_std": float(np.std(taccs)),
                    "scratch_acc": float(np.mean(saccs)),
                    "transfer_gain": float(np.mean(gains)), "transfer_gain_std": float(np.std(gains)),
                    "diag": a == b,
                }
                logger.info(f"[{mod}/{proto}] {key}: transfer={np.mean(taccs):.3f} "
                            f"scratch={np.mean(saccs):.3f} gain={np.mean(gains):+.3f}")

    out = {
        "task": "cross_domain_transfer_ablation",
        "protocols_tested": protocols, "modalities_tested": modalities, "domains": domains,
        "runs": _args.runs, "epochs": _args.epochs, "hidden": _args.hidden,
        "note": "linear_probe=freeze convs[0] train head; full_finetune=init convs[0] from src, train both. "
                "If gain<0 under BOTH protocols AND text+image => negative transfer is real, not artifact.",
        "results": results, "elapsed_sec": round(time.time() - t0, 1),
    }
    sp = out_dir / "transfer_ablation.json"
    with open(sp, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"保存到 {sp}")


if __name__ == "__main__":
    main()
