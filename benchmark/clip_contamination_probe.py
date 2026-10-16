#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P0-3 CLIP 预训练污染探针（对抗审稿 2026-07-16 P0-3）。

问题
----
wikiart 画作(Starry Night / Water Lilies…)几乎必然出现在 CLIP-L/14 的 LAION 预训练集里，
而 CLIP 对 art style 有强零样本能力 → image 特征可能携带 label(style)信息 = 一种隐蔽泄露。
wikiart image-only MLP NC ≈ 0.65(64类)高得不正常，强烈暗示污染。

判别实验
--------
用**训练集确定不含 wikiart 艺术品**的 ImageNet-1k 监督编码器(ResNet50 / ViT-B/16)重新
编码 wikiart image，在和主表**完全一致**的 leak-free NC 协议(stratified split seed=1 +
跨切分边剪枝 + MLP proj_dim=768 + 5 init seed + 300 epoch)下跑 image-only NC，对比 CLIP-L/14。

判读
----
- CLIP F1 ≫ ImageNet-only F1 (Δ>0.05) → CLIP 高分来自其预训练数据见过 wikiart → 污染成立，
  "image 全域转正"头条叙事需改写为 limitation。
- CLIP F1 ≈ ImageNet-only F1 → wikiart image 视觉上本身可分 → 无污染证据，现有叙事安全。

注意：CLIP 行 in_dim=768==proj_dim → get_model 不投影，等价于主表 image-only MLP，应复现
主表 wikiart image MLP F1（作为脚本正确性自检）。

用法
----
  python benchmark/clip_contamination_probe.py --gpu 0 --encoders resnet50_in1k,vitb16_in1k
"""
import os
import argparse

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", type=int, default=0)
_pre.add_argument("--domain", default="wikiart", choices=["wikiart", "steam", "tmdb"])
_pre.add_argument("--encoders", default="resnet50_in1k,vitb16_in1k")
_pre.add_argument("--seeds", type=int, default=5)
_pre.add_argument("--epochs", type=int, default=300)
_pre.add_argument("--limit", type=int, default=-1, help="只取前 N 节点(smoke test)")
_args, _ = _pre.parse_known_args()
if _args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_args.gpu)

import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch_geometric.data import Data

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROCESSED = PROJECT / "benchmark" / "processed"
DOMAIN = _args.domain

from benchmark.data.domain_loader import load_domain_graph, DOMAIN_RAW, _abs
from benchmark.data.splits import stratified_split, prune_cross_split_edges
from benchmark.models.gnn import get_model
from benchmark.tasks.node_classification import train_and_eval


def load_backbone(name, device):
    import torchvision.models as M
    import torchvision.transforms as T
    if name == "resnet50_in1k":
        from torchvision.models import ResNet50_Weights
        m = M.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        m.fc = torch.nn.Identity()                       # → avgpool 输出 2048d
        tfm = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        dim = 2048
    elif name == "vitb16_in1k":
        from torchvision.models import ViT_B_16_Weights
        w = ViT_B_16_Weights.IMAGENET1K_V1
        m = M.vit_b_16(weights=w)
        try:
            m.heads = torch.nn.Identity()                # → 768d cls 特征
        except Exception:
            m.head = torch.nn.Identity()
        tfm = w.transforms()
        dim = 768
    else:
        raise ValueError(f"unknown encoder {name}")
    return m.to(device).eval(), tfm, dim


def encode_backbone(name, node_ids, device, batch=64):
    """ImageNet-only 编码器编码 wikiart 全部图像 → per-node mean-pool + L2-norm。
    返回 [N, dim] float32，无图节点 = 零向量（与 _load_emb 缺失填零一致）。"""
    raw = _abs(DOMAIN_RAW[DOMAIN])
    model, tfm, dim = load_backbone(name, device)

    # 先收集路径（不一次性载入所有图，省内存）
    paths, nidx = [], []
    for i, nid in enumerate(node_ids):
        d = raw / nid / "images"
        if not d.is_dir():
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                paths.append(d / f)
                nidx.append(i)
    print(f"[{name}] {len(paths)} images across {len(set(nidx))}/{len(node_ids)} nodes")
    if not paths:
        raise RuntimeError(f"{name}: 没找到任何图像")

    feats = np.zeros((len(paths), dim), np.float32)
    for s in tqdm(range(0, len(paths), batch), desc=f"encode {name}"):
        chunk = []
        for p in paths[s:s + batch]:
            try:
                chunk.append(tfm(Image.open(p).convert("RGB")))
            except Exception:
                chunk.append(torch.zeros(3, 224, 224))
        x = torch.stack(chunk).to(device)
        with torch.no_grad():
            feats[s:s + batch] = model(x).cpu().numpy()

    nidx = np.array(nidx)
    out = np.zeros((len(node_ids), dim), np.float32)
    cnt = np.zeros(len(node_ids))
    np.add.at(out, nidx, feats)
    np.add.at(cnt, nidx, 1)
    have = cnt > 0
    out[have] = out[have] / cnt[have, None]
    out = out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-8)
    return out


def mlp_nc(x, y, ei, masks, out_dim, seeds, epochs, device):
    """image-only MLP NC，与主表 leak-free 协议一致（proj_dim=768）。"""
    n = x.size(0)
    data = Data(x=x, edge_index=ei, y=y)
    data.num_nodes = n
    data = data.to(device)
    tr, va, te = [m.to(device) for m in masks]
    f1s, accs = [], []
    for seed in range(seeds):
        torch.manual_seed(seed)
        m = get_model("mlp", x.size(1), 256, out_dim,
                      num_layers=2, dropout=0.5, proj_dim=768)
        m = m.to(device)
        r = train_and_eval(m, data, tr, va, te, num_epochs=epochs, lr=0.01, device=device)
        f1s.append(r["test_f1"]); accs.append(r["test_acc"])
    return float(np.mean(f1s)), float(np.std(f1s)), float(np.mean(accs)), float(np.std(accs))


def main():
    print(f"device={DEVICE}, encoders={_args.encoders}, seeds={_args.seeds}, epochs={_args.epochs}")
    graph = load_domain_graph(DOMAIN)
    node_ids = graph["node_ids"]
    if _args.limit > 0:
        node_ids = node_ids[:_args.limit]
        keep = set(node_ids)
        # 过滤边到子集（smoke test 用）
    y = graph["y"]
    n = graph["num_nodes"]
    masks = stratified_split(y, seed=1)
    ei = prune_cross_split_edges(graph["edge_index"], masks)
    out_dim = int(y.max().item()) + 1
    print(f"wikiart: {n} nodes, {out_dim} classes, pruned edges={ei.size(1)//2}")

    embs = {}

    # --- CLIP baseline（复用已编码的 image_embeddings.pt）---
    clip = torch.load(PROCESSED / DOMAIN / "image_embeddings.pt", map_location="cpu")
    id2row = {s: i for i, s in enumerate(clip["ids"])}
    ce = clip["embeddings"].float()
    cx = torch.zeros(n, ce.size(1))
    found = 0
    for i, nid in enumerate(node_ids):
        if nid in id2row:
            cx[i] = ce[id2row[nid]]; found += 1
    print(f"CLIP aligned {found}/{n}")
    embs["clip-vitl14"] = F.normalize(cx, dim=1)

    # --- ImageNet-only 对照编码器 ---
    for name in [e.strip() for e in _args.encoders.split(",") if e.strip()]:
        try:
            arr = encode_backbone(name, node_ids, DEVICE)
            embs[name] = F.normalize(torch.from_numpy(arr), dim=1)
        except Exception as e:
            print(f"[WARN] {name} 编码失败，跳过: {repr(e)[:200]}")

    # --- image-only MLP NC 对照 ---
    results = {}
    print("\n=== image-only MLP NC (leak-free, proj_dim=768) ===")
    for name, x in embs.items():
        f1m, f1s, am, asd = mlp_nc(x, y, ei, masks, out_dim,
                                   _args.seeds, _args.epochs, DEVICE)
        results[name] = {"f1_mean": f1m, "f1_std": f1s, "acc_mean": am, "acc_std": asd,
                         "dim": int(x.size(1))}
        print(f"  {name:<16} dim={x.size(1):<5} F1={f1m:.3f}±{f1s:.3f}  ACC={am:.3f}±{asd:.3f}")

    # --- 判读 ---
    if "clip-vitl14" in results:
        clip_f1 = results["clip-vitl14"]["f1_mean"]
        print("\n=== CLIP 污染判读 (Δ = CLIP − 对照) ===")
        for name in results:
            if name == "clip-vitl14":
                continue
            delta = clip_f1 - results[name]["f1_mean"]
            if delta > 0.05:
                v = "🔴 CLIP 显著更高 → 疑预训练污染(image 高分可能来自 LAION 见过 wikiart)"
            elif delta < -0.05:
                v = "🟢 对照更高 → 无污染(CLIP 反而弱)"
            else:
                v = "🟢 相当(|Δ|≤0.05) → 无污染证据，wikiart image 视觉本身可分"
            print(f"  CLIP − {name:<14}: Δ={delta:+.3f}  {v}")

    outf = PROJECT / "benchmark" / "results" / f"clip_contamination_probe_{DOMAIN}.json"
    json.dump(results, open(outf, "w"), indent=2)
    print(f"\n✅ saved → {outf}")


if __name__ == "__main__":
    main()
