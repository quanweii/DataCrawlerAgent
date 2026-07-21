#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每类评估（T2⑥）—— 补 per-class Precision/Recall/F1 + 混淆矩阵。

动机：在不平衡域（steam/tmdb/wikiart），macro-F1 比 acc 更说明问题，审稿人会查每类细节。
对头条域训练 GCN + MLP（固定切分 seed=1，5 初始化 seed），dump 每类分类报告 + 混淆矩阵。

用法
  python benchmark/run_per_class_eval.py --domain wikiart --gpu 6 --models gcn,mlp
"""
import os
import sys
import argparse
import json
from pathlib import Path

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--domain", required=True, choices=["steam", "tmdb", "wikiart", "arxiv"])
_pre.add_argument("--gpu", type=int, default=None)
_pre.add_argument("--modality", default="text+image")
_pre.add_argument("--models", default="gcn,mlp")
_pre.add_argument("--seeds", type=int, default=5)
_pre.add_argument("--epochs", type=int, default=300)
_args, _ = _pre.parse_known_args()
if _args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_args.gpu)

import numpy as np
import torch
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from loguru import logger

logger.add(sys.stderr, level="INFO")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.data.domain_loader import SingleDomainDataset
from benchmark.data.splits import stratified_split
from benchmark.models.gnn import get_model
from benchmark.tasks.node_classification import train_and_eval


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = PROJECT_ROOT / "benchmark" / "results" / "per_class_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = SingleDomainDataset(_args.domain, modality=_args.modality)
    y = ds.data.y
    masks = stratified_split(y.cpu(), seed=1)
    tr, va, te = [m.to(device) for m in masks]
    data = ds.data.to(device)
    in_dim = data.x.size(1)
    C = int(y.max().item()) + 1
    label_names = ds.graph["label_names"]
    cls_names = [label_names.get(i, str(i)) for i in range(C)]
    models = [m.strip() for m in _args.models.split(",")]

    result = {"domain": _args.domain, "modality": _args.modality, "classes": C, "per_model": {}}
    for model in models:
        # 训练（固定切分，多个初始化 seed 取平均预测）
        all_preds = np.zeros((int(te.sum()), len(range(_args.seeds))), dtype=object)
        for si, seed in enumerate(range(_args.seeds)):
            torch.manual_seed(seed)
            m = get_model(model, in_dim, 256, C, num_layers=2, dropout=0.5)
            train_and_eval(m, data, tr, va, te, num_epochs=_args.epochs, lr=0.01, device=device)
            m.eval()
            with torch.no_grad():
                pred = m(data.x, data.edge_index).argmax(1).cpu().numpy()
            all_preds[:, si] = pred[te.cpu().numpy()]
        # 多数投票聚合 seed 间预测
        voted = np.array([np.bincount(all_preds[i].astype(int), minlength=C).argmax()
                          for i in range(all_preds.shape[0])])
        truth = y[te.cpu()].numpy()
        p, r, f, sup = precision_recall_fscore_support(truth, voted, zero_division=0)
        rep = classification_report(truth, voted, target_names=cls_names,
                                    zero_division=0, output_dict=True)
        cm = confusion_matrix(truth, voted, labels=list(range(C))).tolist()
        result["per_model"][model] = {
            "macro_f1": rep["macro avg"]["f1-score"],
            "per_class": {cls_names[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                         "f1": float(f[i]), "support": int(sup[i])}
                          for i in range(C)},
            "confusion_matrix": cm, "class_names": cls_names,
        }
        # 打印最差/最好的几个类
        order = sorted(range(C), key=lambda i: f[i])
        logger.info(f"[{_args.domain}/{model}] macro-F1={rep['macro avg']['f1-score']:.3f}")
        logger.info(f"  最差3类: {[(cls_names[i], round(float(f[i]),2)) for i in order[:3]]}")
        logger.info(f"  最好3类: {[(cls_names[i], round(float(f[i]),2)) for i in order[-3:]]}")

    sp = out_dir / f"perclass_{_args.domain}_{_args.modality.replace('+','_')}.json"
    with open(sp, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"保存到 {sp}")


if __name__ == "__main__":
    main()
