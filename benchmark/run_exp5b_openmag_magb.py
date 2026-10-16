#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exp5b — 现存 MAG benchmark（OpenMAG）真图 leak-profile（2026-07-31）

堵"leak-free 靠 C2 自证循环"指控：E1 只验探测器在 PyG 公开图有效；本实验对**同产物
（真实多模态属性图 benchmark）**直接跑论文 C2（structure-leakage ceiling 原始→剪枝 vs majority）。

数据源（hf-mirror）：enjun-collab/MMAG 的 **分解张量**（nc_edges-nodeid.pt / labels-w-missing.pt
/ split.pt）——普通 torch list/dict，**无需 dgl**（OpenMAG 官方 *Graph.pt 是 DGL 格式，
dgl 2.1.0 的 graphbolt C++ 库未为 torch 2.8 编译 → 绕开）。
当前覆盖 books-nc / ele-fashion（仓库为这两域提供了分解张量）。

输出 benchmark/results/exp5b_external_mag_leakprofile.{json,md}（与自家 4 域 + E1 同表）。
"""
import os, sys, json
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark.data.splits import stratified_split, prune_cross_split_edges, label_propagation_ceiling

CACHE = ROOT / "benchmark" / "processed" / "exp5b_external"
CACHE.mkdir(parents=True, exist_ok=True)
OUT_JSON = ROOT / "benchmark" / "results" / "exp5b_external_mag_leakprofile.json"
OUT_MD = ROOT / "benchmark" / "results" / "exp5b_external_mag_leakprofile.md"

REPO = "enjun-collab/MMAG"
TARGETS = ["books-nc", "ele-fashion"]


def fetch(path_in_repo):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=REPO, filename=path_in_repo, repo_type="dataset",
                           cache_dir=str(CACHE))


def load_decomposed(dset):
    edges_p = fetch(f"{dset}/nc_edges-nodeid.pt")
    lab_p = fetch(f"{dset}/labels-w-missing.pt")
    split_p = fetch(f"{dset}/split.pt")
    edges = torch.load(edges_p, map_location="cpu", weights_only=False)      # list[[s,d]]
    labels = torch.load(lab_p, map_location="cpu", weights_only=False)       # list[int]
    split = torch.load(split_p, map_location="cpu", weights_only=False)      # dict
    ei = torch.tensor(edges, dtype=torch.long).T.contiguous()                # [2,E]
    y = torch.tensor(labels, dtype=torch.long)
    return y, ei, split


def masks_from_idx(n, idx_list):
    m = torch.zeros(n, dtype=torch.bool)
    m[torch.tensor(idx_list, dtype=torch.long)] = True
    return m


def profile(name, y, ei, split):
    n = y.numel()
    # 缺失标签（-1）→ 限定到有标签节点子图（与论文一致：NC 只在带标签节点上）
    keep = (y >= 0)
    if keep.sum() < n:
        idxmap = torch.full((n,), -1, dtype=torch.long)
        idxmap[keep] = torch.arange(keep.sum())
        m = keep[ei[0]] & keep[ei[1]]
        ei = idxmap[ei[:, m]]
        y = y[keep]
        n = int(keep.sum())
    ei = ei[:, ei[0] != ei[1]]            # 去自环
    ei = torch.unique(ei, dim=1)          # 去重
    ei = torch.cat([ei, ei[[1, 0]]], 0)   # 无向
    n_cls = int(y.max().item()) + 1
    same = (y[ei[0]] == y[ei[1]]).float().mean().item() if ei.numel() else 0.0

    out = {"name": name, "nodes": n, "classes": n_cls,
           "edges": int(ei.size(1) // 2), "homophily": round(same, 3)}

    for split_tag, (tr, va, te) in [
        ("ours_stratified", stratified_split(y, ratios=(0.6, 0.2, 0.2), seed=1, small_class_floor=5)),
        ("theirs_shipped",  (masks_from_idx(n, split["train_idx"]),
                             masks_from_idx(n, split.get("val_idx", [])),
                             masks_from_idx(n, split["test_idx"])))
    ]:
        raw = label_propagation_ceiling(y, ei, tr, te)
        ei_pr = prune_cross_split_edges(ei, [tr, va, te])
        pruned = label_propagation_ceiling(y, ei_pr, tr, te)
        out[f"{split_tag}_lp_raw"] = round(raw["acc"], 3)
        out[f"{split_tag}_lp_pruned"] = round(pruned["acc"], 3)
        out[f"{split_tag}_majority"] = round(pruned["majority_acc"], 3)
        out[f"{split_tag}_leakage"] = round(raw["acc"] - pruned["majority_acc"], 3)
    return out


def main():
    rows = []
    for d in TARGETS:
        try:
            print(f"[exp5b] {d}: fetch+load ...", flush=True)
            y, ei, split = load_decomposed(d)
            print(f"  {d}: {y.numel()} nodes, {ei.size(1)} directed edges, "
                  f"split {len(split['train_idx'])}/{len(split.get('val_idx',[]))}/{len(split['test_idx'])}", flush=True)
            r = profile(d, y, ei, split)
            print(f"  ⇒ {r}", flush=True)
            rows.append(r)
        except Exception as e:
            print(f"  [SKIP] {d}: {repr(e)[:160]}", flush=True)
            rows.append({"name": d, "error": repr(e)[:160]})

    lpt = ROOT / "benchmark" / "results" / "leak_profile_table.json"
    ours = {d: v for d, v in (json.load(open(lpt)).items() if lpt.exists() else [])
            if d in ["arxiv", "steam", "tmdb", "wikiart"]}

    out = {"experiment": "exp5b_external_mag_leakprofile", "source": REPO, "external": rows,
           "ours_4gold": {d: {"lp_ceiling_original": v.get("lp_ceiling_original"),
                              "lp_ceiling_pruned": v.get("lp_ceiling_pruned"),
                              "majority_acc": v.get("majority_acc"), "homophily": v.get("homophily")}
                          for d, v in ours.items()}}
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    md = ["# Exp5b — 现存 MAG benchmark（OpenMAG）真图 leak-profile（2026-07-31）\n",
          f"源：`{REPO}` 分解张量（books-nc / ele-fashion）。对**同产物**跑论文 C2（structure-leakage ceiling 原始→剪枝 vs majority）。\n",
          "## 外部 MAG（OpenMAG，leak-profile）",
          "| 图 | nodes | cls | edges | homo | split | LP-raw | LP-pruned | majority | **leakage(raw−maj)** |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            md.append(f"| {r['name']} | - | - | - | - | - | - | - | - | ERR |")
            continue
        for tag in ["ours_stratified", "theirs_shipped"]:
            md.append(f"| {r['name']} | {r['nodes']} | {r['classes']} | {r['edges']} | {r['homophily']} | {tag} | "
                      f"{r[f'{tag}_lp_raw']} | {r[f'{tag}_lp_pruned']} | {r[f'{tag}_majority']} | "
                      f"**{r[f'{tag}_leakage']:+.3f}** |")
    md.append("\n## 自家 4 金标准域对照（C2 去泄露后）")
    md.append("| 域 | homo | LP-raw | → LP-pruned | majority | leakage(raw−maj) |")
    md.append("|---|---|---|---|---|---|")
    for d, v in ours.items():
        leak = (v.get("lp_ceiling_original") or 0) - (v.get("majority_acc") or 0)
        md.append(f"| {d} | {v.get('homophily')} | {v.get('lp_ceiling_original')} | "
                  f"{v.get('lp_ceiling_pruned')} | {v.get('majority_acc')} | **{leak:+.3f}** |")
    md.append("\n> 解读：外部 MAG 若 leakage(raw−maj) 显著正（如自家去泄露前 wikiart +0.90 / E1 Amazon +0.66）"
              "= ship 结构泄露 → 证领域金标准系统性漏掉，C2 去泄露价值不只自家、是同产物维度。")
    OUT_MD.write_text("\n".join(md))
    print(f"\nwrote {OUT_JSON}\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
