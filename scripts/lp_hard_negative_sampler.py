#!/usr/bin/env python3
"""
链路预测硬负采样工具 — 为 benchmark 实验生成严格(非平凡)负样本。

问题背景
--------
同质性(homophily)高的图上, 传统的「随机负采样」会让链路预测任务退化:
随机抽两个节点组成的负对几乎总是跨标签, 于是「同标签就预测有边」这一
平凡启发式即可拿到接近 1 的 AUC, GNN 学不到任何东西。

WikiArt 尤为严重: shared_style 边按定义连接同类节点(同质性 = 1.0),
随机负采样的 baseline AUC ≈ 0.947。详见 docs/wikiart_lp_protocol.md。

本工具提供三种负采样策略
------------------------
  random       均匀随机非边。作为 baseline/对照在论文中报告。
  hard         同标签约束的非边。主协议, 消除「标签一致」这一捷径。
  hard_degree  同标签 + 同度数桶的非边。最严格, 同时消除「度数」捷径。

输出 (data/{domain}_lp/)
  train_pos.csv / train_neg.csv            (random)
  val_pos.csv   / val_neg.csv              (random)
  val_neg_hard.csv                         (hard)
  val_neg_hard_degree.csv                  (hard_degree)
  test_pos.csv  / test_neg.csv             (random)
  test_neg_hard.csv
  test_neg_hard_degree.csv
  protocol.json                            (策略/seed/统计, 供实验复现)

用法
  python scripts/lp_hard_negative_sampler.py --domain wikiart --seed 42
  python scripts/lp_hard_negative_sampler.py --domain tmdb   --ratio 1
"""
import argparse
import csv
import json
import math
import random
from pathlib import Path
from collections import defaultdict, Counter
from loguru import logger

DOMAINS = {
    "steam":   "data/steam_datasets/steam_multimodal_graph_v3.json",
    "tmdb":    "data/tmdb_datasets/tmdb_multimodal_graph_v3.json",
    "wikiart": "data/wikiart_datasets/wikiart_multimodal_graph_v3.json",
}

# 默认切分比例 (OGB 风格): 训练 80% / 验证 10% / 测试 10%
DEFAULT_VAL_RATIO = 0.10
DEFAULT_TEST_RATIO = 0.10


def load_graph(path):
    with open(path, encoding="utf-8") as f:
        g = json.load(f)
    return g["nodes"], g["edges"]


def build_structures(nodes, edges):
    """构建邻接集合、标签分组、度数分桶。"""
    ids = [n["id"] for n in nodes]
    id_set = set(ids)
    id2label = {n["id"]: n["attributes"]["label"] for n in nodes}
    label2nodes = defaultdict(list)
    for n in nodes:
        label2nodes[n["attributes"]["label"]].append(n["id"])

    # 邻接: 无向, 标准化 key = (min, max)
    adj = set()
    degree = Counter()
    for e in edges:
        s, t = e["source"], e["target"]
        if s == t:
            continue
        k = (s, t) if s < t else (t, s)
        adj.add(k)
        degree[s] += 1
        degree[ t] += 1
    return id_set, id2label, label2nodes, adj, degree


def log_degree_buckets(degree, n_buckets=10):
    """把节点按度数分到 n_buckets 个对数桶, 返回 node->bucket_idx。"""
    if not degree:
        return {}
    maxd = max(degree.values())
    if maxd <= 1:
        return {k: 0 for k in degree}
    buckets = {}
    for k, d in degree.items():
        idx = min(n_buckets - 1, int(math.log2(max(d, 1)) / math.log2(maxd) * (n_buckets - 1)))
        buckets[k] = idx
    return buckets


def split_edges(edges, val_ratio, test_ratio, seed):
    """边切分: 随机分配到 train/val/test。返回三组边列表。"""
    rng = random.Random(seed)
    idx = list(range(len(edges)))
    rng.shuffle(idx)
    n_val = int(len(idx) * val_ratio)
    n_test = int(len(idx) * test_ratio)
    val_idx = set(idx[:n_val])
    test_idx = set(idx[n_val:n_val + n_test])
    train, val, test = [], [], []
    for i, e in enumerate(edges):
        if i in val_idx:
            val.append(e)
        elif i in test_idx:
            test.append(e)
        else:
            train.append(e)
    return train, val, test


# ---- 三种负采样器 ----

class RandomNegSampler:
    """均匀随机非边。"""
    name = "random"
    def __init__(self, id_list, adj, rng):
        self.ids = id_list
        self.n = len(id_list)
        self.adj = adj
        self.rng = rng
        self.seen = set()

    def sample(self, count):
        out = []
        attempts = 0
        cap = count * 200 + 1000
        while len(out) < count and attempts < cap:
            attempts += 1
            a = self.ids[self.rng.randrange(self.n)]
            b = self.ids[self.rng.randrange(self.n)]
            if a == b:
                continue
            k = (a, b) if a < b else (b, a)
            if k in self.adj or k in self.seen:
                continue
            self.seen.add(k)
            out.append((a, b))
        return out


class HardNegSampler:
    """同标签约束的非边 (硬负采样, 主协议)。"""
    name = "hard"
    def __init__(self, label2nodes, adj, rng):
        self.groups = {l: v for l, v in label2nodes.items() if len(v) >= 2}
        self.adj = adj
        self.rng = rng
        self.seen = set()
        # 每个标签组内可用非边的近似量 ∝ C(|g|,2) - 内部边数, 用作采样权重
        internal_edges = defaultdict(int)
        for (a, b) in adj:
            # 这里只统计同标签的内部边数(近似), 需 label 一致
            # 但我们没存 label on edge, 跳过精确; 用组大小平方作权重足够
            pass
        self.labels = list(self.groups.keys())
        sizes = [len(self.groups[l]) for l in self.labels]
        tot = sum(s * s for s in sizes) or 1
        self.weights = [s * s / tot for s in sizes]

    def sample(self, count):
        out = []
        attempts = 0
        cap = count * 300 + 2000
        while len(out) < count and attempts < cap:
            attempts += 1
            label = self.rng.choices(self.labels, weights=self.weights, k=1)[0]
            grp = self.groups[label]
            a = grp[self.rng.randrange(len(grp))]
            b = grp[self.rng.randrange(len(grp))]
            if a == b:
                continue
            k = (a, b) if a < b else (b, a)
            if k in self.adj or k in self.seen:
                continue
            self.seen.add(k)
            out.append((a, b))
        return out


class HardDegreeNegSampler:
    """同标签 + 同度数桶的非边 (最严格)。"""
    name = "hard_degree"
    def __init__(self, label2nodes, adj, degree, rng):
        buckets = log_degree_buckets(degree)
        # (label, bucket) -> [nodes]
        cell = defaultdict(list)
        for n, d in degree.items():
            pass
        # 用所有节点(含孤立)
        for l, nodes_ in label2nodes.items():
            for nd in nodes_:
                b = buckets.get(nd, 0)
                cell[(l, b)].append(nd)
        self.cells = {k: v for k, v in cell.items() if len(v) >= 2}
        self.adj = adj
        self.rng = rng
        self.seen = set()
        self.keys = list(self.cells.keys())
        sizes = [len(self.cells[k]) for k in self.keys]
        tot = sum(s * s for s in sizes) or 1
        self.weights = [s * s / tot for s in sizes]

    def sample(self, count):
        out = []
        attempts = 0
        cap = count * 400 + 2000
        while len(out) < count and attempts < cap:
            attempts += 1
            key = self.rng.choices(self.keys, weights=self.weights, k=1)[0]
            grp = self.cells[key]
            a = grp[self.rng.randrange(len(grp))]
            b = grp[self.rng.randrange(len(grp))]
            if a == b:
                continue
            k = (a, b) if a < b else (b, a)
            if k in self.adj or k in self.seen:
                continue
            self.seen.add(k)
            out.append((a, b))
        return out


def write_csv(path, pairs):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "target"])
        for a, b in pairs:
            w.writerow([a, b])


def run_domain(domain, seed, ratio, val_ratio, test_ratio):
    gpath = DOMAINS[domain]
    out_dir = Path("data") / f"{domain}_lp"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"{'='*60}\n[{domain}] 加载 {gpath}")
    nodes, edges = load_graph(gpath)
    logger.info(f"节点 {len(nodes):,}  边 {len(edges):,}")

    id_set, id2label, label2nodes, adj, degree = build_structures(nodes, edges)
    id_list = [n["id"] for n in nodes]

    # 1. 切分正样本边
    train_e, val_e, test_e = split_edges(edges, val_ratio, test_ratio, seed)
    train_pos = [(e["source"], e["target"]) for e in train_e]
    val_pos = [(e["source"], e["target"]) for e in val_e]
    test_pos = [(e["source"], e["target"]) for e in test_e]
    logger.info(f"正样本切分  train={len(train_pos):,} val={len(val_pos):,} test={len(test_pos):,}")

    # 2. 生成负样本 (val/test 各生成 random + hard + hard_degree; train 用 random)
    rng = random.Random(seed)
    def neg_count(n_pos):
        return int(n_pos * ratio)

    # random
    s_rand_train = RandomNegSampler(id_list, adj, random.Random(seed + 1))
    train_neg = s_rand_train.sample(neg_count(len(train_pos)))
    s_rand_val = RandomNegSampler(id_list, adj, random.Random(seed + 2))
    val_neg = s_rand_val.sample(neg_count(len(val_pos)))
    s_rand_test = RandomNegSampler(id_list, adj, random.Random(seed + 3))
    test_neg = s_rand_test.sample(neg_count(len(test_pos)))

    # hard (同标签)
    val_neg_hard = HardNegSampler(label2nodes, adj, random.Random(seed + 4)).sample(neg_count(len(val_pos)))
    test_neg_hard = HardNegSampler(label2nodes, adj, random.Random(seed + 5)).sample(neg_count(len(test_pos)))

    # hard_degree (同标签+同度数桶)
    val_neg_hd = HardDegreeNegSampler(label2nodes, adj, degree, random.Random(seed + 6)).sample(neg_count(len(val_pos)))
    test_neg_hd = HardDegreeNegSampler(label2nodes, adj, degree, random.Random(seed + 7)).sample(neg_count(len(test_pos)))

    # 3. 写文件
    write_csv(out_dir / "train_pos.csv", train_pos)
    write_csv(out_dir / "train_neg.csv", train_neg)
    write_csv(out_dir / "val_pos.csv", val_pos)
    write_csv(out_dir / "val_neg.csv", val_neg)
    write_csv(out_dir / "val_neg_hard.csv", val_neg_hard)
    write_csv(out_dir / "val_neg_hard_degree.csv", val_neg_hd)
    write_csv(out_dir / "test_pos.csv", test_pos)
    write_csv(out_dir / "test_neg.csv", test_neg)
    write_csv(out_dir / "test_neg_hard.csv", test_neg_hard)
    write_csv(out_dir / "test_neg_hard_degree.csv", test_neg_hd)

    # 4. 诊断: 计算 random vs hard 负样本的「同标签比例」(越接近正样本越好)
    def hom_share(pairs):
        if not pairs:
            return 0.0
        same = sum(1 for a, b in pairs if id2label.get(a) == id2label.get(b))
        return same / len(pairs)

    stats = {
        "domain": domain,
        "seed": seed,
        "neg_pos_ratio": ratio,
        "split": {"train_pos": len(train_pos), "val_pos": len(val_pos), "test_pos": len(test_pos),
                  "val_neg": len(val_neg), "val_neg_hard": len(val_neg_hard), "val_neg_hard_degree": len(val_neg_hd),
                  "test_neg": len(test_neg), "test_neg_hard": len(test_neg_hard), "test_neg_hard_degree": len(test_neg_hd)},
        "label_agreement": {  # 负样本里「恰好同标签」的比例; random 应≈1/num_labels, hard≈1.0
            "test_pos": round(hom_share(test_pos), 3),
            "test_neg_random": round(hom_share(test_neg), 3),
            "test_neg_hard": round(hom_share(test_neg_hard), 3),
            "test_neg_hard_degree": round(hom_share(test_neg_hd), 3),
        },
        "num_labels": len(label2nodes),
        "edge_relations": dict(Counter(e["attributes"].get("relation") for e in edges)),
    }
    with open(out_dir / "protocol.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(f"标签一致率 (test集): pos={stats['label_agreement']['test_pos']}  "
                f"neg_random={stats['label_agreement']['test_neg_random']}  "
                f"neg_hard={stats['label_agreement']['test_neg_hard']}")
    logger.info(f"  → random 负样本同标签率远低于正样本 = 存在泄漏, 必须用 hard")
    logger.info(f"✅ 写入 {out_dir}/")
    logger.info(f"{'='*60}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=list(DOMAINS) + ["all"], default="all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratio", type=float, default=1.0, help="负:正比例")
    ap.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO)
    ap.add_argument("--test_ratio", type=float, default=DEFAULT_TEST_RATIO)
    args = ap.parse_args()

    targets = list(DOMAINS) if args.domain == "all" else [args.domain]
    all_stats = {}
    for dom in targets:
        all_stats[dom] = run_domain(dom, args.seed, args.ratio, args.val_ratio, args.test_ratio)

    # 汇总
    logger.info("汇总 (test 集 label-agreement):")
    for dom, s in all_stats.items():
        la = s["label_agreement"]
        logger.info(f"  {dom:9s} pos={la['test_pos']}  neg_random={la['test_neg_random']}  "
                    f"neg_hard={la['test_neg_hard']}")


if __name__ == "__main__":
    main()
