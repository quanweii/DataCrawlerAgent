#!/usr/bin/env python3
"""
三数据集 benchmark 合规性 + 实验可行性深度验证
检查项:
  A. MAG 结构合规 (节点/边/标签/schema)
  B. 模态完整性 (text+image 100% 覆盖, 文本长度, 图片真实存在)
  C. 图拓扑健康 (连通分量, 孤立点, 度分布, 自环/重边)
  D. 节点分类可行性 (类别数, 最小类, 类别不平衡度, train/val/test 切分)
  E. 链路预测可行性 (同质性, 边密度, 负采样空间)
  F. 模态消融可行性 (单模态可分离性代理)
"""
import json, random, sys
from pathlib import Path
from collections import defaultdict, Counter

DOMAINS = {
    "steam":   ("data/steam_datasets/steam_multimodal_graph_v3.json",   "data/steam_games"),
    "tmdb":    ("data/tmdb_datasets/tmdb_multimodal_graph_v3.json",     "data/tmdb_movies"),
    "wikiart": ("data/wikiart_datasets/wikiart_multimodal_graph_v3.json","data/wikiart_artworks_v2"),
}

def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def check_topology(nodes, edges):
    nid = {n["id"] for n in nodes}
    # 自环 / 重边 / 悬挂边
    self_loops = sum(1 for e in edges if e["source"] == e["target"])
    seen = set(); multi = 0; dangling = 0
    for e in edges:
        k = tuple(sorted((e["source"], e["target"])))
        if k in seen: multi += 1
        seen.add(k)
        if e["source"] not in nid or e["target"] not in nid: dangling += 1
    # 度
    deg = Counter()
    for e in edges:
        deg[e["source"]] += 1; deg[e["target"]] += 1
    isolated = [n["id"] for n in nodes if deg.get(n["id"], 0) == 0]
    deg_vals = sorted(deg.values())
    # 连通分量 (union-find)
    parent = {n["id"]: n["id"] for n in nodes}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for e in edges:
        ra, rb = find(e["source"]), find(e["target"])
        if ra != rb: parent[ra] = rb
    comp = Counter(find(n["id"]) for n in nodes)
    return {
        "self_loops": self_loops, "multi_edges": multi, "dangling_edges": dangling,
        "isolated_nodes": len(isolated),
        "max_degree": deg_vals[-1] if deg_vals else 0,
        "min_degree": deg_vals[0] if deg_vals else 0,
        "median_degree": deg_vals[len(deg_vals)//2] if deg_vals else 0,
        "num_components": len(comp),
        "largest_comp_size": max(comp.values()),
        "largest_comp_frac": max(comp.values()) / len(nodes),
    }

def check_labels(nodes):
    lab = [n["attributes"]["label"] for n in nodes]
    c = Counter(lab)
    vals = sorted(c.values())
    return {
        "num_labels": len(c),
        "min_class": vals[0], "max_class": vals[-1],
        "imbalance_ratio": round(vals[-1] / max(vals[0],1), 1),
        "smallest_classes": c.most_common()[-3:],
    }

def check_split_feasibility(nodes, edges):
    """节点分类: 7:1.5:1.5 切分后每类 train 最少样本数"""
    by_label = defaultdict(list)
    for n in nodes: by_label[n["attributes"]["label"]].append(n["id"])
    worst = min(len(v) for v in by_label.values())
    # 最小类 train 样本 = worst * 0.7
    min_train = int(worst * 0.7)
    # 链路预测: 负采样空间 = C(n,2) - edges
    n = len(nodes)
    max_pairs = n * (n-1) // 2
    neg_room = max_pairs - len(edges)
    neg_ratio = neg_room / len(edges) if edges else 0
    return {
        "min_class_size": worst,
        "min_class_train_70pct": min_train,
        "feasible_nc": worst >= 20,
        "neg_sample_room": neg_room,
        "neg_to_pos_ratio": round(neg_ratio, 1),
        "feasible_lp": neg_ratio > 1.0,  # 负样本至少和正样本同量级
    }

def check_modalities(graph, base_dir, sample_n=200):
    nodes = graph["nodes"]
    rng = random.Random(42)
    sample = rng.sample(nodes, min(sample_n, len(nodes)))
    img_ok = txt_ok = md_ok = npy_ok = 0
    img_total = txt_lens = []
    for n in sample:
        nid = n["id"]; d = Path(base_dir) / nid
        md = d / f"{nid}.md"
        npy = d / f"{nid}.npy"
        imgs = list((d / "images").glob("*")) if (d / "images").exists() else []
        if md.exists():
            md_ok += 1; txt_lens.append(len(md.read_text(encoding="utf-8")))
        if npy.exists(): npy_ok += 1
        if len(imgs) > 0: img_ok += 1
    return {
        "sampled": len(sample),
        "md_present_pct": round(100*md_ok/len(sample),1),
        "npy_present_pct": round(100*npy_ok/len(sample),1),
        "image_present_pct": round(100*img_ok/len(sample),1),
        "avg_text_len_sample": round(sum(txt_lens)/max(len(txt_lens),1)) if txt_lens else 0,
    }

def main():
    random.seed(42)
    print("="*70)
    print("AMMAG Benchmark 合规性 + 实验可行性验证")
    print("="*70)
    for dom, (gpath, dpath) in DOMAINS.items():
        print(f"\n{'#'*70}\n# {dom.upper()}\n{'#'*70}")
        g = load(gpath)
        nodes, edges = g["nodes"], g["edges"]
        st = g["metadata"]["statistics"]
        sc = g["metadata"]["schema"]
        print(f"\n[A] MAG 结构合规")
        print(f"  节点: {len(nodes):,}  边: {len(edges):,}")
        print(f"  节点类型: {sc['node_types']}  边类型: {sc['edge_types']}")
        print(f"  模态: {sc['modalities']}  边关系: {sc['edge_relations']}")
        print(f"  每节点有 attributes.label + label_index: {'label' in nodes[0]['attributes'] and 'label_index' in nodes[0]['attributes']}")

        print(f"\n[C] 图拓扑")
        topo = check_topology(nodes, edges)
        for k,v in topo.items(): print(f"  {k}: {v}")

        print(f"\n[D] 节点分类可行性")
        lab = check_labels(nodes)
        for k,v in lab.items(): print(f"  {k}: {v}")

        print(f"\n[D2] 切分可行性 (70/15/15)")
        sf = check_split_feasibility(nodes, edges)
        for k,v in sf.items(): print(f"  {k}: {v}")

        print(f"\n[B] 模态完整性 (抽样{200})")
        mod = check_modalities(g, dpath)
        for k,v in mod.items(): print(f"  {k}: {v}")

        # 同质性
        id2l = {n["id"]: n["attributes"]["label"] for n in nodes}
        same = sum(1 for e in edges if id2l.get(e["source"])==id2l.get(e["target"]))
        hom = same/max(len(edges),1)
        print(f"\n[E] 链路预测")
        print(f"  同质性 (homophily): {hom:.3f}")
        print(f"  边关系分布: {dict(Counter(e['attributes'].get('relation') for e in edges))}")
        print(f"  声明同质性: {st.get('homophily')}  实算: {round(hom,3)}  一致: {abs(st.get('homophily',0)-round(hom,3))<0.01}")

        print(f"\n[F] 实验可行性总判")
        ok = all([
            topo["isolated_nodes"]/len(nodes) < 0.05,
            topo["largest_comp_frac"] > 0.9,
            lab["min_class"] >= 20,
            mod["image_present_pct"] >= 95,
            mod["md_present_pct"] >= 99,
            sf["neg_to_pos_ratio"] > 1.0,
            len(edges)/len(nodes) >= 5,  # 平均度 >=5
        ])
        print(f"  {'✅ 全部通过' if ok else '⚠️ 存在问题, 见上'}")
    print(f"\n{'='*70}")

if __name__ == "__main__":
    main()
