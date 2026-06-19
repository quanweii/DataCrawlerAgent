"""
crawlers.common — 三域共用的建图原语
- scan_cslo_attrs: 扫描 cs_lo 目录得到节点文本/图像属性
- build_density_edges: 密度优化边构建 (移植自 rebuild_* 脚本的 build/build_attr_edges)
- compute_homophily: 同质性
- filter_small_classes: 小类过滤 (节点 + 关联边)
"""
import random
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations


def scan_cslo_attrs(cs_lo_dir):
    """扫描 cs_lo 目录, 返回 {node_id: {markdown_length, image_count, markdown_size}}"""
    cs_lo_dir = Path(cs_lo_dir)
    attrs = {}
    if not cs_lo_dir.exists():
        return attrs
    for d in cs_lo_dir.iterdir():
        if not d.is_dir():
            continue
        nid = d.name
        md = d / f"{nid}.md"
        if not md.exists():
            continue
        text = md.read_text(encoding="utf-8")
        img_dir = d / "images"
        ic = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        attrs[nid] = {
            "markdown_length": len(text),
            "image_count": ic,
            "markdown_size": md.stat().st_size,
        }
    return attrs


class DensityEdgeBuilder:
    """跨关系共享 edge_set 的密度优化边构建器 (移植自 rebuild 脚本).

    关键: 所有关系共用一个 edge_set, 因此若两节点既同 genre 又同 developer,
    只会产生一条边 (取先处理的关系). 这与原始脚本完全一致.

    对每个 属性值→节点列表 的组:
    - 过滤高频组 (len > freq_filter * n_total) 与小组 (len < floor)
    - 组内 <= cap: 全连接 (combinations)
    - 组内 >  cap: 随机采样 cap 个全连接, 剩余节点各连一条到采样集
    """

    def __init__(self, valid_ids, n_total, rng_seed=42):
        self.valid_ids = valid_ids
        self.n_total = n_total
        self.rng_seed = rng_seed
        self.edges = []
        self.edge_set = set()
        self.relation_counts = defaultdict(int)

    def _add(self, a, b, relation):
        k = tuple(sorted([a, b]))
        if k not in self.edge_set and k[0] != k[1]:
            self.edge_set.add(k)
            self.edges.append({
                "source": k[0], "target": k[1],
                "type": "co_attribute",
                "attributes": {"relation": relation},
            })
            self.relation_counts[relation] += 1

    def add_group(self, group_to_ids, relation, cap, freq_filter=None, floor=2):
        """为一个关系构建边, 返回被高频过滤的组数"""
        filtered = 0
        for _val, ids in group_to_ids.items():
            ids = [i for i in ids if i in self.valid_ids]
            if freq_filter is not None and len(ids) > freq_filter * self.n_total:
                filtered += 1
                continue
            if len(ids) < floor or len(ids) < 2:
                continue
            if len(ids) <= cap:
                for a, b in combinations(ids, 2):
                    self._add(a, b, relation)
            else:
                sampled = random.Random(self.rng_seed).sample(ids, cap)
                for a, b in combinations(sampled, 2):
                    self._add(a, b, relation)
                for rid in set(ids) - set(sampled):
                    self._add(rid, random.Random(hash(rid) % 2 ** 32).choice(sampled), relation)
        return filtered


def compute_homophily(nodes, edges):
    """同质性: 同标签边 / 总边"""
    id2l = {n["id"]: n["attributes"]["label"] for n in nodes}
    if not edges:
        return 0.0
    same = sum(1 for e in edges if id2l.get(e["source"]) == id2l.get(e["target"]))
    return same / len(edges)


def filter_small_classes(nodes, edges, min_class):
    """小类过滤: 删除 label 计数 < min_class 的节点及其关联边, 重建 label_index"""
    if not min_class or min_class <= 1:
        return nodes, edges
    label_count = Counter(n["attributes"]["label"] for n in nodes)
    small = {l for l, c in label_count.items() if c < min_class}
    if not small:
        return nodes, edges
    keep_ids = {n["id"] for n in nodes if n["attributes"]["label"] not in small}
    new_nodes = [n for n in nodes if n["id"] in keep_ids]
    # 重建 label_index
    new_labels = sorted(set(n["attributes"]["label"] for n in new_nodes))
    label_map = {l: i for i, l in enumerate(new_labels)}
    for n in new_nodes:
        n["attributes"]["label_index"] = label_map[n["attributes"]["label"]]
    new_edges = [e for e in edges if e["source"] in keep_ids and e["target"] in keep_ids]
    return new_nodes, new_edges


def make_stats(nodes, edges, label_map=None, extra=None):
    """组装 v3 风格 statistics 字典"""
    ld = Counter(n["attributes"]["label"] for n in nodes)
    rel_counts = Counter(e["attributes"].get("relation", "") for e in edges)
    hom = compute_homophily(nodes, edges)
    n_nodes = len(nodes)
    total_img = sum(n["attributes"].get("image_count", 0) for n in nodes)
    total_md = sum(n["attributes"].get("markdown_length", 0) for n in nodes)
    stats = {
        "node_count": n_nodes,
        "edge_count": len(edges),
        "total_images": total_img,
        "nodes_with_images": sum(1 for n in nodes if n["attributes"].get("image_count", 0) > 0),
        "total_markdown_chars": total_md,
        "avg_md_length": round(total_md / max(n_nodes, 1), 0),
        "avg_degree": round(2 * len(edges) / max(n_nodes, 1), 1),
        "homophily": round(hom, 3),
        "label_distribution": dict(ld),
        "num_labels": len(label_map) if label_map is not None else len(ld),
        "edge_relation_counts": dict(rel_counts),
    }
    if extra:
        stats.update(extra)
    return stats
