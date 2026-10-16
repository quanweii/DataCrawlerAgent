"""
crawlers.steam — Steam 游戏同构图构建 (移植自 scripts/rebuild_steam_graph_10k.py)

流程 (filter-then-build):
1. 扫描 cs_lo 目录得节点属性
2. label = genres[0], 过滤小类 (<20)
3. 在保留节点上建密度优化 co_attribute 边:
   shared_genre(30, 0.30) / same_developer(100) / same_publisher(100) / shared_category(30, 0.40, floor5)
4. 计算 label/label_index/homophily
"""
from collections import defaultdict, Counter

from .common import scan_cslo_attrs, DensityEdgeBuilder, compute_homophily, make_stats

MIN_CLASS = 20


def build_steam_graph(items, cs_lo_dir, opts=None):
    """构建 Steam 图.

    items: list[dict] (steam_games_10k.json), 含 appid/genres/developers/publishers/categories
    cs_lo_dir: cs_lo 目录路径
    opts: 可选覆盖 {min_class, caps, freq_filters}
    返回 (nodes, edges, stats)
    """
    opts = opts or {}
    min_class = opts.get("min_class", MIN_CLASS)
    caps = opts.get("caps", {})
    ff = opts.get("freq_filters", {})

    node_attrs = scan_cslo_attrs(cs_lo_dir)
    id2g = {str(g["appid"]): g for g in items}
    # 节点取 cs_lo 目录 ∩ items: 全量运行时恒等(cslo 全在 items 中),
    # subset 测试(max_games 限制 items)时把节点限制到该子集
    valid = set(node_attrs) & set(id2g)

    # label = genres[0]
    def label_of(appid):
        gs = id2g.get(appid, {}).get("genres") or ["Unknown"]
        return gs[0]

    # 小类过滤
    label_count = Counter(label_of(a) for a in valid)
    small = {l for l, c in label_count.items() if c < min_class}
    keep = {a for a in valid if label_of(a) not in small}

    label_map = {l: i for i, l in enumerate(sorted({label_of(a) for a in keep}))}

    nodes = []
    for a in keep:
        g = id2g.get(a, {})
        nodes.append({
            "id": a,
            "type": "game",
            "attributes": {
                **node_attrs[a],
                "appid": a,
                "name": g.get("name", ""),
                "label": label_of(a),
                "label_index": label_map.get(label_of(a), -1),
            },
        })

    n_total = len(keep)
    builder = DensityEdgeBuilder(valid_ids=keep, n_total=n_total, rng_seed=42)

    g2i = defaultdict(list)
    for a in keep:
        for genre in id2g.get(a, {}).get("genres", []):
            g2i[genre].append(a)
    builder.add_group(g2i, "shared_genre", caps.get("shared_genre", 30),
                      freq_filter=ff.get("shared_genre", 0.30))

    d2i = defaultdict(list)
    for a in keep:
        for d in id2g.get(a, {}).get("developers", []):
            if d:
                d2i[d].append(a)
    builder.add_group(d2i, "same_developer", caps.get("same_developer", 100))

    p2i = defaultdict(list)
    for a in keep:
        for p in id2g.get(a, {}).get("publishers", []):
            if p:
                p2i[p].append(a)
    builder.add_group(p2i, "same_publisher", caps.get("same_publisher", 100))

    c2i = defaultdict(list)
    for a in keep:
        for c in id2g.get(a, {}).get("categories", []):
            c2i[c].append(a)
    builder.add_group(c2i, "shared_category", caps.get("shared_category", 30),
                      freq_filter=ff.get("shared_category", 0.40), floor=5)

    edges = builder.edges
    hom = compute_homophily(nodes, edges)
    stats = make_stats(nodes, edges, label_map, extra={
        "edge_relation_counts": dict(builder.relation_counts),
    })
    stats["homophily"] = round(hom, 3)
    return nodes, edges, stats
