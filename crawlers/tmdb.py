"""
crawlers.tmdb — TMDb 电影同构图构建
(移植自 scripts/rebuild_tmdb_graph.py 的 build_graph + scripts/filter_small_classes.py 的 <20 过滤)

流程 (build-then-filter, 与 TMDb v3 实际管线一致):
1. 扫描 cs_lo 目录得节点属性 (全部有效电影)
2. label = genres[0], 在全部节点上建密度优化边:
   shared_genre(30, 0.30) / same_director(30) / shared_actor(30, top3 cast)
3. 过滤小类 (<20): 删节点 + 关联边, 重建 label_index
4. 计算 homophily
"""
from collections import defaultdict

from .common import scan_cslo_attrs, DensityEdgeBuilder, compute_homophily, make_stats, filter_small_classes

MIN_CLASS = 20


def build_tmdb_graph(items, cs_lo_dir, opts=None):
    """构建 TMDb 图.

    items: list[dict] (tmdb_movies_raw.json), 含 movie_id/genres/directors/actors
    cs_lo_dir: cs_lo 目录路径
    opts: 可选 {min_class, caps, freq_filters}
    返回 (nodes, edges, stats)
    """
    opts = opts or {}
    min_class = opts.get("min_class", MIN_CLASS)
    caps = opts.get("caps", {})
    ff = opts.get("freq_filters", {})

    node_attrs = scan_cslo_attrs(cs_lo_dir)
    id2m = {str(m["movie_id"]): m for m in items}
    # 节点取 cs_lo 目录 ∩ items: 全量运行时恒等, subset 测试时限制到该子集
    valid = set(node_attrs) & set(id2m)

    def label_of(mid):
        gs = id2m.get(mid, {}).get("genres") or ["Unknown"]
        return gs[0]

    # label_map 覆盖所有出现过的流派
    all_genres = set()
    for m in items:
        all_genres.update(m.get("genres", []))
    label_map = {g: i for i, g in enumerate(sorted(all_genres))}

    nodes = []
    for mid, na in node_attrs.items():
        if mid not in valid:
            continue
        m = id2m.get(mid, {})
        label = label_of(mid)
        nodes.append({
            "id": mid,
            "type": "movie",
            "attributes": {
                **na,
                "movie_id": mid,
                "title": m.get("title", ""),
                "label": label,
                "label_index": label_map.get(label, -1),
            },
        })

    n_total = len(valid)
    builder = DensityEdgeBuilder(valid_ids=valid, n_total=n_total, rng_seed=42)

    g2i = defaultdict(list)
    for mid in valid:
        for genre in id2m.get(mid, {}).get("genres", []):
            g2i[genre].append(mid)
    builder.add_group(g2i, "shared_genre", caps.get("shared_genre", 30),
                      freq_filter=ff.get("shared_genre", 0.30))

    d2i = defaultdict(list)
    for mid in valid:
        for d in id2m.get(mid, {}).get("directors", []):
            if d:
                d2i[d].append(mid)
    builder.add_group(d2i, "same_director", caps.get("same_director", 30))

    a2i = defaultdict(list)
    for mid in valid:
        for a in id2m.get(mid, {}).get("actors", [])[:3]:
            if a:
                a2i[a].append(mid)
    builder.add_group(a2i, "shared_actor", caps.get("shared_actor", 30))

    nodes, edges = filter_small_classes(nodes, builder.edges, min_class)

    # 重建 label_map 为过滤后的标签
    kept_labels = sorted(set(n["attributes"]["label"] for n in nodes))
    label_map = {l: i for i, l in enumerate(kept_labels)}

    hom = compute_homophily(nodes, edges)
    stats = make_stats(nodes, edges, label_map, extra={
        "edge_relation_counts": dict(builder.relation_counts),
    })
    stats["homophily"] = round(hom, 3)
    return nodes, edges, stats
