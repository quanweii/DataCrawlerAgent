"""
crawlers.wikiart — WikiArt 画作同构图构建 (移植自 scripts/rebuild_wikiart_v3.py)

流程 (输入已过滤):
1. enriched 数据 (wikiart_v2_enriched.json) 已在 enrich 阶段过滤 <20 小类
2. 扫描 cs_lo 目录得节点属性
3. label = style, 建密度优化边:
   same_artist(100) / shared_style(50)
4. 计算 homophily

id 清洗: painting_id 经 re.sub(r'[^\\w\\-.]', '_', pid) 得到 cs_lo 目录名,
此清洗在本模块统一应用, 保证 collector 与 graph_builder 一致.
"""
import re
from collections import defaultdict

from .common import scan_cslo_attrs, DensityEdgeBuilder, compute_homophily, make_stats

MAX_ARTIST = 100
MAX_STYLE = 50


def sanitize_pid(painting_id):
    """painting_id → cs_lo 目录名 (与 rebuild_wikiart_v3.py 一致)"""
    return re.sub(r'[^\w\-.]', '_', painting_id)


def build_wikiart_graph(items, cs_lo_dir, opts=None):
    """构建 WikiArt 图.

    items: list[dict] (wikiart_v2_enriched.json), 含 painting_id/title/artist/style
    cs_lo_dir: cs_lo 目录路径
    opts: 可选 {max_artist, max_style}
    返回 (nodes, edges, stats)
    """
    opts = opts or {}
    max_artist = opts.get("max_artist", MAX_ARTIST)
    max_style = opts.get("max_style", MAX_STYLE)

    node_attrs = scan_cslo_attrs(cs_lo_dir)
    valid = set(node_attrs)  # 目录名 (已 sanitized)

    pid_to_painting = {}
    for p in items:
        pid = sanitize_pid(p["painting_id"])
        if pid in valid:
            pid_to_painting[pid] = p

    styles = sorted(set(p["style"] for p in pid_to_painting.values() if p.get("style")))
    label_map = {s: i for i, s in enumerate(styles)}

    nodes = []
    for pid, na in node_attrs.items():
        if pid not in pid_to_painting:
            continue
        p = pid_to_painting[pid]
        style = p.get("style", "Unknown")
        nodes.append({
            "id": pid,
            "type": "artwork",
            "attributes": {
                **na,
                "painting_id": pid,
                "title": p.get("title", ""),
                "label": style,
                "label_index": label_map.get(style, -1),
            },
        })

    # 只对有 painting 元数据的节点建边
    valid_nodes = {n["id"] for n in nodes}
    n_total = len(valid_nodes)
    builder = DensityEdgeBuilder(valid_ids=valid_nodes, n_total=n_total, rng_seed=42)

    a2i = defaultdict(list)
    for pid in valid_nodes:
        artist = pid_to_painting.get(pid, {}).get("artist")
        if artist:
            a2i[artist].append(pid)
    builder.add_group(a2i, "same_artist", max_artist)

    s2i = defaultdict(list)
    for pid in valid_nodes:
        style = pid_to_painting.get(pid, {}).get("style")
        if style:
            s2i[style].append(pid)
    builder.add_group(s2i, "shared_style", max_style)

    edges = builder.edges
    hom = compute_homophily(nodes, edges)
    stats = make_stats(nodes, edges, label_map, extra={
        "edge_relation_counts": dict(builder.relation_counts),
    })
    stats["homophily"] = round(hom, 3)
    return nodes, edges, stats
