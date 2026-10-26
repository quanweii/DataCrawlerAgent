"""
crawlers.config_driven — 通用配置驱动 co_attribute 图构建。

将 steam/tmdb/wikiart 三域的重复建图逻辑提取为单一通用函数，
用户只需提供描述数据 schema 和关系定义的 config dict。

支持三种过滤时序:
  - "filter-then-build" (steam 模式): 先过滤小类, 再用保留节点建边
  - "build-then-filter" (tmdb 模式): 先建边, 再过滤小类节点+关联边
  - "pre-filtered" (wikiart 模式): 输入已预过滤, 直接建边

config 结构:
{
    "id_field": "appid",                    # item 中的唯一 id 字段名
    "id_sanitizer": None,                   # 可选: id → 目录名的清洗函数
    "label_field": "genres[0]",             # 标签字段, 支持 "a.b" 或 "a[0]" 路径
    "min_class": 20,                        # 小类过滤阈值 (0=不过滤)
    "build_order": "filter-then-build",     # 过滤时序
    "relations": [
        {
            "name": "shared_genre",         # 关系名
            "group_by": "genres",           # 分组字段, 支持 "a.b" 或 "a[0]"
            "field_slice": None,            # 可选: 对字段值切片 (e.g. ":3")
            "cap": 30,                      # 组内最大全连接数
            "freq_filter": 0.30,            # 可选: >此比例的组被跳过
            "floor": 2,                     # 可选: 组内最少节点数
        },
    ],
}
"""

from collections import defaultdict, Counter
from typing import Optional, Callable

from .common import scan_cslo_attrs, DensityEdgeBuilder, compute_homophily, make_stats, filter_small_classes


def _resolve_field(item: dict, field_path: str, field_slice: Optional[str] = None,
                   default: any = None):
    """解析字段路径: 'genres[0]' → item['genres'][0], 'artist' → item['artist']。

    支持语法:
      field          → item[field]
      field[0]       → item[field][0]  (取列表首元素)
      field.sub      → item[field]['sub']  (嵌套字典)

    空列表时返回 default (与原始代码的 `or ["Unknown"]` 模式一致)。
    """
    from agents.schema_contract import resolve, PathError
    try:
        value = resolve(item, field_path)
    except PathError:
        return default
    return default if value is None or value == [] else value


def _group_items(items: list, field_path: str, field_slice: Optional[str] = None):
    """按字段值分组, 返回 {value: [id, ...]}。

    field_slice 示例: ':3' 表示取列表前 3 个元素; '3' 表示取第 3 个元素后。
    对多值字段 (如 genres/actors), 每个值都将该 id 加入对应组。
    """
    groups = defaultdict(list)
    for it in items:
        id_val = it.get("_config_id")
        if id_val is None:
            continue
        vals = _resolve_field(it, field_path, default=[])
        vals = vals if isinstance(vals, list) else [vals]
        if field_slice:
            if field_slice.startswith(":"):
                vals = vals[:int(field_slice[1:])]
            else:
                i = int(field_slice); vals = vals[i:i+1]
        for v in vals:
            if v is not None and v != "":
                if isinstance(v, (dict, list)):
                    raise ValueError("Relation path must resolve to scalar/list-of-scalars")
                groups[v].append(id_val)
    return dict(groups)


def _label_of(item: dict, label_field: str, default: str = "Unknown"):
    """从 item 中提取标签值。label_field 如 'genres[0]' 或 'style'。
    空列表/缺失值返回 default (与原始代码的 `or [\"Unknown\"]` 模式一致)。
    """
    value = _resolve_field(item, label_field, default=default)
    if isinstance(value, (list, dict)):
        raise ValueError("Target must be scalar; list targets require an explicit indexed path")
    return value


def build_graph_from_config(items: list, cs_lo_dir: str, config: dict, opts: dict = None):
    """通用 co_attribute 图构建。

    Args:
        items:         list[dict], 每条是一个节点的元数据 (需含 id_field 和 label_field)
        cs_lo_dir:     cs_lo 目录路径 (含 {id}/{id}.md 和 {id}/images/)
        config:        域配置 dict (见模块 docstring)
        opts:          可选运行时覆盖 (caps, freq_filters, max_items 等)

    Returns:
        (nodes, edges, stats) — 与现有 build_X_graph 完全一致
    """
    opts = opts or {}
    id_field = config["id_field"]
    id_sanitizer = config.get("id_sanitizer", None)
    label_field = config["label_field"]
    min_class = config.get("min_class", 20)
    build_order = config.get("build_order", "filter-then-build")
    relations = config.get("relations", [])

    caps = opts.get("caps", {})
    ff = opts.get("freq_filters", {})

    # 1. 扫描节点属性 + 加 _config_id
    node_attrs = scan_cslo_attrs(cs_lo_dir)

    # 构建 id → item 映射
    if id_sanitizer:
        id_map = {}
        for it in items:
            sid = id_sanitizer(str(_resolve_field(it, id_field, default="")))
            id_map[sid] = it
    else:
        id_map = {str(_resolve_field(it, id_field, default="")): it for it in items if _resolve_field(it, id_field) not in (None, "")}

    # 交叉验证: 将 item 的配置 id 注入, 仅保留同时存在于 cs_lo 和 items 的节点
    for sid, it in id_map.items():
        it["_config_id"] = sid

    valid = set(node_attrs) & set(id_map)

    # ---- 标签映射 ----
    if build_order == "filter-then-build":
        # Steam 模式: 先过滤后建标签映射
        label_count = Counter(_label_of(id_map[a], label_field) for a in valid)
        small = {l for l, c in label_count.items() if c < min_class}
        keep = {a for a in valid if _label_of(id_map[a], label_field) not in small}
    elif build_order == "pre-filtered":
        keep = valid
        min_class = 0  # 不再过滤
    else:
        # build-then-filter (tmdb 模式): 先建全部标签映射
        keep = valid

    # 建标签映射
    if build_order == "build-then-filter":
        # TMDb 模式: label_map 覆盖所有出现过的标签 (包括后续可能被过滤的)
        all_labels = set()
        for it in items:
            lbl = _label_of(it, label_field)
            if lbl is not None and lbl != "":
                all_labels.add(lbl)
        label_map = {l: i for i, l in enumerate(sorted(all_labels))}
    else:
        kept_labels = sorted(set(_label_of(id_map[a], label_field) for a in keep))
        label_map = {l: i for i, l in enumerate(kept_labels)}

    # ---- 节点构建 ----
    nodes = []
    for nid in sorted(keep):
        it = id_map.get(nid, {})
        label = _label_of(it, label_field)
        label = "Unknown" if label is None or label == "" else label
        # 从 node_attrs 取文本/图像属性
        na = node_attrs.get(nid, {})
        node = {
            "id": nid,
            "type": config.get("node_type", "item"),
            "attributes": {
                **na,
                id_field: it.get(id_field, nid),
                "name": it.get("name", it.get("title", "")),
                "label": label,
                "label_index": label_map.get(label, -1),
            },
        }
        # 注入原始 item 中的额外属性 (如 title/name 等, 但不用额外字段名)
        if "name" not in node["attributes"] or not node["attributes"]["name"]:
            node["attributes"]["name"] = it.get("title", it.get("name", ""))
        nodes.append(node)

    keep_ids = {n["id"] for n in nodes}
    n_total = len(keep_ids)
    builder = DensityEdgeBuilder(valid_ids=keep_ids, n_total=n_total, rng_seed=42)

    # ---- 边构建 ----
    items_with_ids = [id_map[a] for a in keep_ids if a in id_map]

    for rel in relations:
        rel_name = rel["name"]
        group_by = rel["group_by"]
        rel_cap = caps.get(rel_name, rel.get("cap", 100))
        rel_ff = ff.get(rel_name, rel.get("freq_filter", None))
        rel_floor = rel.get("floor", 2)
        field_slice = rel.get("field_slice", None)

        groups = _group_items(items_with_ids, group_by, field_slice)
        builder.add_group(groups, rel_name, rel_cap,
                          freq_filter=rel_ff, floor=rel_floor)

    edges = builder.edges

    # ---- build-then-filter: 后过滤小类 ----
    if build_order == "build-then-filter" and min_class > 1:
        nodes, edges = filter_small_classes(nodes, edges, min_class)
        # 重建 label_map
        kept_labels = sorted(set(n["attributes"]["label"] for n in nodes))
        label_map = {l: i for i, l in enumerate(kept_labels)}
        for n in nodes:
            n["attributes"]["label_index"] = label_map.get(n["attributes"]["label"], -1)

    # ---- 统计 ----
    hom = compute_homophily(nodes, edges)
    stats = make_stats(nodes, edges, label_map, extra={
        "edge_relation_counts": dict(builder.relation_counts),
    })
    stats["homophily"] = round(hom, 3)
    return nodes, edges, stats
