"""
crawlers.arxiv — arxiv 统一 MAG 构建逻辑 (与 steam/tmdb/wikiart 同层级的 SoT)。

arxiv 与其它三域的区别
----------------------
steam/tmdb/wikiart 的图由 graph-builder agent 在「爬取到的原始数据」上构造 co_attribute 边。
arxiv 的图是**引用网络**: 节点 = 论文, 边 = 引用(reference_graph_builder agent 从 PDF/markdown
里抽 References 并在 arxiv 元数据里匹配得到)。原始引用边由 reference_graph_builder 产出,
经 benchmark/data/preprocessor.py 无损张量化为 benchmark/processed/graph_data.pt
(仅格式转换: edge_list -> tensor, 不构造任何新数据)。

本模块把 reference_graph_builder 的引用网络统一成 newdomain 协议的 MAG
(4 子域 cs_si/cs_lo/math_lo/physics_socph 合并为单一图, label = 子域, relation = "cites"),
与 scripts 侧的 port_arxiv_to_newdomain.py 同逻辑、同结果 (45,672 节点 / 57,045 无向引用边)。

诚实边界: graph_data.pt 的引用边由 agent(reference_graph_builder)构造; 本函数与 formatter agent
只做「统一 + 封装」, 不造数据。
"""
from pathlib import Path

import torch

from .common import compute_homophily, make_stats

# label_index -> 子域名 (与 preprocessor.py / port_arxiv_to_newdomain.py 一致)
DOMAIN_NAME = {0: "cs_si", 1: "cs_lo", 2: "math_lo", 3: "physics_socph"}

DEFAULT_GRAPH_DATA = Path(__file__).resolve().parents[1] / "benchmark" / "processed" / "graph_data.pt"


def build_arxiv_graph(graph_data_pt=None):
    """从 reference_graph_builder 的引用网络(graph_data.pt)构建统一 arxiv MAG。

    graph_data_pt: graph_data.pt 路径 (默认 benchmark/processed/graph_data.pt)。
    返回 (nodes, edges, stats), 与 steam/tmdb/wikiart 的 build_*_graph 同签名。

    节点 schema: {"id": str(paper_id), "attributes": {"label": 子域名, "label_index": int}}
    边 schema:   {"source": str, "target": str, "attributes": {"relation": "cites"}}
    边为无向去重 (每条无向边存一次), 与 newdomain v3 约定一致。
    """
    gpt = Path(graph_data_pt) if graph_data_pt else DEFAULT_GRAPH_DATA
    g = torch.load(gpt, map_location="cpu", weights_only=False)
    ei = g["edge_index"]
    y = g["labels"]
    pids = g["paper_ids"]

    nodes = []
    for i, pid in enumerate(pids):
        li = int(y[i].item())
        nodes.append({
            "id": str(pid),
            "attributes": {"label": DOMAIN_NAME.get(li, str(li)), "label_index": li},
        })

    seen = set()
    edges = []
    src = ei[0].tolist()
    dst = ei[1].tolist()
    for a, b in zip(src, dst):
        if a == b:
            continue
        k = (a, b) if a < b else (b, a)
        if k in seen:
            continue
        seen.add(k)
        edges.append({
            "source": str(pids[a]),
            "target": str(pids[b]),
            "attributes": {"relation": "cites"},
        })

    stats = make_stats(nodes, edges)
    return nodes, edges, stats
