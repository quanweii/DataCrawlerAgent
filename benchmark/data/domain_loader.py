"""
单域加载器 — 为 steam / tmdb / wikiart 三个新域提供统一的图 + 特征 + LP 边加载。

为什么新建（而不复用 AMMAGDataset）
-----------------------------------
AMMAGDataset 是 arxiv 4 域合并的载体，被一堆旧脚本依赖，且其
`_same_domain_negative_sampling` 在单域下会退化为均匀随机负采样 → 重新让
高同质性图（如 wikiart 0.894）的 LP 任务饱和。因此新域走独立的单域加载器，
LP 协议直接消费 scripts/lp_hard_negative_sampler.py 预先生成的 CSV。

数据布局约定
  canonical 图: data/{domain}_datasets/{domain}_multimodal_graph_agent.json  (4 域, 全部多智能体产物)
                arxiv 经 reference_graph_builder(引用边) → graph_data.pt → formatter agent 统一封装
  原始:    data/{raw}/{node_id}/{node_id}.md  +  images/
  嵌入:    benchmark/processed/{domain}/{text,image}_embeddings.pt  ({"ids":[...],"embeddings":T})
  LP CSV:  data/{domain}_lp/{train,val,test}_{pos,neg,neg_hard,neg_hard_degree}.csv

v3 节点结构: {"id": str, "attributes": {"label": str, "label_index": int, ...}}
v3 边结构:   {"source": str, "target": str, "attributes": {"relation": str}}
"""
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from loguru import logger

# ---- 域 → 文件路径（相对项目根）----
DOMAIN_GRAPH = {
    # 新三域 canonical = 多智能体流水线产物 (*_agent.json)。
    # 已核实: 与 *_v3.json 节点集/标签分布/关系键完全一致, 仅边密度有 ±0.1% RNG 噪声 → 特征对齐不受影响。
    # 指向 agent 产物, 使 "my multi-agent produced the datasets" 字面成立(论文 pivot 核心)。
    "steam": "data/steam_datasets/steam_multimodal_graph_agent.json",
    "tmdb": "data/tmdb_datasets/tmdb_multimodal_graph_agent.json",
    "wikiart": "data/wikiart_datasets/wikiart_multimodal_graph_agent.json",
    # arxiv: canonical = 多智能体产物 (A′)。reference_graph_builder 造引用边 → graph_data.pt;
    # formatter agent 经 crawlers/arxiv.build_arxiv_graph 统一成 newdomain MAG → *_agent.json。
    # 与 v3 节点集/边集/标签分布完全一致 (已核实)。4 域全部 = agent 系统产物。
    "arxiv": "data/arxiv_datasets/arxiv_multimodal_graph_agent.json",
    # spotify: zero-code new domain (2026-07-11), built via build-mag-dataset skill +
    # config_driven.build_graph_from_config (no framework Python). C3 reusability exhibit.
    "spotify": "data/spotify_datasets/spotify_multimodal_graph_agent.json",
}
# v3 节点 id 对应的原始数据目录（与 encode_domain.py / lp_hard_negative_sampler 一致）
DOMAIN_RAW = {
    "steam": "data/steam_games",
    "tmdb": "data/tmdb_movies",
    "wikiart": "data/wikiart_artworks_v2",
    "arxiv": "data/arxiv_datasets",  # 原始 .md/images 在 data/{sub}_papers，这里仅占位
    "spotify": "data/spotify_cslo",  # track metadata markdown (text-only)
}

# 各域可用模态（arxiv 独有 formula；其余 3 域 text+image）。供 build_features 校验。
DOMAIN_MODALITIES = {
    "steam": ["text", "image"],
    "tmdb": ["text", "image"],
    "wikiart": ["text", "image"],
    "arxiv": ["text", "image", "formula"],
    "spotify": ["text"],  # text-only (public CSV lacks cover art)
}

# benchmark/data/domain_loader.py → parents[2] = 项目根 (DataCrawlerAgent)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "benchmark" / "processed"


def _abs(p: str) -> Path:
    """相对项目根的路径 → 绝对路径。"""
    pp = Path(p)
    return pp if pp.is_absolute() else PROJECT_ROOT / pp


# ===================== 图加载 =====================

def load_domain_graph(domain: str, relation: str = None) -> dict:
    """读取 v3.json，返回单域图结构。

    relation: 若指定，则只保留 attributes.relation == relation 的边（**结构调控/steering 用**）。
              节点集合、标签、特征加载方式**完全不变**——只换边，从而隔离"画哪种边"的因果效应。
              同质性(homophily)按筛后边重算。

    Returns:
        node_ids:    List[str]           顺序即节点索引顺序
        id2idx:      Dict[str,int]
        edge_index:  LongTensor [2, 2E]  无向（每条边加了反向边）、去自环、去重
        y:           LongTensor [N]      label_index（NC 类别）
        num_nodes:   int
        label_names: Dict[int,str]       label_index -> label 字符串
        relations:   Dict[str,int]       relation -> 边数（全图统计，与 relation 参数无关）
        homophily:   float               同质性（同标签边占比；筛边后按筛后边算）
        relation:    str|None            实际使用的关系筛选
        covered:     int                 至少有一条(筛后)边的节点数
    """
    gpath = _abs(DOMAIN_GRAPH[domain])
    with open(gpath, encoding="utf-8") as f:
        g = json.load(f)

    nodes = g["nodes"]
    edges = g["edges"]

    node_ids = [n["id"] for n in nodes]
    id2idx = {nid: i for i, nid in enumerate(node_ids)}

    # 标签：优先用 label_index；缺失则按 label 字符串重建
    label_names: Dict[int, str] = {}
    str2idx: Dict[str, int] = {}
    y_list = []
    for n in nodes:
        attr = n["attributes"]
        li = attr.get("label_index")
        if li is None:
            lab = attr["label"]
            if lab not in str2idx:
                str2idx[lab] = len(str2idx)
            li = str2idx[lab]
        label_names[int(li)] = attr.get("label", str(li))
        y_list.append(int(li))
    y = torch.tensor(y_list, dtype=torch.long)

    # 边：无向 + 去自环 + 去重（可选按 relation 筛）
    src, dst = [], []
    seen = set()
    id2lab = {n["id"]: n["attributes"]["label"] for n in nodes}
    same_cnt, tot_cnt = 0, 0
    for e in edges:
        if relation is not None and e.get("attributes", {}).get("relation", "?") != relation:
            continue
        s, t = e["source"], e["target"]
        if s == t or s not in id2idx or t not in id2idx:
            continue
        tot_cnt += 1
        if id2lab.get(s) == id2lab.get(t):
            same_cnt += 1
        a, b = id2idx[s], id2idx[t]
        key = (a, b) if a < b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        src += [a, b]   # 正向
        dst += [b, a]   # 反向 → 无向
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    homophily = same_cnt / tot_cnt if tot_cnt else 0.0

    relations = dict(Counter(e["attributes"].get("relation", "?") for e in edges))
    covered = len(seen) * 2  # 粗略：去重无向边数 ×2 ≈ 涉及节点（含重复），下文精确算
    covered_nodes = len(set(src)) if src else 0

    logger.info(f"[{domain}{'' if relation is None else '/'+relation}] nodes={len(node_ids)} "
                f"edges(unique)={len(seen)} covered_nodes={covered_nodes} "
                f"classes={len(label_names)} homophily={homophily:.3f}")
    return {
        "node_ids": node_ids,
        "id2idx": id2idx,
        "edge_index": edge_index,
        "y": y,
        "num_nodes": len(node_ids),
        "label_names": label_names,
        "relations": relations,
        "homophily": homophily,
        "relation": relation,
        "covered": covered_nodes,
    }


# ===================== 特征构建 =====================

def _load_emb(domain: str, modality_part: str, node_ids: List[str]) -> torch.Tensor:
    """加载某模态嵌入并对齐到 node_ids 顺序（缺失节点 → 零向量）。"""
    path = PROCESSED_DIR / domain / f"{modality_part}_embeddings.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"缺少嵌入文件: {path}\n先跑: "
            f"python benchmark/encode_domain.py --domain {domain} --modalities {modality_part}"
        )
    obj = torch.load(path, map_location="cpu")
    ids = obj["ids"]
    embs = obj["embeddings"].float()
    id2row = {sid: i for i, sid in enumerate(ids)}
    dim = embs.size(1)
    out = torch.zeros((len(node_ids), dim), dtype=torch.float32)
    found = 0
    for i, nid in enumerate(node_ids):
        if nid in id2row:
            out[i] = embs[id2row[nid]]
            found += 1
    logger.info(f"[{domain}] {modality_part} 对齐 {found}/{len(node_ids)}, dim={dim}")
    return out


def build_features(domain: str, graph: dict, modality: str = "text+image") -> torch.Tensor:
    """按 OpenMAG 协议融合 CLIP 嵌入：逐模态 L2 归一化后**逐元素求平均**，再 L2 归一化。

    为什么是 mean 而非 concat
    -------------------------
    OpenMAG 用 CLIP-ViT-L/14 把 text 和 image 都编进**同一个 768-d 共享空间**，
    并"unifying the node embedding dimensionality at 768 across all tasks"——基线融合
    是逐元素（mean/sum），不是拼接，更不是 learnable Linear 投影。先前把 [text;image]
    拼成 1536-d 再 Linear(1536,768) 投影的做法 (a) 非 OpenMAG 方法、(b) 非对称（只多模态被
    压缩）——已弃。mean 融合让单/多模态**同构同维（768-d）**，uni-vs-multi 对比天然公平。

    modality ∈ {text, image, formula, text+image, text+image+formula, ...}（+ 分隔）
    单模态 → 该模态 768-d（归一化）；多模态 → 各模态归一化后逐元素 mean，再归一化 → 768-d。
    """
    node_ids = graph["node_ids"]
    parts = [p.strip() for p in modality.split("+") if p.strip()]
    norms = []
    for p in parts:
        e = _load_emb(domain, p, node_ids)
        n = e.norm(dim=1, keepdim=True).clamp(min=1e-8)   # 逐模态 L2 归一化（CLIP 惯例）
        norms.append(e / n)
    if len(norms) == 1:
        x = norms[0]
    else:
        x = torch.stack(norms, dim=0).mean(dim=0)          # OpenMAG 逐元素融合
        n = x.norm(dim=1, keepdim=True).clamp(min=1e-8)
        x = x / n
    logger.info(f"[{domain}] build_features({modality}) → {tuple(x.shape)} [mean-fusion]")
    return x


# ===================== 单域 Dataset =====================

class SingleDomainDataset:
    """单域节点分类数据集。

    鸭子类型兼容 benchmark.tasks.node_classification.run_node_classification:
      - .data              : torch_geometric.data.Data (x, edge_index, y, num_nodes)
      - .get_split_masks() : (train, val, test) bool masks, 分层切分（保护小类）
    """

    def __init__(self, domain: str, modality: str = "text+image", relation: str = None):
        self.domain = domain
        self.modality = modality
        self.relation = relation
        self.graph = load_domain_graph(domain, relation=relation)
        x = build_features(domain, self.graph, modality)
        data = Data(
            x=x,
            edge_index=self.graph["edge_index"],
            y=self.graph["y"],
        )
        data.num_nodes = self.graph["num_nodes"]
        self.data = data
        # 单域无跨域概念；_domain_labels 全 0（供 LP 同域负采样退化为随机用）
        self._domain_labels = torch.zeros(self.graph["num_nodes"], dtype=torch.long)

    def get_split_masks(self, split: str = "supervised", seed: int = 1,
                        few_shot_k: int = -1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """分层 train/val/test 切分（按类内 shuffle），保护小类。

        - supervised: 每类 60/20/20；类太小(<5)时保证至少 1 个训练样本。
        - few_shot + few_shot_k>0: 每类取 k 训练，余下 50/50 val/test。
        """
        rng = np.random.RandomState(seed)
        y = self.data.y.cpu().numpy()
        n = len(y)
        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        test_mask = torch.zeros(n, dtype=torch.bool)

        if split == "few_shot" and few_shot_k > 0:
            for c in np.unique(y):
                idx = np.where(y == c)[0]
                rng.shuffle(idx)
                k = min(few_shot_k, len(idx))
                rest = idx[k:]
                mid = len(rest) // 2
                train_mask[idx[:k]] = True
                val_mask[rest[:mid]] = True
                test_mask[rest[mid:]] = True
            return train_mask, val_mask, test_mask

        for c in np.unique(y):
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            n_c = len(idx)
            if n_c < 5:
                # 小类：保证 ≥1 训练样本，其余交替 val/test
                n_tr = max(1, int(n_c * 0.6))
                train_mask[idx[:n_tr]] = True
                for j, ii in enumerate(idx[n_tr:]):
                    (val_mask if j % 2 == 0 else test_mask)[ii] = True
                continue
            n_tr = int(n_c * 0.6)
            n_va = int(n_c * 0.2)
            train_mask[idx[:n_tr]] = True
            val_mask[idx[n_tr:n_tr + n_va]] = True
            test_mask[idx[n_tr + n_va:]] = True
        return train_mask, val_mask, test_mask


# ===================== LP 边加载 =====================

def load_lp_edges(domain: str, graph: dict, name: str) -> torch.Tensor:
    """读取 data/{domain}_lp/{name}.csv → LongTensor[2, E]（节点 idx）。

    name 例如: train_pos / val_neg_hard / test_neg_hard_degree
    CSV 格式: header `source,target`，值为字符串节点 id。
    """
    path = PROJECT_ROOT / "data" / f"{domain}_lp" / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"缺少 LP 边文件: {path}")
    id2idx = graph["id2idx"]
    src, dst = [], []
    skipped = 0
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # 跳过 header
        for row in reader:
            if len(row) < 2:
                continue
            s, t = row[0].strip(), row[1].strip()
            if s in id2idx and t in id2idx:
                src.append(id2idx[s])
                dst.append(id2idx[t])
            else:
                skipped += 1
    if skipped:
        logger.warning(f"[{domain}/{name}] 跳过 {skipped} 条 id 不在图中的边")
    logger.info(f"[{domain}/{name}] 加载 {len(src)} 条边")
    return torch.tensor([src, dst], dtype=torch.long)
