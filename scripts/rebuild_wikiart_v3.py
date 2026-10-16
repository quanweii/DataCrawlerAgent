#!/usr/bin/env python3
"""WikiArt v3 — 用 enriched 文本(306字符)重建图, 64类, 过滤小类"""
import json, csv, re, random
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations
from datetime import datetime
from loguru import logger
import numpy as np

BASE_DIR = Path("./data")
RAW_DIR = BASE_DIR / "wikiart_raw"
ARTWORKS_DIR = BASE_DIR / "wikiart_artworks_v2"
DATASETS_DIR = BASE_DIR / "wikiart_datasets"
GRAPHS_DIR = BASE_DIR / "wikiart_graphs"
ENRICHED_JSON = RAW_DIR / "wikiart_v2_enriched.json"
GRAPH_JSON = DATASETS_DIR / "wikiart_multimodal_graph_v3.json"
EDGES_CSV = GRAPHS_DIR / "co_attribute_edges_v3.csv"

MAX_ARTIST = 100
MAX_STYLE = 50


def main():
    logger.info("="*60)
    logger.info("WikiArt v3 图重建 (enriched文本 + 64类)")
    logger.info("="*60)
    with open(ENRICHED_JSON, encoding="utf-8") as f:
        paintings = json.load(f)
    logger.info(f"画作: {len(paintings)}, 类: {len(set(p['style'] for p in paintings))}")

    # 扫描 cs_lo 目录获取节点属性
    node_attrs = {}
    for p in paintings:
        pid = re.sub(r'[^\w\-.]', '_', p["painting_id"])
        art_dir = ARTWORKS_DIR / pid
        md_path = art_dir / f"{pid}.md"
        if not md_path.exists():
            continue
        text = md_path.read_text(encoding="utf-8")
        images_dir = art_dir / "images"
        img_count = len(list(images_dir.glob("*"))) if images_dir.exists() else 0
        node_attrs[pid] = {
            "painting_id": pid,
            "markdown_length": len(text),
            "image_count": img_count,
            "markdown_size": md_path.stat().st_size,
        }
    logger.info(f"有效节点: {len(node_attrs)}")

    valid_ids = set(node_attrs.keys())
    pid_to_painting = {}
    for p in paintings:
        pid = re.sub(r'[^\w\-.]', '_', p["painting_id"])
        if pid in valid_ids:
            pid_to_painting[pid] = p

    # 标签映射
    styles = sorted(set(p["style"] for p in pid_to_painting.values() if p.get("style")))
    label_map = {s: i for i, s in enumerate(styles)}
    logger.info(f"标签: {len(label_map)} 类")

    nodes = []
    for pid, na in node_attrs.items():
        p = pid_to_painting.get(pid, {})
        style = p.get("style", "Unknown")
        nodes.append({
            "id": pid, "type": "artwork",
            "attributes": {**na, "title": p.get("title",""),
                          "label": style, "label_index": label_map.get(style,-1)}
        })

    edges = []
    edge_set = set()
    edge_counts = defaultdict(int)
    def add_edge(a, b, rel):
        key = tuple(sorted([a, b]))
        if key not in edge_set and key[0]!=key[1]:
            edge_set.add(key)
            edges.append({"source":key[0],"target":key[1],"type":"co_attribute",
                         "attributes":{"relation":rel}})
            edge_counts[rel]+=1
    def build_edges(m, rel, cap):
        for v, ids in m.items():
            ids=[i for i in ids if i in valid_ids]
            if len(ids)<2: continue
            if len(ids)<=cap:
                for a,b in combinations(ids,2): add_edge(a,b,rel)
            else:
                s=random.Random(42).sample(ids,cap)
                for a,b in combinations(s,2): add_edge(a,b,rel)
                for rid in set(ids)-set(s):
                    add_edge(rid,random.Random(hash(rid)%2**32).choice(s),rel)

    logger.info("same_artist 边...")
    a2i=defaultdict(list)
    for pid in valid_ids:
        p=pid_to_painting.get(pid,{})
        if p.get("artist"): a2i[p["artist"]].append(pid)
    build_edges(a2i,"same_artist",MAX_ARTIST)
    logger.info(f"  {edge_counts['same_artist']:,}")

    logger.info("shared_style 边...")
    s2i=defaultdict(list)
    for pid in valid_ids:
        p=pid_to_painting.get(pid,{})
        if p.get("style"): s2i[p["style"]].append(pid)
    build_edges(s2i,"shared_style",MAX_STYLE)
    logger.info(f"  {edge_counts['shared_style']:,}")

    logger.info(f"节点:{len(nodes)} 边:{len(edges):,} 平均度:{2*len(edges)/max(len(nodes),1):.1f}")
    id2l={n["id"]:n["attributes"]["label"] for n in nodes}
    same=sum(1 for e in edges if id2l.get(e["source"])==id2l.get(e["target"]))
    hom=same/max(len(edges),1)
    logger.info(f"同质性:{hom:.3f}")

    # 保存
    total_img=sum(n["attributes"]["image_count"] for n in nodes)
    total_md=sum(n["attributes"]["markdown_length"] for n in nodes)
    ld=Counter(n["attributes"]["label"] for n in nodes)
    top=sorted(ld.items(),key=lambda x:-x[1])
    dataset={"nodes":nodes,"edges":edges,"metadata":{
        "name":"wikiart_multimodal_graph_v3",
        "created_at":datetime.now().strftime("%Y-%m-%d"),"domain":"wikiart",
        "statistics":{"node_count":len(nodes),"edge_count":len(edges),
            "total_images":total_img,"artworks_with_images":sum(1 for n in nodes if n["attributes"]["image_count"]>0),
            "total_markdown_chars":total_md,"avg_md_length":round(total_md/max(len(nodes),1),0),
            "avg_degree":round(2*len(edges)/max(len(nodes),1),1),"homophily":round(hom,3),
            "label_distribution":dict(ld),"num_labels":len(label_map),
            "top_labels":[{"label":l,"count":c} for l,c in top[:15]]},
        "schema":{"node_types":["artwork"],"edge_types":["co_attribute"],
            "node_attributes":["painting_id","title","markdown_length","image_count","markdown_size","label","label_index"],
            "edge_relations":["same_artist","shared_style"],"modalities":["text","image"]}}}
    with open(GRAPH_JSON,"w",encoding="utf-8") as f:
        json.dump(dataset,f,ensure_ascii=False,indent=2)
    logger.info(f"图JSON:{GRAPH_JSON}")
    with open(EDGES_CSV,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["source_id","target_id","edge_type","relation"])
        for e in edges: w.writerow([e["source"],e["target"],e["type"],e["attributes"].get("relation","")])
    logger.info(f"边CSV:{EDGES_CSV}")
    logger.info("="*60)
    logger.info(f"✅ WikiArt v3 完成: {len(nodes)}节点/{len(edges):,}边/avg度{2*len(edges)/max(len(nodes),1):.1f}/同质{hom:.3f}/{len(label_map)}类/avg文本{total_md//max(len(nodes),1)}字符")
    logger.info("="*60)


if __name__=="__main__":
    main()
