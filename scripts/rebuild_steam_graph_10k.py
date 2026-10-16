#!/usr/bin/env python3
"""Steam 10K 图重建 — 2791节点, v3优化策略+小类过滤"""
import json, csv, re, random
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations
from datetime import datetime
from loguru import logger

BASE=Path("./data"); RAW=BASE/"steam_raw"/"steam_games_10k.json"
GAMES_DIR=BASE/"steam_games"; DATASETS=BASE/"steam_datasets"; GRAPHS=BASE/"steam_graphs"
GRAPH=DATASETS/"steam_multimodal_graph_v3.json"
EDGES=GRAPHS/"co_attribute_edges_v3.csv"
MIN_CLASS=20

def main():
    logger.info("Steam 图重建 (2791节点, v3优化)")
    games=json.load(open(RAW,encoding="utf-8"))
    logger.info(f"游戏: {len(games)}")

    # 扫描目录
    na={}
    for gd in GAMES_DIR.iterdir():
        if not gd.is_dir(): continue
        appid=gd.name; md=gd/f"{appid}.md"
        if not md.exists(): continue
        t=md.read_text(encoding="utf-8")
        im=gd/"images"; ic=len(list(im.glob("*"))) if im.exists() else 0
        na[appid]={"appid":appid,"markdown_length":len(t),"image_count":ic,"markdown_size":md.stat().st_size}
    logger.info(f"有效节点: {len(na)}")

    valid=set(na); id2g={str(g["appid"]):g for g in games}
    # 过滤小类(用第一个genre做label)
    lc=Counter()
    for appid in valid:
        g=id2g.get(appid,{}); gs=g.get("genres",[]); lc[gs[0] if gs else "Unknown"]+=1
    small={l for l,c in lc.items() if c<MIN_CLASS}
    logger.info(f"过滤小类<{MIN_CLASS}: {sorted(small)}")

    keep={appid for appid in valid if (id2g.get(appid,{}).get("genres") or ["Unknown"])[0] not in small}
    logger.info(f"过滤后: {len(keep)} 节点")

    allg=sorted({(id2g.get(a,{}).get("genres") or ["Unknown"])[0] for a in keep})
    lm={l:i for i,l in enumerate(allg)}
    nodes=[{"id":a,"type":"game","attributes":{**na[a],"name":id2g.get(a,{}).get("name",""),
        "label":(id2g.get(a,{}).get("genres") or ["Unknown"])[0],"label_index":lm.get((id2g.get(a,{}).get("genres") or ["Unknown"])[0],-1)}}
        for a in keep]

    edges=[]; es=set(); ec=defaultdict(int)
    def add(a,b,r):
        k=tuple(sorted([a,b]))
        if k not in es and k[0]!=k[1]:
            es.add(k); edges.append({"source":k[0],"target":k[1],"type":"co_attribute","attributes":{"relation":r}}); ec[r]+=1
    def build(m,r,cap,ff=None):
        n=len(keep)
        fil=0
        for v,ids in m.items():
            ids=[i for i in ids if i in keep]
            if ff is not None and len(ids)>ff*n: fil+=1; continue
            if len(ids)<2: continue
            if len(ids)<=cap:
                for a,b in combinations(ids,2): add(a,b,r)
            else:
                s=random.Random(42).sample(ids,cap)
                for a,b in combinations(s,2): add(a,b,r)
                for rid in set(ids)-set(s): add(rid,random.Random(hash(rid)%2**32).choice(s),r)
        return fil

    g2i=defaultdict(list)
    for a in keep:
        for g in id2g.get(a,{}).get("genres",[]): g2i[g].append(a)
    f=build(g2i,"shared_genre",30,0.30); logger.info(f"shared_genre(过滤{f}高频):{ec['shared_genre']:,}")

    d2i=defaultdict(list)
    for a in keep:
        for d in id2g.get(a,{}).get("developers",[]):
            if d: d2i[d].append(a)
    build(d2i,"same_developer",100); logger.info(f"same_developer:{ec['same_developer']:,}")

    p2i=defaultdict(list)
    for a in keep:
        for p in id2g.get(a,{}).get("publishers",[]):
            if p: p2i[p].append(a)
    build(p2i,"same_publisher",100); logger.info(f"same_publisher:{ec['same_publisher']:,}")

    c2i=defaultdict(list)
    for a in keep:
        for c in id2g.get(a,{}).get("categories",[]): c2i[c].append(a)
    n=len(keep); cf={k:v for k,v in c2i.items() if len(v)<=0.40*n and len(v)>=5}
    logger.info(f"category过滤:{len(c2i)}→{len(cf)}")
    build(cf,"shared_category",30); logger.info(f"shared_category:{ec['shared_category']:,}")

    id2l={n_["id"]:n_["attributes"]["label"] for n_ in nodes}
    same=sum(1 for e in edges if id2l.get(e["source"])==id2l.get(e["target"]))
    hom=same/max(len(edges),1)
    logger.info(f"节点:{len(nodes)} 边:{len(edges):,} 度:{2*len(edges)/max(len(nodes),1):.1f} 同质:{hom:.3f}")

    ld=Counter(n_["attributes"]["label"] for n_ in nodes)
    ti=sum(n_["attributes"]["image_count"] for n_ in nodes)
    tm=sum(n_["attributes"]["markdown_length"] for n_ in nodes)
    data={"nodes":nodes,"edges":edges,"metadata":{"name":"steam_multimodal_graph_v3",
        "created_at":datetime.now().strftime("%Y-%m-%d"),"domain":"steam",
        "statistics":{"node_count":len(nodes),"edge_count":len(edges),"total_images":ti,
            "nodes_with_images":sum(1 for n_ in nodes if n_["attributes"]["image_count"]>0),
            "total_markdown_chars":tm,"avg_md_length":round(tm/max(len(nodes),1),0),
            "avg_degree":round(2*len(edges)/max(len(nodes),1),1),"homophily":round(hom,3),
            "label_distribution":dict(ld),"num_labels":len(lm),
            "top_labels":[{"label":l,"count":c} for l,c in sorted(ld.items(),key=lambda x:-x[1])[:15]]},
        "schema":{"node_types":["game"],"edge_types":["co_attribute"],
            "node_attributes":["appid","name","markdown_length","image_count","markdown_size","label","label_index"],
            "edge_relations":["shared_genre","same_developer","same_publisher","shared_category"],
            "modalities":["text","image"]}}}
    json.dump(data,open(GRAPH,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    with open(EDGES,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["source_id","target_id","edge_type","relation"])
        for e in edges: w.writerow([e["source"],e["target"],e["type"],e["attributes"].get("relation","")])
    logger.info(f"✅ 完成: {GRAPH}")
    logger.info(f"节点{len(nodes)} 边{len(edges):,} 类{len(lm)} 度{2*len(edges)/max(len(nodes),1):.1f} 同质{hom:.3f}")

if __name__=="__main__":
    main()
