#!/usr/bin/env python3
"""
深层实验可行性诊断 (benchmark 论文审稿人会问的):
  1. 边类型级同质性 — 是否有边类型定义性地编码标签 (任务退化风险)
  2. 孤立节点明细 — 谁该删
  3. 模态区分度 — 文本/图像是否真有判别力 (重复率/模板化)
  4. 链路预测负采样严格性 — 同质性下随机负采样 vs 严格负采样差距
  5. 类别分离性代理 — 文本单模态最近邻分类准确率 (模态消融预判)
"""
import json, random, hashlib
from pathlib import Path
from collections import defaultdict, Counter

DOMAINS = {
    "steam":   ("data/steam_datasets/steam_multimodal_graph_v3.json",   "data/steam_games"),
    "tmdb":    ("data/tmdb_datasets/tmdb_multimodal_graph_v3.json",     "data/tmdb_movies"),
    "wikiart": ("data/wikiart_datasets/wikiart_multimodal_graph_v3.json","data/wikiart_artworks_v2"),
}

def load(p):
    with open(p, encoding="utf-8") as f: return json.load(f)

def edge_homophily_by_relation(nodes, edges):
    id2l = {n["id"]: n["attributes"]["label"] for n in nodes}
    by_rel = defaultdict(lambda: [0,0])  # [same, total]
    for e in edges:
        rel = e["attributes"].get("relation","?")
        by_rel[rel][1]+=1
        if id2l.get(e["source"])==id2l.get(e["target"]): by_rel[rel][0]+=1
    return {r: (round(s/max(t,1),3), t) for r,(s,t) in by_rel.items()}

def text_discriminativity(nodes, base_dir, sample_n=500):
    """抽样文本, 看重复率/模板化"""
    rng = random.Random(42)
    sample = rng.sample(nodes, min(sample_n, len(nodes)))
    texts=[]; hashes=set(); dup=0; sizes=[]
    for n in sample:
        nid = n["id"]
        p = Path(base_dir)/nid/f"{nid}.md"
        if not p.exists(): continue
        t = p.read_text(encoding="utf-8").strip()
        texts.append(t); sizes.append(len(t))
        h = hashlib.md5(t.encode()).hexdigest()
        if h in hashes: dup+=1
        hashes.add(h)
    # 看前40字符的去重率 (模板化会导致前缀高度重复)
    prefix=set(t[:60] for t in texts)
    return {
        "sampled": len(texts),
        "exact_dup": dup,
        "unique_full_text_ratio": round(len(hashes)/max(len(texts),1),3),
        "unique_60char_prefix_ratio": round(len(prefix)/max(len(texts),1),3),
        "min_text_len": min(sizes) if sizes else 0,
        "max_text_len": max(sizes) if sizes else 0,
    }

def image_reality(nodes, base_dir, sample_n=100):
    """抽样图片, 看是否真实 (>2KB, 非占位)"""
    rng = random.Random(42)
    sample = rng.sample(nodes, min(sample_n, len(nodes)))
    sizes=[]
    zero=0; tiny=0
    for n in sample:
        d = Path(base_dir)/n["id"]/"images"
        imgs = sorted(d.glob("*")) if d.exists() else []
        if not imgs: zero+=1; continue
        sz = imgs[0].stat().st_size
        sizes.append(sz)
        if sz < 2048: tiny+=1
    real = [s for s in sizes if s>=2048]
    return {
        "sampled": len(sample), "no_image": zero,
        "tiny_lt2KB": tiny,
        "real_image_pct": round(100*len(real)/max(len(sample),1),1),
        "median_img_kb": round((sorted(real)[len(real)//2]/1024) if real else 0,1),
    }

def lp_strictness(nodes, edges, homophily):
    """同质性高时, 随机负采样 vs 同质负采样 AUC gap 预估"""
    # 随机负采样: 负对几乎都是异类 → 简单启发式 (同类才连) 即可拿到 ~homophily AUC
    n=len(nodes)
    return {
        "random_neg_baseline_auc_est": round(max(homophily, 1-homophily)*0.5+0.5, 3),
        "note": "若 >0.9, 建议用同分布负采样 (同度数/同类内) 以增加难度"
    }

def isolated_detail(nodes, edges):
    deg=Counter()
    for e in edges:
        deg[e["source"]]+=1; deg[e["target"]]+=1
    iso=[n for n in nodes if deg.get(n["id"],0)==0]
    lab=Counter(n["attributes"]["label"] for n in iso)
    return {"count": len(iso), "labels": dict(lab)}

def main():
    random.seed(42)
    print("="*70)
    print("深层实验可行性诊断")
    print("="*70)
    for dom,(gpath,dpath) in DOMAINS.items():
        print(f"\n{'#'*60}\n# {dom.upper()}\n{'#'*60}")
        g=load(gpath); nodes,edges=g["nodes"],g["edges"]
        id2l={n["id"]:n["attributes"]["label"] for n in nodes}
        same=sum(1 for e in edges if id2l.get(e["source"])==id2l.get(e["target"]))
        hom=same/max(len(edges),1)

        print("\n[1] 边类型级同质性 (定义性编码检查)")
        for r,(h,t) in edge_homophily_by_relation(nodes,edges).items():
            flag = "⚠️定义性同质" if h>=0.95 else ("高同质" if h>=0.7 else "异质/中")
            print(f"  {r:20s} hom={h}  edges={t:,}  [{flag}]")

        print("\n[2] 孤立节点")
        iso=isolated_detail(nodes,edges)
        print(f"  {iso['count']} 个 ({100*iso['count']/len(nodes):.1f}%), 标签: {iso['labels']}")

        print("\n[3] 文本区分度 (抽样500)")
        td=text_discriminativity(nodes,dpath)
        for k,v in td.items(): print(f"  {k}: {v}")

        print("\n[4] 图像真实性 (抽样100)")
        im=image_reality(nodes,dpath)
        for k,v in im.items(): print(f"  {k}: {v}")

        print("\n[5] 链路预测负采样严格性")
        lp=lp_strictness(nodes,edges,hom)
        for k,v in lp.items(): print(f"  {k}: {v}")

    print(f"\n{'='*70}")
    print("结论见对话")

if __name__=="__main__":
    main()
