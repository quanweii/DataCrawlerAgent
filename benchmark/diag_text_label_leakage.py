#!/usr/bin/env python3
"""诊断: text 特征里是否直接包含 label 字符串 (定义级泄露)。
对每个域: 读 graph(node id, label) → 读原始 text → 检查 label 是否出现在 text 里。
这是 C2 leak-free 协议的致命缺口检查: 之前只剪结构边泄露, 没查特征/属性泄露。
"""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRAPH = {
    "wikiart": "data/wikiart_datasets/wikiart_multimodal_graph_agent.json",
    "steam":   "data/steam_datasets/steam_multimodal_graph_agent.json",
    "tmdb":    "data/tmdb_datasets/tmdb_multimodal_graph_agent.json",
}
# text 源: node id → 该目录下 {id}.md (或子目录 {id}/{id}.md)
RAW = {
    "wikiart": "data/wikiart_artworks_v2",
    "steam":   "data/steam_games",
    "tmdb":    "data/tmdb_movies",
}

def load_text(dom, nid):
    base = Path(RAW[dom])
    # 尝试: {base}/{nid}/{nid}.md , {base}/{nid}.md
    for cand in [base/nid/f"{nid}.md", base/f"{nid}.md", base/nid]:
        if cand.is_file():
            try:
                return cand.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                return None
    return None

def norm(s):
    return re.sub(r"\s+", " ", (s or "").lower()).strip()

for dom, gp in GRAPH.items():
    g = json.load(open(ROOT/gp))
    nodes = g["nodes"]
    n = min(len(nodes), 4000)
    hit_field = 0      # text 含 "label: <labelvalue>" 这种结构化字段
    hit_substr = 0     # text 含 label 子串
    hit_artist = 0     # (wikiart) artist→label 近确定
    samples = []
    for nd in nodes[:n]:
        a = nd["attributes"]
        lbl = norm(a.get("label"))
        if not lbl: continue
        nid = nd["id"]
        txt = load_text(dom, nid)
        if txt is None: continue
        t = norm(txt)
        if lbl in t:
            hit_substr += 1
            # 结构化字段: "style: x" / "genre: x" / "label: x"
            if re.search(rf"(style|genre|label|category|tags?)\s*[:=]\s*[^a-z0-9]*{re.escape(lbl)}", t):
                hit_field += 1
                if len(samples) < 2:
                    m = re.search(rf".{{0,30}}(?:style|genre|label|category|tags?)\s*[:=]\s*[^a-z0-9]*{re.escape(lbl)}[^a-z].{{0,20}}", t)
                    samples.append((nid, lbl, m.group(0) if m else "?"))
    print(f"\n=== {dom} (扫描 {n} 节点) ===")
    print(f"  label 子串出现在 text : {hit_substr}/{n} = {hit_substr/n:.1%}")
    print(f"  结构化字段(style/genre: X): {hit_field}/{n} = {hit_field/n:.1%}  {'<<< 定义级泄露' if hit_field/n>0.3 else ''}")
    for s in samples:
        print(f"    样例: label={s[1]!r}  text片段={s[2]!r}")
