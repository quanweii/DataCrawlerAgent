#!/usr/bin/env python3
"""铁证: 不用 CLIP、不用 GNN, 仅用正则从 raw text 解析 'Style:/Genres:' 字段,
看能否直接恢复 label。若恢复率 ≈ text-only NC 准确率, 则证明 NC 模型只是在
'读 text 里的答案', text 特征是 label 的噪声编码 (定义级泄露)。"""
import json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
GRAPH = {
    "wikiart": "data/wikiart_datasets/wikiart_multimodal_graph_agent.json",
    "steam":   "data/steam_datasets/steam_multimodal_graph_agent.json",
    "tmdb":    "data/tmdb_datasets/tmdb_multimodal_graph_agent.json",
}
RAW = {"wikiart":"data/wikiart_artworks_v2","steam":"data/steam_games","tmdb":"data/tmdb_movies"}

def load_text(dom, nid):
    b=Path(RAW[dom])
    for c in [b/nid/f"{nid}.md", b/f"{nid}.md"]:
        if c.is_file(): return c.read_text(encoding="utf-8",errors="ignore")
    return None

def norm(s): return re.sub(r"\s+"," ",(s or "").lower()).strip()

for dom,gp in GRAPH.items():
    g=json.load(open(ROOT/gp)); nodes=g["nodes"]
    N=min(len(nodes),6000)
    correct_field=0; label_in_set=0; has_field=0; evaluated=0
    # label 集合 (归一化)
    all_labels=set(norm(n["attributes"]["label"]) for n in nodes if n["attributes"].get("label"))
    for nd in nodes[:N]:
        lbl=norm(nd["attributes"].get("label"))
        if not lbl: continue
        txt=load_text(dom,nd["id"])
        if txt is None: continue
        evaluated+=1; t=norm(txt)
        # 解析 Style/Genres 字段
        m=re.search(r"(?:style|genres?)\s*[:=]\s*([^\n]+)", t)
        if m:
            has_field+=1
            field_vals=set()
            for tok in re.split(r"[,;|/]", m.group(1)):
                tok=tok.strip().strip(".")
                # 去掉尾部多余词 (如 'racing simulation sports' 已被逗号切)
                if tok: field_vals.add(tok)
            # 也允许整段匹配
            if lbl in field_vals or any(lbl in fv for fv in field_vals):
                correct_field+=1
        # label 是否出现在 genres 列表里 (任意位置)
        if lbl in t: label_in_set+=1
    print(f"\n=== {dom} (评估 {evaluated}) ===")
    print(f"  有 Style/Genres 字段       : {has_field}/{evaluated} = {has_field/evaluated:.1%}")
    print(f"  label 在该字段值集合里     : {correct_field}/{evaluated} = {correct_field/evaluated:.1%}  <- 字符串解析准确率(无CLIP无GNN)")
    print(f"  label 在全文任意位置       : {label_in_set}/{evaluated} = {label_in_set/evaluated:.1%}")
