#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E3B-探测: 对比 generation-agent 合成数据 vs 真实+去泄露数据的泄露 profile。
复用论文 label-leakage 诊断的同一指标(hit_substr / hit_field, 见 diag_text_label_leakage.py),
并加 TF-IDF+LR "语义泄露上限"(text 能多大程度恢复 label)。
论点: 合成数据 label 被 LLM 织进语义(不可像字段一样剥离); 真实+去泄露数据 label-in-text≈0。
"""
import json, re, sys, random
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
GRAPH = {"tmdb": "data/tmdb_datasets/tmdb_multimodal_graph_agent.json",
         "steam": "data/steam_datasets/steam_multimodal_graph_agent.json"}
RAW = {"tmdb": "data/tmdb_movies", "steam": "data/steam_games"}
KEEP_TOP = 8  # 与合成同 label 集

def load_text(dom, nid):
    base = Path(RAW[dom])
    for cand in [base/nid/f"{nid}.md", base/f"{nid}.md", base/nid]:
        if cand.is_file():
            try: return cand.read_text(encoding="utf-8", errors="ignore")
            except Exception: return None
    return None

def norm(s): return re.sub(r"\s+", " ", (s or "").lower()).strip()

# 修 bug: 原 regex 不匹配复数 "Genres"/"Styles"; 现支持单复数
FIELD_RE = r"(style|styles|genre|genres|label|labels|category|categories|tags?)\s*[:=]"
# 论文 C2 的字段剥离(strip 整行 "Genres: ..." / "Style: ...")
STRIP_LINE_RE = re.compile(r"^\s*(style|styles|genre|genres|label|labels|category|categories|tags?)\s*[:=].*$",
                           re.IGNORECASE | re.MULTILINE)

def strip_label_fields(text):
    return STRIP_LINE_RE.sub("", text or "")

def label_in_text_metrics(items):
    """items = [{label,text}]; 复用论文 hit_substr/hit_field 口径(修单复数 bug)。"""
    n = len(items); hit_sub = 0; hit_field = 0
    for it in items:
        lbl = norm(it["label"]); t = norm(it["text"])
        if not lbl or not t: continue
        if lbl in t: hit_sub += 1
        if re.search(rf"{FIELD_RE}\s*[^a-z0-9]*{re.escape(lbl)}", t):
            hit_field += 1
    return {"n": n, "hit_substr": round(hit_sub/n, 3) if n else 0,
            "hit_field": round(hit_field/n, 3) if n else 0}

def tfidf_lr_acc(items):
    """TF-IDF + LR 5-fold —— text→label 可恢复性上限(语义泄露代理)。"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    texts = [it["text"] for it in items if it.get("text")]
    labels = [it["label"] for it in items if it.get("text")]
    if len(set(labels)) < 2: return None
    X = TfidfVectorizer(max_features=5000, ngram_range=(1, 2)).fit_transform(texts)
    s = cross_val_score(LogisticRegression(max_iter=400), X, labels, cv=5, scoring="accuracy")
    return round(s.mean(), 3)

def load_real_sample(dom, per_class=30):
    g = json.load(open(ROOT/GRAPH[dom]))
    nodes = g["nodes"]
    # 按 label 分桶, 取 top-K, 每类采样 per_class
    by_lab = {}
    for nd in nodes:
        l = norm(nd["attributes"].get("label"))
        if l: by_lab.setdefault(l, []).append(nd["id"])
    top = [l for l, _ in Counter({k: len(v) for k, v in by_lab.items()}).most_common(KEEP_TOP)]
    rnd = random.Random(7)
    out = []
    for l in top:
        ids = by_lab[l]
        for nid in rnd.sample(ids, min(per_class, len(ids))):
            t = load_text(dom, nid)
            if t: out.append({"label": l, "text": t})
    return out

def main():
    res = {}
    for dom in ("tmdb", "steam"):
        syn = json.load(open(ROOT/f"data/e3b_synthetic_{dom}.json"))
        real = load_real_sample(dom)
        # 合成也过一遍 strip(证明它无字段可剥, 泄露是语义级)
        syn_stripped = [{"label": it["label"], "text": strip_label_fields(it["text"])} for it in syn]
        real_stripped = [{"label": it["label"], "text": strip_label_fields(it["text"])} for it in real]
        conds = {
            "synthetic_raw": syn, "synthetic_stripped": syn_stripped,
            "real_raw_on_disk": real, "real_after_C2_strip": real_stripped,
        }
        out = {}
        print(f"\n=== {dom} ===")
        for name, items in conds.items():
            m = label_in_text_metrics(items)
            acc = tfidf_lr_acc(items)
            out[name] = {**m, "tfidf_lr_acc": acc}
            print(f"  {name:22s}: hit_substr={m['hit_substr']:.1%}  hit_field={m['hit_field']:.1%}  TF-IDF+LR acc={acc}")
        res[dom] = out
    (ROOT/"benchmark/results/e3b_synthetic_leakage.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print("\nsaved -> benchmark/results/e3b_synthetic_leakage.json")

if __name__ == "__main__":
    main()
