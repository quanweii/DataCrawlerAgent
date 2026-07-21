#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PATH-B Phase-0 · P3-probe: 语义脏数据检测, LLM vs 廉价 embedding 基线。

往 200 条干净 wikiart 注入 20 条语义脏数据(4 类×5):
  - semantic_dup   : 拷贝+新id+释义标题(embedding-dedup 应抓)
  - artist_variant : 拷贝+新id+艺术家缩写 "Vincent van Gogh"->"V. van Gogh"
  - mislabel       : 拷贝+新id+错误 style(label-centroid 应抓)
  - field_swap     : 拷贝+新id+交换 title<->artist(embedding 拼接是交换不变→抓不到;LLM 应抓)

廉价 embedding 基线(最强组合):
  - emb_dedup    : 文本最大余弦>thr → 近重复
  - emb_centroid : record 离自己 style 质心远 → 标签/字段异常
  combined = dedup OR centroid.

判读:LLM 召回 >> combined-embedding(尤其在 field_swap)→ P3 有真实价值点;
      combined-embedding ≈ LLM → P3 弱/死(任务有廉价代理)。
"""
from __future__ import annotations
import json, os, re, sys, time, math, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ITEMS = ROOT / "data/wikiart_raw/wikiart_v2_enriched.json"


def _load_env():
    envf = ROOT / ".env"
    for line in envf.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def embed_batch(texts, batch=10):
    from openai import OpenAI
    _load_env()
    c = OpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("MODEL_BASE_URL"))
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        for attempt in range(4):
            try:
                r = c.embeddings.create(model="text-embedding-v3", input=chunk)
                out.extend([d.embedding for d in r.data])
                break
            except Exception:
                if attempt < 3:
                    time.sleep(3 * (attempt + 1))
                else:
                    raise
    return out


def cosine(a, b):
    d = sum(x * y for x, y in zip(a, b))
    return d / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) + 1e-12)


# ---------- 注入 ----------
def make_dirty(clean, styles, rnd):
    """从 clean 抽 20 条基记录造 4 类脏数据。返回 (records, dirty_ids)。"""
    bases = rnd.sample(clean, 20)
    styles = [s for s in styles if s]
    recs, dirty = [], set()

    def copy(b, tag):
        r = dict(b); r["painting_id"] = b["painting_id"] + "_" + tag; return r

    # 1. semantic_dup: 释义标题
    for b in bases[0:5]:
        r = copy(b, "dup")
        toks = b["title"].split()
        if len(toks) > 1:
            rnd.shuffle(toks); r["title"] = " ".join(toks)
        else:
            r["title"] = b["title"] + " (study)"
        r["_dirty"] = "semantic_dup"; recs.append(r); dirty.add(r["painting_id"])
    # 2. artist_variant: 缩写
    for b in bases[5:10]:
        r = copy(b, "av")
        parts = b["artist"].split()
        if len(parts) >= 2:
            r["artist"] = f"{parts[0][0]}. " + " ".join(parts[1:])
        else:
            r["artist"] = b["artist"] + " (?)"
        r["_dirty"] = "artist_variant"; recs.append(r); dirty.add(r["painting_id"])
    # 3. mislabel: 错 style
    for b in bases[10:15]:
        r = copy(b, "ml")
        new_style = rnd.choice([s for s in styles if s != b.get("style")])
        r["style"] = new_style
        r["_dirty"] = "mislabel"; recs.append(r); dirty.add(r["painting_id"])
    # 4. field_swap: title<->artist
    for b in bases[15:20]:
        r = copy(b, "fs")
        r["title"], r["artist"] = b["artist"], b["title"]
        r["_dirty"] = "field_swap"; recs.append(r); dirty.add(r["painting_id"])
    return recs, dirty


# ---------- 廉价 embedding 检测 ----------
def cheap_detect(recs, embs, dedup_thr=0.86, centroid_low=0.45):
    n = len(recs)
    texts = [f"{r.get('title','')} | {r.get('artist','')}" for r in recs]
    # dedup: 每条对其它最大余弦
    maxsim = [0.0] * n
    for i in range(n):
        for j in range(n):
            if i != j:
                s = cosine(embs[i], embs[j])
                if s > maxsim[i]:
                    maxsim[i] = s
    dedup_flag = {recs[i]["painting_id"] for i in range(n) if maxsim[i] > dedup_thr}
    # centroid: 每 style 质心, record 离自己 style 质心余弦
    by_style = {}
    for i, r in enumerate(recs):
        by_style.setdefault(r.get("style", "?"), []).append(i)
    cent = {}
    for st, idxs in by_style.items():
        if not idxs:
            continue
        d = len(embs[0])
        v = [0.0] * d
        for i in idxs:
            for k in range(d):
                v[k] += embs[i][k]
        nrm = math.sqrt(sum(x * x for x in v)) + 1e-12
        cent[st] = [x / nrm for x in v]
    cent_flag = set()
    for i, r in enumerate(recs):
        st = r.get("style", "?")
        if st in cent:
            s = cosine(embs[i], cent[st])
            if s < centroid_low:
                cent_flag.add(r["painting_id"])
    return dedup_flag, cent_flag, dedup_flag | cent_flag


# ---------- LLM 审计 ----------
def llm_audit(recs, chunk=25):
    from openai import OpenAI
    _load_env()
    c = OpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("MODEL_BASE_URL"))
    model = os.getenv("MODEL_NAME", "qwen-plus")
    SYS = ("You are a data-quality auditor for an artwork graph. You receive a batch of records "
           "(id, title, artist, style, genre). Flag records that are: (a) a DUPLICATE/paraphrase "
           "of another record in the batch; (b) have the artist NAME ABBREVIATED oddly or "
           "inconsistent; (c) MISLABELED (style clearly wrong for the work); or (d) have FIELDS IN "
           "THE WRONG PLACE (e.g. a person's name in the title field, or artwork name in artist). "
           "Output ONLY compact JSON: {\"flag\":[{\"id\":\"...\",\"reason\":\"...\"}]} . "
           "If all clean, output {\"flag\":[]}.")
    flagged = {}
    for s in range(0, len(recs), chunk):
        batch = recs[s:s + chunk]
        lines = []
        for r in batch:
            lines.append(f'id={r["painting_id"]} | title={r.get("title","")} | '
                         f'artist={r.get("artist","")} | style={r.get("style","")} | '
                         f'genre={r.get("genre","")}')
        user = "Records:\n" + "\n".join(lines) + "\n\nFlag dirty ones."
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
        txt = ""
        for attempt in range(4):
            try:
                r = c.chat.completions.create(model=model, temperature=0.0, max_tokens=1500,
                                              timeout=90, messages=msgs)
                txt = (r.choices[0].message.content or "").strip()
                break
            except Exception:
                if attempt < 3:
                    time.sleep(4 * (attempt + 1))
        m = re.search(r"\{[\s\S]*\}", txt)
        if m:
            try:
                pol = json.loads(m.group(0))
                for e in pol.get("flag", []):
                    if isinstance(e, dict) and e.get("id"):
                        flagged[e["id"]] = e.get("reason", "")
            except json.JSONDecodeError:
                pass
    return flagged


def main(n_clean=200):
    rnd = random.Random(1)
    items = json.load(open(ITEMS))
    clean = rnd.sample(items, n_clean)
    styles = list({it.get("style") for it in items})
    dirty_recs, dirty_ids = make_dirty(items, styles, rnd)
    recs = clean + dirty_recs
    id2dirty = {r["painting_id"]: r.get("_dirty") for r in recs}
    print(f"corpus: {len(clean)} clean + {len(dirty_recs)} dirty = {len(recs)}")

    texts = [f"{r.get('title','')} | {r.get('artist','')} | {r.get('genre','')}" for r in recs]
    print(f"embedding {len(texts)} records...")
    embs = embed_batch(texts, batch=10)

    dedup, cent, combined = cheap_detect(recs, embs)
    llm = llm_audit(recs, chunk=25)

    # 打分
    def per_type(flagset):
        out = {}
        for t in ("semantic_dup", "artist_variant", "mislabel", "field_swap"):
            ids = [rid for rid, dt in id2dirty.items() if dt == t]
            out[t] = sum(1 for rid in ids if rid in flagset) / max(len(ids), 1)
        clean_ids = [r["painting_id"] for r in recs if not r.get("_dirty")]
        out["clean_FP"] = sum(1 for rid in clean_ids if rid in flagset) / len(clean_ids)
        out["dirty_recall"] = sum(1 for rid in dirty_ids if rid in flagset) / len(dirty_ids)
        return out

    arms = {"emb_dedup": dedup, "emb_centroid": cent, "emb_combined": combined, "LLM": set(llm)}
    print("\n=== P3-probe: per-type recall + clean false-positive ===")
    print(f"{'arm':16s} {'dup':>5s} {'artist':>6s} {'mislbl':>6s} {'fswap':>6s} "
          f"{'dirty_rec':>9s} {'cleanFP':>8s}")
    summary = {}
    for name, fs in arms.items():
        s = per_type(fs)
        summary[name] = s
        print(f"{name:16s} {s['semantic_dup']:5.2f} {s['artist_variant']:6.2f} "
              f"{s['mislabel']:6.2f} {s['field_swap']:6.2f} {s['dirty_recall']:9.2f} "
              f"{s['clean_FP']:8.2f}")
    print("\n判读:field_swap 列 LLM>>emb → 语义结构错误是 LLM 独有价值点;")
    print("      若 emb_combined 召回≈LLM → 任务有廉价代理,P3 弱。")
    (ROOT / "benchmark/results/path_b_p3_probe.json").write_text(
        json.dumps({"summary": summary, "llm_reasons": llm}, ensure_ascii=False, indent=2))
    print("saved -> benchmark/results/path_b_p3_probe.json")


if __name__ == "__main__":
    main(n_clean=int(sys.argv[1]) if len(sys.argv) > 1 else 200)
