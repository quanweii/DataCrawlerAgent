#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PATH-B Phase-0 · P2-probe: 引用对齐里 embedding 能否追平 LLM？

公平设计：对每个 ref，用 word_index(倒排) 重建候选集(= rule 搜的同一批硬负例)，
对【同一候选短名单】分别打分：
  - rule     : SequenceMatcher.ratio() (现有脚本)
  - embedding: text-embedding-v3 cosine (廉价 ML 基线)
  - LLM      : qwen-plus 在候选里选 (语义)
比较 top-1 recall，按难度桶(SequenceMatcher(ref,gold_title) 的相似度分桶)。
杀手问题：fuzzy 桶里 embedding_recall ≈ LLM_recall 吗？
  是 → embedding 是强廉价代理 → Pillar 2 弱。
  否(LLM >> embedding) → LLM 语义匹配有真实价值。
"""
from __future__ import annotations
import json, os, re, sys, time, pickle, math
from pathlib import Path
from difflib import SequenceMatcher

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "benchmark/results/e2_citation_ablation/meta_index.pkl"
REFS = ROOT / "benchmark/results/e2_citation_ablation/e2_results.json"

STOP = set("a an the of in for and on to with by at from is are was were be been as or "
           "not no but its it this that we our their can will all using based".split())


def _load_env():
    envf = ROOT / ".env"
    for line in envf.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def normalize(t: str) -> str:
    t = t.lower()
    t = re.sub(r"(\w)-\s+([a-z])", r"\1\2", t)        # 去 OCR 换行连字符
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def sig_words(t: str):
    return {w for w in t.split() if len(w) >= 3 and w not in STOP}


# ---------- embedding ----------
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
            except Exception as e:
                if attempt < 3:
                    time.sleep(3 * (attempt + 1))
                else:
                    raise
    return out


def cosine(a, b):
    d = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return d / (na * nb + 1e-12)


# ---------- LLM ----------
def _call_llm(system, user, temperature=0.0, max_tokens=64):
    from openai import OpenAI
    _load_env()
    c = OpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("MODEL_BASE_URL"))
    model = os.getenv("MODEL_NAME", "qwen-plus")
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for attempt in range(4):
        try:
            r = c.chat.completions.create(model=model, temperature=temperature,
                                          max_tokens=max_tokens, timeout=60, messages=msgs)
            return (r.choices[0].message.content or "").strip()
        except Exception:
            if attempt < 3:
                time.sleep(3 * (attempt + 1))
            else:
                return ""


def llm_pick(ref, cands):
    """cands: list of (id, title, authors). 返回选中的 id 或 None."""
    SYS = ("You match a bibliographic reference to its paper among candidates. Pick the ONE "
           "candidate that is the SAME paper as the reference. Tolerate OCR noise, hyphenation, "
           "word reordering, and abbreviation. If none matches, answer NONE. "
           "Reply with ONLY the candidate number (1-N) or NONE — no explanation.")
    lines = "\n".join(f"{i+1}. {t} — {a}" for i, (_, t, a) in enumerate(cands))
    user = f"Reference: {ref}\n\nCandidates:\n{lines}\n\nAnswer:"
    txt = _call_llm(SYS, user)
    m = re.search(r"\b(\d+)\b", txt)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(cands):
            return cands[idx][0]
    if txt.upper().startswith("NONE") or "NONE" in txt.upper():
        return None
    return None


def bucket_of(ratio):
    if ratio >= 0.95:
        return "clean(>=0.95)"
    if ratio >= 0.90:
        return "medium(0.90-0.95)"
    return "fuzzy(<0.90)"


def main(max_refs=60, cap_cand=30):
    meta = pickle.load(open(META, "rb"))
    title_to_id = meta["title_to_id"]
    id_to_title = meta["id_to_title"]
    id_to_authors = meta.get("id_to_authors", {})
    widx = meta["word_index"]

    rows = json.load(open(REFS))["part1_rows"][:max_refs]
    print(f"loaded {len(rows)} refs")

    # 1) 重建每个 ref 的候选集：强制 gold 进 + top 硬负例(按词重叠降序)。
    # 目的是隔离【匹配/排序】能力——gold 一定在候选里，看谁排第一。
    ref_cands = []
    for row in rows:
        ref, gold, gold_title = row["ref"], row["gold"], row["gold_title"]
        rn = normalize(ref)
        sw = sig_words(rn)
        scores = {}
        for w in sw:
            for title in widx.get(w, ()):
                scores[title] = scores.get(title, 0) + 1
        # 词重叠降序的硬负例(≥1 重叠), 排除 gold 自身
        gold_t_norm = normalize(gold_title)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        distractor_titles = [t for t, c in ranked if c >= 1 and t != gold_t_norm][:cap_cand - 1]
        selected = [gold_t_norm] + distractor_titles
        cands = []
        for t in selected:
            cid = title_to_id.get(t)
            if cid is None and t == gold_t_norm:
                cid = gold  # gold 一定进
            if cid is None:
                continue
            cands.append((cid, id_to_title.get(cid, t), id_to_authors.get(cid, "")))
        cands = cands[:cap_cand]
        ref_cands.append((row, rn, gold, cands))

    # 2) embed 所有 ref + 候选 title (去重)
    all_texts = []
    seen = {}
    for row, rn, gold, cands in ref_cands:
        if rn not in seen:
            seen[rn] = len(all_texts); all_texts.append(rn)
        for cid, t, a in cands:
            tn = normalize(t)
            if tn not in seen:
                seen[tn] = len(all_texts); all_texts.append(tn)
    print(f"embedding {len(all_texts)} unique texts...")
    embs = embed_batch(all_texts, batch=10)
    E = {i: embs[i] for i in range(len(all_texts))}

    # 3) 逐 ref 打分
    results = []
    for row, rn, gold, cands in ref_cands:
        ref_e = E[seen[rn]]
        # rule
        rule_scores = [(SequenceMatcher(None, rn, normalize(t)).ratio(), cid) for cid, t, a in cands]
        rule_scores.sort(reverse=True)
        rule_pick = rule_scores[0][1] if rule_scores and rule_scores[0][0] >= 0.90 else None
        # embedding
        emb_scores = [(cosine(ref_e, E[seen[normalize(t)]]), cid) for cid, t, a in cands]
        emb_scores.sort(reverse=True)
        emb_pick = emb_scores[0][1] if emb_scores else None
        # llm
        lp = llm_pick(row["ref"], cands)
        # bucket by ref-gold string similarity
        ratio = SequenceMatcher(None, rn, normalize(row["gold_title"])).ratio()
        results.append({
            "bucket": bucket_of(ratio), "gold": gold,
            "n_cand": len(cands), "gold_in_cand": any(c[0] == gold for c in cands),
            "rule_correct": rule_pick == gold, "emb_correct": emb_pick == gold,
            "llm_correct": lp == gold,
            "rule_score": round(rule_scores[0][0], 3) if rule_scores else None,
            "emb_score": round(emb_scores[0][0], 3) if emb_scores else None,
        })

    # 4) 分桶汇总
    print("\n=== P2-probe: per-bucket top-1 recall (rule vs embedding vs LLM) ===")
    print(f"{'bucket':22s} {'n':>3s} {'gold_in':>7s} {'rule':>6s} {'emb':>6s} {'llm':>6s}")
    summary = {}
    for b in ["clean(>=0.95)", "medium(0.90-0.95)", "fuzzy(<0.90)"]:
        rs = [r for r in results if r["bucket"] == b]
        if not rs:
            continue
        n = len(rs)
        gi = sum(r["gold_in_cand"] for r in rs) / n
        rc = sum(r["rule_correct"] for r in rs) / n
        ec = sum(r["emb_correct"] for r in rs) / n
        lc = sum(r["llm_correct"] for r in rs) / n
        summary[b] = {"n": n, "gold_in_cand": round(gi, 3),
                      "rule": round(rc, 3), "emb": round(ec, 3), "llm": round(lc, 3)}
        print(f"{b:22s} {n:3d} {gi:7.3f} {rc:6.3f} {ec:6.3f} {lc:6.3f}")
    alln = len(results)
    summary["ALL"] = {
        "n": alln,
        "rule": round(sum(r["rule_correct"] for r in results) / alln, 3),
        "emb": round(sum(r["emb_correct"] for r in results) / alln, 3),
        "llm": round(sum(r["llm_correct"] for r in results) / alln, 3)}
    print(f"{'ALL':22s} {alln:3d} {'-':>7s} {summary['ALL']['rule']:6.3f} "
          f"{summary['ALL']['emb']:6.3f} {summary['ALL']['llm']:6.3f}")
    print("\n判读:fuzzy 桶 emb≈llm → embedding 是廉价代理(P2弱); llm>>emb → LLM 语义有真实价值")
    out = {"summary": summary, "rows": results}
    (ROOT / "benchmark/results/path_b_p2_probe.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2))
    print("saved -> benchmark/results/path_b_p2_probe.json")


if __name__ == "__main__":
    main(max_refs=int(sys.argv[1]) if len(sys.argv) > 1 else 60)
