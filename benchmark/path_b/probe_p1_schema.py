#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PATH-B Phase-0 · P1-probe: schema 发现是否是 agent 的真实价值点？

问题：藏起人写 YAML，给 LLM 一份"数据画像"(字段统计 + 少量样例)，
单次让它推断 id_field / label_field / relations。能否恢复金标准？
关键杀手测试(spotify)：金标准【刻意不】建 shared_genre 关系——因为
playlist_genre 就是 label，建它=定义性泄露。一个脚本启发式会无脑建它；
LLM 若有语义理解应避免它。

对照：
  B1 naive     : 名字启发式(id 类名→id, category 类名→label, 其余重复值字段→shared_<f>)
  B2 structural: 更强确定性基线(剔除 label/纯数值/过高基数, 留合理基数字段)
  LLM(qwen+)   : 单次 schema 推断(给数据画像), 多 seed 看方差

打分(对每个金标准域):
  id_field 命中 / label_field 命中(base 字段名) / relations 召回(group_by 命中)
  / 定义性泄露: 是否提了 group_by==label_field 的关系(spotify 金标准=否)
"""
from __future__ import annotations
import json, os, re, sys, time
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]


# ---------- env / LLM ----------
def _load_env():
    envf = ROOT / ".env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _call_llm(system, user, temperature=0.2, max_tokens=2048, seed=0):
    from openai import OpenAI
    _load_env()
    c = OpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("MODEL_BASE_URL"))
    model = os.getenv("MODEL_NAME", "qwen-plus")
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for max_tok in (max_tokens, 4096):
        for attempt in range(3):
            try:
                r = c.chat.completions.create(
                    model=model, temperature=temperature, max_tokens=max_tok,
                    timeout=90, messages=msgs, seed=seed)
                txt = (r.choices[0].message.content or "").strip()
                if txt:
                    m = re.search(r"\{[\s\S]*\}", txt)
                    if m:
                        try:
                            return json.loads(m.group(0)), txt
                        except json.JSONDecodeError:
                            pass
                if r.choices[0].finish_reason != "length":
                    return None, txt
                break
            except Exception:
                if attempt < 2:
                    time.sleep(4 * (2 ** attempt))
                else:
                    return None, ""
    return None, ""


# ---------- 数据画像 ----------
def _base_field(field_ref: str) -> str:
    """genres[0] -> genres ; a.b -> a"""
    return re.split(r"[\[\.]", field_ref)[0]


def profile_fields(items):
    fields = list(items[0].keys())
    prof = []
    for f in fields:
        vals = [it.get(f) for it in items]
        nonempty = [v for v in vals if v not in (None, "", [], {})]
        n_unique = len(set(map(str, nonempty)))
        # dtype
        sample = nonempty[:5]
        if sample and isinstance(sample[0], list):
            dt = "list"
        elif sample and isinstance(sample[0], (int, float)) and not isinstance(sample[0], bool):
            dt = "number"
        else:
            dt = "str"
        samp = [str(v)[:40] for v in sample]
        prof.append({
            "name": f, "dtype": dt, "n_unique": n_unique,
            "n_missing": len(vals) - len(nonempty), "sample_values": samp,
        })
    return prof


# ---------- 金标准 ----------
GOLD = {
    "spotify": {
        "id": "track_id", "label": "playlist_genre",
        "relations": [("same_artist", "track_artist"), ("same_album", "track_album_id")],
        "avoid_definitional": True,   # 金标准刻意不建 shared_genre
    },
    "wikiart": {
        "id": "painting_id", "label": "style",
        "relations": [("same_artist", "artist"), ("shared_style", "style")],
        "avoid_definitional": False,  # 金标准含定义性 shared_style
    },
    "steam": {
        "id": "appid", "label": "genres",
        "relations": [("shared_genre", "genres"), ("same_developer", "developers"),
                      ("same_publisher", "publishers"), ("shared_category", "categories")],
        "avoid_definitional": False,
    },
    "tmdb": {
        "id": "movie_id", "label": "genres",
        "relations": [("shared_genre", "genres"), ("same_director", "director"),
                      ("shared_actor", "actors")],
        "avoid_definitional": False,
    },
}
DATA_FILE = {
    "spotify": "./data/spotify_raw/spotify_tracks.json",
    "wikiart": "./data/wikiart_raw/wikiart_v2_enriched.json",
    "steam": "./data/steam_raw/steam_games_10k.json",
    "tmdb": "./data/tmdb_raw/tmdb_movies_raw.json",
}


# ---------- 脚本基线 ----------
def baseline_naive(prof, items):
    """B1: 名字启发式。id=*_id/id/appid; label=*genre/category/style/...;
    其余有重复值的非数值字段 -> shared_<f>(会包含 label 字段=泄露)."""
    names = [p["name"] for p in prof]
    idf = next((n for n in names if re.search(r"(^id$|_id$|^appid$)", n)), names[0])
    lab = next((n for n in names if re.search(r"(genre|categ|style|label|type|class)", n)),
               None) or names[0]
    rels = []
    for p in prof:
        n = p["name"]
        if n == idf:
            continue
        if p["dtype"] == "number":
            continue
        if p["n_unique"] < len(items):  # 有重复值
            rels.append((f"shared_{n}", n))
    return idf, lab, rels


def baseline_structural(prof, items):
    """B2: 更强。剔除 label/纯数值/基数过高(>500)或过低(<2), 留合理基数 co-attr."""
    names = [p["name"] for p in prof]
    idf = next((n for n in names if re.search(r"(^id$|_id$|^appid$)", n)), names[0])
    lab = next((n for n in names if re.search(r"(genre|categ|style|label|type|class)", n)),
               None) or names[0]
    rels = []
    for p in prof:
        n = p["name"]
        if n in (idf, lab):
            continue
        if p["dtype"] == "number":
            continue
        if not (2 <= p["n_unique"] <= 500):
            continue
        rels.append((f"shared_{n}", n))
    return idf, lab, rels


# ---------- LLM schema 推断 ----------
SYS = ("You are a multimodal-attribute-graph schema designer. Given a data profile of a new "
       "domain, design the graph config. Pick: (1) id_field = the unique identifier field; "
       "(2) label_field = ONE meaningful categorical field with a SMALL number of balanced classes "
       "(the node-classification target — NOT a unique/high-cardinality field); (3) relations = a "
       "list of co-attribute edges, each {name, group_by, cap}, where group_by is a field whose "
       "SHARED value meaningfully connects items (artists, albums, categories...). "
       "CRITICAL decontamination rule: do NOT create a relation whose group_by is the same field "
       "as label_field — such edges directly reveal the label (a definitional leak). Choose cap "
       "10-100 by how coarse the grouping is. Output ONLY compact JSON, no prose:\n"
       '{"id_field":"...","label_field":"...","relations":[{"name":"...","group_by":"...","cap":30}]}')


def llm_schema(prof, items, seed=0):
    """单次 LLM 推断 schema。返回 (id_field, label_field, [(name,group_by)...], raw_text)."""
    prof_str = "\n".join(
        f"- {p['name']} : dtype={p['dtype']}, n_unique={p['n_unique']}, "
        f"n_missing={p['n_missing']}, sample={p['sample_values']}" for p in prof)
    ex = json.dumps(items[:6], ensure_ascii=False, indent=1)[:3500]
    user = (f"Domain profile (N={len(items)} items):\n{prof_str}\n\n"
            f"Example records:\n{ex}\n\nDesign the schema.")
    pol, raw = _call_llm(SYS, user, temperature=0.2, max_tokens=1500, seed=seed)
    if not pol:
        return None, None, [], raw
    idf = pol.get("id_field")
    lab = pol.get("label_field")
    rels = []
    for r in pol.get("relations", []):
        gb = r.get("group_by")
        if gb:
            rels.append((r.get("name", f"shared_{gb}"), gb))
    return idf, lab, rels, raw


# ---------- 打分 ----------
def score(idf, lab, rels, gold):
    g_id, g_lab, g_rels = gold["id"], _base_field(gold["label"]), gold["relations"]
    g_lab_base = _base_field(gold["label"])
    id_ok = (idf == g_id)
    lab_ok = (_base_field(lab) == g_lab_base) if lab else False
    g_gb = {gb for _, gb in g_rels}
    p_gb = {gb for _, gb in rels}
    rel_recall = len(g_gb & p_gb) / max(len(g_gb), 1)
    rel_prec = len(g_gb & p_gb) / max(len(p_gb), 1)
    # 定义性泄露: 是否提了 group_by==label_base 的关系
    has_defn = any(_base_field(gb) == g_lab_base for _, gb in rels)
    defn_ok = (not has_defn) if gold["avoid_definitional"] else (has_defn or True)
    # 对 avoid 域: defn_clean = 没提 label 关系; 对非 avoid 域: 不扣分(金标准本身含)
    defn_clean = (not has_defn) if gold["avoid_definitional"] else None
    return {
        "id_ok": id_ok, "lab_ok": lab_ok,
        "rel_recall": round(rel_recall, 2), "rel_prec": round(rel_prec, 2),
        "n_rel_proposed": len(rels),
        "has_definitional": has_defn,
        "defn_clean": defn_clean,  # True=正确避开(avoid域) / None=不适用
        "proposed_group_by": sorted(p_gb),
    }


def run_domain(domain, n_sample=200, seeds=(1, 2, 3)):
    import random
    items = json.load(open(ROOT / DATA_FILE[domain]))
    rnd = random.Random(1)
    sample = rnd.sample(items, min(n_sample, len(items)))
    prof = profile_fields(sample)
    gold = GOLD[domain]

    out = {"domain": domain, "gold": {
        "id": gold["id"], "label": gold["label"],
        "relations_group_by": sorted({gb for _, gb in gold["relations"]}),
        "avoid_definitional": gold["avoid_definitional"]}}
    for bname, bfn in [("B1_naive", baseline_naive), ("B2_structural", baseline_structural)]:
        bi, bl, br = bfn(prof, sample)
        out[bname] = {"id": bi, "label": bl, "rels_gb": sorted({g for _, g in br}),
                      "score": score(bi, bl, br, gold)}
    llm_runs = []
    for s in seeds:
        li, ll, lr, raw = llm_schema(prof, sample, seed=s)
        if li is None:
            llm_runs.append({"seed": s, "error": "no_parse",
                             "raw_head": (raw or "")[:200]})
            continue
        llm_runs.append({"seed": s, "id": li, "label": ll,
                         "rels_gb": sorted({g for _, g in lr}),
                         "score": score(li, ll, lr, gold)})
    out["LLM_qwen"] = llm_runs
    return out


def print_domain(out):
    d = out["domain"]
    g = out["gold"]
    print(f"\n{'='*70}\n[{d}]  GOLD: id={g['id']}  label={g['label']}  "
          f"rel_gb={g['relations_group_by']}  avoid_defn={g['avoid_definitional']}")
    for arm in ("B1_naive", "B2_structural"):
        s = out[arm]["score"]
        print(f"  {arm:14s}: id={s['id_ok']} lab={s['lab_ok']} "
              f"rel_recall={s['rel_recall']} rel_prec={s['rel_prec']} "
              f"n_prop={s['n_rel_proposed']} has_defn={s['has_definitional']} "
              f"defn_clean={s['defn_clean']}")
        print(f"               proposed_gb={out[arm]['rels_gb']}")
    runs = out["LLM_qwen"]
    ok_runs = [r for r in runs if "score" in r]
    print(f"  LLM_qwen ({len(ok_runs)}/{len(runs)} parsed):")
    if not ok_runs:
        for r in runs:
            print(f"    seed={r['seed']}: {r.get('error')} :: {r.get('raw_head','')[:120]}")
    else:
        for r in ok_runs:
            s = r["score"]
            print(f"    seed={r['seed']}: id={s['id_ok']} lab={s['lab_ok']} "
                  f"rel_recall={s['rel_recall']} rel_prec={s['rel_prec']} "
                  f"n_prop={s['n_rel_proposed']} has_defn={s['has_definitional']} "
                  f"defn_clean={s['defn_clean']}")
            print(f"             proposed_gb={r['rels_gb']}  (label picked: {r['label']})")


if __name__ == "__main__":
    domains = sys.argv[1:] or ["spotify", "wikiart", "steam", "tmdb"]
    summary = {}
    for dom in domains:
        try:
            out = run_domain(dom)
            print_domain(out)
            summary[dom] = out
        except Exception as e:
            import traceback
            print(f"[{dom}] ERROR: {e}")
            traceback.print_exc()
    (ROOT / "benchmark/results/path_b_p1_probe.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nsaved -> benchmark/results/path_b_p1_probe.json")
