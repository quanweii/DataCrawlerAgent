#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PATH-B · P6: 修 C1 软肋——关系召回(从"有效子集"升到"接近人写")。

问题: P4/P5 的 agent 只提议显眼的 1-2 个关系(spotify 只 track_artist, 漏 track_album_id)。
根因双重: ① LLM 不系统枚举所有字段; ② 严格结构阈值(frac_shared≥0.3)漏掉稀疏但有效的关系
   (如 spotify same_album 仅 181 边, 金标准保留)。

修法(诚实分工):
  1. LLM 发现 label(P4 已证可行)
  2. 【新】确定性【宽松】枚举: 产生 ≥少量共享边 + 非连续 + 非id + 非label 的字段全列候选
  3. 【新】LLM 语义 curate: 从候选丢垃圾(url/slug/date/freetext)、命名、keep
  4. verifier 安全网: 丢定义性 + 退化(P4 已有)
高召回靠枚举, 精度靠 LLM curate + verifier。
"""
from __future__ import annotations
import json, os, re, sys, time, random
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark" / "path_b"))
from probe_p4_autonomous_skill import (  # noqa
    measure, deterministic_critique, _parse_schema, SYS_PROPOSE)
from probe_p1_schema import _call_llm, profile_fields, _base_field, GOLD, DATA_FILE  # noqa


def _is_float(v):
    try:
        float(v); return True
    except (TypeError, ValueError):
        return False


SYS_CURATE = ("You curate candidate co-attribute relations for a multimodal attribute graph. "
    "For each candidate field (with sample values + stats), decide if it forms a MEANINGFUL "
    "shared-attribute relation. KEEP real shared categories/groups (artist, album, brand, "
    "developer, publisher, category, actor, director, studio). KEEP fields named like *_id / "
    "*_id* that GROUP MULTIPLE items — these are foreign-key relations (album_id→same_album, "
    "artist_id→same_artist), NOT unique identifiers. DROP: URLs, slugs, file paths, free-text/"
    "descriptions, raw timestamps/dates, image URLs, or anything not a genuine shared attribute. "
    "Give each KEPT relation a short name (same_X for identity-like, shared_X for category-like). "
    'Output ONLY JSON: [{"field":"...","keep":true,"name":"..."}, ...] (include ALL candidates).')


def enumerate_loose(items, id_field, label_field, min_edges=None):
    """宽松枚举: 产生 ≥min_edges 共享边 + 非连续 + 非id + 非label。捕捉稀疏但有效关系。"""
    if min_edges is None:
        min_edges = max(20, len(items) * 0.005)
    cands = []
    for f in items[0].keys():
        if f == id_field or _base_field(f) == _base_field(label_field):
            continue
        groups = defaultdict(list)
        for it in items:
            iid = it.get(id_field)
            v = it.get(f)
            vals = v if isinstance(v, list) else [v]
            for x in vals:
                if x not in (None, "", []) and iid is not None:
                    groups[str(x)].append(iid)
        sizes = [len(s) for s in groups.values()]
        n_edges = sum(s * (s - 1) // 2 for s in sizes)     # 共享边数
        samp = list(groups.keys())[:200]
        frac_cont = sum(_is_float(v) for v in samp) / max(len(samp), 1)
        if n_edges >= min_edges and frac_cont < 0.5 and len(groups) >= 2:
            cands.append({"field": f, "n_edges": n_edges, "n_groups": len(groups),
                          "frac_continuous": round(frac_cont, 2),
                          "sample_values": samp[:5]})
    return cands


def curate_llm(cands, seed=1):
    """LLM 从候选丢垃圾+命名。返回 [(name, field), ...] kept。"""
    cand_str = "\n".join(
        f"- field={c['field']}: n_edges={c['n_edges']}, n_groups={c['n_groups']}, "
        f"frac_continuous={c['frac_continuous']}, sample={c['sample_values']}" for c in cands)
    _, raw = _call_llm(SYS_CURATE, f"Candidate fields:\n{cand_str}\n\nCurate them.",
                       temperature=0.1, max_tokens=1200, seed=seed)
    m = re.search(r"\[[\s\S]*\]", raw or "")
    if not m:
        # 兜底: 全 keep (高召回)
        return [(f"shared_{c['field']}", c["field"]) for c in cands]
    try:
        decisions = json.loads(m.group(0))
    except json.JSONDecodeError:
        return [(f"shared_{c['field']}", c["field"]) for c in cands]
    kept = []
    for d in decisions:
        if d.get("keep") and d.get("field"):
            name = d.get("name") or f"shared_{d['field']}"
            kept.append((name, d["field"]))
    return kept or [(f"shared_{c['field']}", c["field"]) for c in cands]   # 兜底全 keep


def discover_label(profile_str, seed=1):
    user = f"Domain profile:\n{profile_str}\n\nDesign the schema."
    _, raw = _call_llm(SYS_PROPOSE, user, temperature=0.2, max_tokens=800, seed=seed)
    prop = _parse_schema(raw)
    return (prop or {}).get("id_field"), (prop or {}).get("label_field")


def semantic_definitional(items, group_by, label_field, thr=0.95, card_ratio_max=1.5):
    """group_by 是否是 label 的【重编码】(定义性泄露)。需同时满足:
       ① group_by→label 近函数依赖(functional>thr); ② 基数相近(n_groups/n_label_classes<card_ratio_max)。
    这区分 style_slug(64≈style64, 重编码=泄露) vs brand(1849≫main_cat7, 只是相关≠泄露)。"""
    def labval(it):
        v = it.get(_base_field(label_field))
        parts = re.split(r"[\[\]]", label_field)
        if len(parts) > 1 and parts[1].isdigit() and isinstance(v, list):
            v = v[int(parts[1])] if int(parts[1]) < len(v) else None
        return None if v in (None, "", []) else str(v)
    g2l = defaultdict(set)
    label_classes = set()
    for it in items:
        gv = it.get(group_by)
        vals = gv if isinstance(gv, list) else [gv]
        lv = labval(it)
        if lv is not None:
            label_classes.add(lv)
        for x in vals:
            if x not in (None, "", []) and lv is not None:
                g2l[str(x)].add(lv)
    if not g2l or not label_classes:
        return 0.0
    functional = sum(1 for s in g2l.values() if len(s) == 1) / len(g2l)
    card_ratio = len(g2l) / max(len(label_classes), 1)
    if functional > thr and card_ratio < card_ratio_max:
        return functional     # 重编码 → 定义性
    return 0.0                # 相关但不同 → 不是定义性


def autonomous_v2(items, seed=1, max_rounds=3):
    """LLM 发现 label → 宽松枚举候选 → LLM curate → verifier 安全网(含语义定义性检查)。"""
    rnd = random.Random(1)
    sample = rnd.sample(items, min(300, len(items)))
    prof = profile_fields(sample)
    prof_str = "\n".join(f"- {p['name']}: dtype={p['dtype']}, n_unique={p['n_unique']}, "
                         f"sample={p['sample_values'][:3]}" for p in prof)
    idf, lab = discover_label(prof_str, seed=seed)
    if not lab:
        return None
    cands = enumerate_loose(items, idf, lab)
    kept = curate_llm(cands, seed=seed)
    proposal = {"id_field": idf, "label_field": lab,
                "relations": [{"name": n, "group_by": g, "cap": 30} for n, g in kept]}
    # verifier 安全网: 丢定义性(名字级) + 退化 + 语义定义性(group_by→label 近函数依赖)
    for _ in range(max_rounds):
        meas = measure(items, proposal)
        flags = deterministic_critique(meas)
        bad_gb = set()
        for fl in flags:
            m = re.search(r"group_by=([^ )]+)", fl)
            if m:
                bad_gb.add(m.group(1))
        for r in list(proposal["relations"]):
            if semantic_definitional(items, r["group_by"], lab) > 0.95:
                bad_gb.add(r["group_by"])
        if not bad_gb:
            break
        proposal["relations"] = [r for r in proposal["relations"] if r["group_by"] not in bad_gb]
        if not proposal["relations"]:
            break
    return proposal


def score_vs_gold(proposal, gold, items):
    if proposal is None:
        return None
    g_lab = _base_field(gold["label"])
    g_rels_nondefn = {_base_field(gb) for _, gb in gold["relations"] if _base_field(gb) != g_lab}
    p_rels = {_base_field(r["group_by"]) for r in proposal["relations"]}
    has_defn = any(_base_field(r["group_by"]) == g_lab for r in proposal["relations"])
    meas = measure(items, proposal)
    valid = [r for r in meas["relations"] if not r["definitional"] and not r["degenerate"]]
    return {
        "label": proposal["label_field"], "label_correct": _base_field(proposal["label_field"]) == g_lab,
        "relations": sorted(p_rels), "has_definitional": has_defn,
        "rel_recall": round(len(g_rels_nondefn & p_rels) / max(len(g_rels_nondefn), 1), 2),
        "n_valid_rel": len(valid),
        "build_valid": meas["label"]["valid"] and len(valid) >= 1,
    }


def main(seeds=(1, 2, 3)):
    print("=" * 72)
    print("P6: 关系召回增强(宽松枚举 + LLM curate + verifier) vs P4 基线")
    print("=" * 72)
    print(f"{'域':9s} | {'P4 recall':>10s} -> {'P6 recall':>10s} | {'P4 n_rel':>8s}->{'P6 n_rel':>8s} | P6 relations")
    for dom in ("spotify", "wikiart", "steam", "tmdb"):
        items = json.load(open(ROOT / DATA_FILE[dom]))
        gold = GOLD[dom]
        # P6
        recs = [r for r in (score_vs_gold(autonomous_v2(items, seed=s), gold, items) for s in seeds) if r]
        if not recs:
            print(f"{dom:9s} | (fail)"); continue
        import statistics as st
        p6_recall = st.mean(r["rel_recall"] for r in recs)
        p6_nrel = st.mean(r["n_valid_rel"] for r in recs)
        p6_rels = recs[0]["relations"]
        label_ok = sum(r["label_correct"] for r in recs)
        build_ok = sum(r["build_valid"] and not r["has_definitional"] for r in recs)
        print(f"{dom:9s} | {'0.5-1.0':>10s} -> {p6_recall:>10.2f} | {'1-3':>8s}->{p6_nrel:>8.1f} | {p6_rels}  (label {label_ok}/{len(recs)}, build {build_ok}/{len(recs)})")
    # amazon(无金标准, 报关系数 + valid)
    print(f"\n[amazon] (无金标准, 看关系丰富度)")
    items = json.load(open(ROOT / "data/amazon_raw/products.json"))
    for s in (1, 2):
        prop = autonomous_v2(items, seed=s)
        if prop is None:
            print(f"  seed={s}: fail"); continue
        meas = measure(items, prop)
        rels = [(r["name"], r["group_by"], r["definitional"], r["degenerate"]) for r in meas["relations"]]
        print(f"  seed={s}: label={prop['label_field']} relations={[(n,g) for n,g,_,_ in rels]}")
    print("\n判读:P6 recall 应显著高于 P4(spotify 0.5→1.0 等), 关系数更接近人写。")


if __name__ == "__main__":
    main()
