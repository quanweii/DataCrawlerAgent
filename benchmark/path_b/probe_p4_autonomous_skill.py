#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PATH-B · P4 决定性测试: 自治 skill 能否无人复现/产出合法 leak-profiled MAG？

自治 agent = LLM 发现 schema(id/label/relations) + 确定性 verifier(测量驱动去泄露),
全程不问人。在 4 域 raw 数据上跑,三栏评分:
  1. 功能有效(硬门槛): label 合法 + ≥1 合法关系 + 图非退化
  2. 去泄露正确(per C2): 终 schema 不含定义性关系(group_by==label)
  3. 金标准复现(透明): id/label/非定义性关系匹配

设计要点(诚实分工):
  - LLM 负责"发现"(无廉价代理: 哪个字段是 label/关系) —— Phase-0 P1 的 single-shot 不可靠
  - 确定性 verifier 负责"去泄露/纠错"(测 cardinality/连续性/定义性) —— 已知有效,不重蹈 IGC
  - 闭环 = LLM 发现 + verifier 测量纠错。问题: 闭环能否修 P1 的错(steam price/过度去泄露)?
"""
from __future__ import annotations
import json, os, re, sys, time, random
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark" / "path_b"))
from probe_p1_schema import _load_env, _call_llm, profile_fields, _base_field, GOLD, DATA_FILE  # noqa

SYS_PROPOSE = ("You are a multimodal-attribute-graph schema designer. Given a data profile of a "
    "new domain, design the graph config. Pick: (1) id_field = unique identifier; (2) label_field "
    "= ONE meaningful categorical field with a SMALL number of balanced classes (the NC target — "
    "NOT unique/continuous); (3) relations = co-attribute edges [{name,group_by,cap}], where "
    "group_by is a field whose SHARED value meaningfully connects items. Do NOT create a relation "
    "whose group_by equals label_field (definitional leak). Output ONLY compact JSON: "
    '{"id_field":"...","label_field":"...","relations":[{"name":"...","group_by":"...","cap":30}]}')

SYS_REVISE = ("You are fixing a graph-schema based on MEASURED problems. The measurements are "
    "deterministic and trustworthy. Address every flagged problem: change a bad label to a "
    "categorical field; drop definitional relations (group_by==label); drop continuous/near-unique "
    "fields that are not meaningful co-attributes. Keep at least one valid relation. Output ONLY "
    "the corrected compact JSON schema (same shape).")


# ---------- 值解析 + 测量 ----------
def _resolve(val, field_ref):
    """genres[0] -> first elem; a.b -> nested; plain -> val."""
    if val is None:
        return None
    parts = re.split(r"[\[\]]", field_ref)
    # field_ref like 'genres[0]' -> ['genres','0','']
    base = parts[0]
    idx = None
    if len(parts) > 1 and parts[1].isdigit():
        idx = int(parts[1])
    v = val.get(base) if isinstance(val, dict) else None
    if idx is not None and isinstance(v, list):
        return v[idx] if idx < len(v) else None
    return v


def _is_float(v):
    try:
        float(v); return True
    except (TypeError, ValueError):
        return False


def measure(items, proposal, min_class_size=20):
    """确定性测量: label 合法性 + 每个关系的 cardinality/连续性/定义性。"""
    idf = proposal.get("id_field"); lab = proposal.get("label_field")
    rels = proposal.get("relations", [])
    lab_base = _base_field(lab) if lab else None
    N = len(items)
    out = {"id_field": idf, "label_field": lab, "label_base": lab_base, "relations": []}

    # label stats
    lab_vals = [_resolve(it, lab) for it in items] if lab else []
    lab_vals = [v for v in lab_vals if v not in (None, "", [])]
    lab_nonempty = [v for v in lab_vals]
    cnt = Counter(map(str, lab_vals))
    n_classes = len(cnt)
    smallest = min(cnt.values()) if cnt else 0
    # pipeline 语义: min_class_size 过滤后保留的类数(长尾小类会被删,不影响 label 合法性)
    classes_meeting_min = sum(1 for c in cnt.values() if c >= min_class_size)
    frac_cont_lab = sum(_is_float(v) for v in lab_nonempty[:200]) / max(min(200, len(lab_nonempty)), 1)
    out["label"] = {"n_classes": n_classes, "classes_meeting_min": classes_meeting_min,
                    "smallest_class": smallest, "frac_continuous": round(frac_cont_lab, 2),
                    "valid": (2 <= classes_meeting_min <= 100 and frac_cont_lab < 0.5)}

    # per relation —— 用"共享度"判合法性(非 n_unique): album_id 值很多但有效分组
    for r in rels:
        gb = r.get("group_by")
        gb_base = _base_field(gb) if gb else None
        groups = {}
        for it in items:
            v = it.get(gb) if gb else None
            vals = v if isinstance(v, list) else [v]
            for x in vals:
                if x not in (None, "", []):
                    groups.setdefault(str(x), 0)
                    groups[str(x)] += 1
        sizes = list(groups.values())
        shared_items = sum(s for s in sizes if s >= 2)   # 落在≥2元组里的 item 次数
        frac_shared = shared_items / max(N, 1)
        # 连续性: 取样值可解析为 float 的比例
        sample_vals = list(groups.keys())[:200]
        frac_cont = sum(_is_float(v) for v in sample_vals) / max(len(sample_vals), 1)
        is_defn = (gb_base is not None and lab_base is not None and gb_base == lab_base)
        degenerate = (frac_cont > 0.5 or frac_shared < 0.3)   # 连续 或 几乎不共享
        out["relations"].append({"name": r.get("name"), "group_by": gb, "group_by_base": gb_base,
            "n_groups": len(groups), "frac_shared": round(frac_shared, 2),
            "frac_continuous": round(frac_cont, 2),
            "definitional": is_defn, "degenerate": degenerate})
    return out


def deterministic_critique(meas):
    """读测量值,确定性产出纠错 flag(不靠 LLM 读测量=不重蹈 IGC)。"""
    flags = []
    if not meas["label"]["valid"]:
        lb = meas["label"]
        if lb["frac_continuous"] >= 0.5:
            flags.append(f"LABEL '{meas['label_field']}' looks CONTINUOUS (frac_float={lb['frac_continuous']}); "
                         f"replace with a categorical field (few balanced classes).")
        elif lb.get("classes_meeting_min", 0) > 100:
            flags.append(f"LABEL '{meas['label_field']}' has {lb['classes_meeting_min']} valid classes (too many); "
                         f"replace with a coarser categorical field.")
        elif lb.get("classes_meeting_min", 0) < 2:
            flags.append(f"LABEL '{meas['label_field']}' has too few classes meeting min size "
                         f"({lb.get('classes_meeting_min',0)}); replace with a more balanced categorical field.")
        else:
            flags.append(f"LABEL '{meas['label_field']}' invalid; replace with a categorical field.")
    for r in meas["relations"]:
        if r["definitional"]:
            flags.append(f"RELATION '{r['name']}' (group_by={r['group_by']}) is DEFINITIONAL "
                         f"(same field as label) — direct label leak. DROP it.")
        elif r["degenerate"]:
            why = ("continuous" if r["frac_continuous"] > 0.5 else
                   f"not a real co-attribute (only {r['frac_shared']} of items share a group)")
            flags.append(f"RELATION '{r['name']}' (group_by={r['group_by']}) is {why}; "
                         f"DROP it.")
    return flags


def _parse_schema(raw):
    if not raw:
        return None
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return None
    try:
        pol = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    rels = []
    for r in pol.get("relations", []):
        gb = r.get("group_by")
        if gb:
            rels.append({"name": r.get("name", f"shared_{gb}"), "group_by": gb, "cap": r.get("cap", 30)})
    return {"id_field": pol.get("id_field"), "label_field": pol.get("label_field"), "relations": rels}


def autonomous_discover(profile_str, items, seed=1, max_rounds=4):
    """LLM 发现 + 确定性 verifier 闭环。返回 (final_proposal, trace)。"""
    user0 = f"Domain profile:\n{profile_str}\n\nDesign the schema."
    _, raw0 = _call_llm(SYS_PROPOSE, user0, temperature=0.2, max_tokens=1200, seed=seed)
    proposal = _parse_schema(raw0)
    trace = [{"round": 0, "proposal": proposal, "flags": None}]
    if proposal is None:
        return None, trace
    for rnd in range(1, max_rounds + 1):
        meas = measure(items, proposal)
        flags = deterministic_critique(meas)
        trace[-1]["measured"] = meas
        trace[-1]["flags"] = flags
        if not flags:
            break
        rev_user = (f"Available fields in the data (pick ONLY from these REAL field names; "
                    f"do NOT invent field names):\n{profile_str}\n\n"
                    f"Current schema: {json.dumps(proposal)}\n\nMeasured problems:\n- " +
                    "\n- ".join(flags) + "\n\nOutput the corrected schema JSON, using ONLY field "
                    f"names from the list above.")
        _, raw = _call_llm(SYS_REVISE, rev_user, temperature=0.2, max_tokens=1200, seed=seed)
        new_prop = _parse_schema(raw)
        trace.append({"round": rnd, "proposal": new_prop, "flags": None})
        if new_prop is None or new_prop == proposal:
            proposal = new_prop or proposal
            break
        proposal = new_prop
    return proposal, trace


def score(proposal, gold, items):
    if proposal is None:
        return {"parse_ok": False}
    meas = measure(items, proposal)
    lab = meas["label"]
    g_lab_base = _base_field(gold["label"])
    g_rels = {gb for _, gb in gold["relations"]}
    # 非定义性金标准关系(排除 group_by==gold_label 的)
    g_rels_nondefn = {gb for _, gb in gold["relations"] if _base_field(gb) != g_lab_base}
    p_rels = {r["group_by_base"] for r in meas["relations"]}
    p_rels_nondefn = {r["group_by_base"] for r in meas["relations"] if not r["definitional"]}
    has_defn = any(r["definitional"] for r in meas["relations"])
    valid_rels = [r for r in meas["relations"] if not r["definitional"] and not r["degenerate"]]
    return {
        "parse_ok": True,
        "id_correct": proposal.get("id_field") == gold["id"],
        "label_correct": _base_field(proposal.get("label_field", "")) == g_lab_base,
        "label_valid": lab["valid"],
        "build_valid": lab["valid"] and len(valid_rels) >= 1,    # 硬门槛
        "decontam_correct": not has_defn,                         # per C2
        "gold_rel_recall_nondefn": round(len(g_rels_nondefn & p_rels) / max(len(g_rels_nondefn), 1), 2),
        "n_valid_relations": len(valid_rels),
        "has_definitional": has_defn,
        "final_label": proposal.get("label_field"),
        "final_rels": sorted(p_rels),
    }


def main(domains=("spotify", "wikiart", "steam", "tmdb"), seeds=(1, 2, 3)):
    summary = {}
    for dom in domains:
        items = json.load(open(ROOT / DATA_FILE[dom]))
        rnd = random.Random(1)
        sample = rnd.sample(items, min(300, len(items)))
        prof = profile_fields(sample)
        prof_str = "\n".join(f"- {p['name']} : dtype={p['dtype']}, n_unique={p['n_unique']}, "
                             f"sample={p['sample_values']}" for p in prof)
        gold = GOLD[dom]
        runs = []
        for s in seeds:
            prop, trace = autonomous_discover(prof_str, items, seed=s)
            sc = score(prop, gold, items)
            runs.append({"seed": s, "score": sc,
                         "n_rounds": len(trace) - 1,
                         "round0_label": (trace[0]["proposal"] or {}).get("label_field") if trace[0]["proposal"] else None})
        summary[dom] = {"gold_label": gold["label"],
                        "gold_rels": sorted({gb for _, gb in gold["relations"]}),
                        "runs": runs}
        # 打印
        print(f"\n{'='*72}\n[{dom}] GOLD label={gold['label']} rels={summary[dom]['gold_rels']}")
        for r in runs:
            sc = r["score"]
            if not sc.get("parse_ok"):
                print(f"  seed={r['seed']}: PARSE FAIL"); continue
            print(f"  seed={r['seed']}: round0_label={r['round0_label']} -> final_label={sc['final_label']} "
                  f"| build_valid={sc['build_valid']} decontam_correct={sc['decontam_correct']} "
                  f"label_correct={sc['label_correct']} gold_rel_recall={sc['gold_rel_recall_nondefn']} "
                  f"n_valid_rel={sc['n_valid_relations']}")
            print(f"           final_rels={sc['final_rels']} has_defn={sc['has_definitional']}")
    # 汇总: 多少 seed 过硬门槛
    print(f"\n{'='*72}\n汇总(硬门槛=build_valid AND decontam_correct):")
    for dom, d in summary.items():
        ok = sum(1 for r in d["runs"] if r["score"].get("build_valid") and r["score"].get("decontam_correct"))
        lc = sum(1 for r in d["runs"] if r["score"].get("label_correct"))
        n = len(d["runs"])
        print(f"  {dom:9s}: 硬门槛过 {ok}/{n}  |  label 复现 {lc}/{n}")
    (ROOT / "benchmark/results/path_b_p4_probe.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nsaved -> benchmark/results/path_b_p4_probe.json")


if __name__ == "__main__":
    main()
