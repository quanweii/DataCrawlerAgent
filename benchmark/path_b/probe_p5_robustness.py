#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PATH-B · P5: label 发现稳健性 + 新域泛化(agent 主线的最后证伪)。

Part A —— 多 prompt/profile 变体下 label 发现是否稳健(还是 P4 的 12/12 只是 prompt 运气):
  V1 terse      : name/dtype/n_unique/sample (P4 用的,已知 12/12)
  V2 +examples  : V1 + 6 条完整样例(P1 风格,疑让 steam 选 price)
  V3 +structural: name/dtype/n_unique/连续性/cardinality-ratio (更强结构提示)
  V4 minimal    : name/dtype 只(最少信息)
  每变体 × 4 域 × 2 seed, single-shot 选 label(隔离发现,不跑 loop)。指标=label 对不对。

Part B —— 新真实域(amazon products,本地、脏、非 4 金标准)端到端:
  跑完整自治闭环 → 看 label/main_cat 发现 + same_brand + 避定义性 + 建图 + LP-ceiling。
"""
from __future__ import annotations
import json, os, re, sys, time, random
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark" / "path_b"))
from probe_p4_autonomous_skill import (  # noqa
    measure, deterministic_critique, autonomous_discover, _parse_schema, SYS_PROPOSE)
from probe_p1_schema import _call_llm, profile_fields, _base_field, GOLD, DATA_FILE  # noqa


def _is_float(v):
    try:
        float(v); return True
    except (TypeError, ValueError):
        return False


# ---------- Part A: profile 变体 ----------
def prof_terse(prof, items, sample):
    return "\n".join(f"- {p['name']}: dtype={p['dtype']}, n_unique={p['n_unique']}, "
                     f"sample={p['sample_values'][:3]}" for p in prof)


def prof_examples(prof, items, sample):
    base = prof_terse(prof, items, sample)
    ex = json.dumps(sample[:6], ensure_ascii=False, indent=1)[:2500]
    return base + "\n\nExample records:\n" + ex


def prof_structural(prof, items, sample):
    N = len(sample)
    lines = []
    for p in prof:
        # 连续性
        vals = [it.get(p["name"]) for it in sample if it.get(p["name"]) not in (None, "", [])]
        flat = []
        for v in vals:
            if isinstance(v, list):
                flat.extend(v)
            else:
                flat.append(v)
        fc = sum(_is_float(str(v)) for v in flat[:200]) / max(min(200, len(flat)), 1)
        ratio = p["n_unique"] / max(N, 1)
        kind = "continuous" if fc > 0.5 else ("unique-id" if ratio > 0.9 else "categorical")
        lines.append(f"- {p['name']}: {kind} (n_unique={p['n_unique']}, ratio={ratio:.2f}, "
                     f"frac_float={fc:.2f})")
    return "\n".join(lines)


def prof_minimal(prof, items, sample):
    return "\n".join(f"- {p['name']}: dtype={p['dtype']}" for p in prof)


VARIANTS = {"V1_terse": prof_terse, "V2_examples": prof_examples,
            "V3_structural": prof_structural, "V4_minimal": prof_minimal}


def single_shot_label(profile_str, seed=1):
    user = f"Domain profile:\n{profile_str}\n\nDesign the schema."
    _, raw = _call_llm(SYS_PROPOSE, user, temperature=0.2, max_tokens=1200, seed=seed)
    prop = _parse_schema(raw)
    return (prop or {}).get("label_field"), raw


def part_A():
    print("=" * 72)
    print("Part A: label 发现稳健性 (4 变体 × 4 域 × 2 seed, single-shot)")
    print("=" * 72)
    res = {}
    header = f"{'variant':16s} " + " ".join(f"{d:>9s}" for d in GOLD) + f" {'ALL':>6s}"
    print(header)
    for vname, vfn in VARIANTS.items():
        row = {}
        all_ok = 0
        cells = []
        for dom in GOLD:
            items = json.load(open(ROOT / DATA_FILE[dom]))
            rnd = random.Random(1)
            sample = rnd.sample(items, min(300, len(items)))
            prof = profile_fields(sample)
            ps = vfn(prof, items, sample)
            gold_lab = _base_field(GOLD[dom]["label"])
            ok = 0
            for s in (1, 2):
                lab, _ = single_shot_label(ps, seed=s)
                if lab and _base_field(lab) == gold_lab:
                    ok += 1
            row[dom] = ok
            all_ok += ok
            cells.append(f"{ok}/2")
        res[vname] = row
        print(f"{vname:16s} " + " ".join(f"{c:>9s}" for c in cells) + f" {all_ok:>3d}/8")
    print("\n判读:某变体 steam≠2/2 = 该变体让 label 发现崩(prompt 依赖);全变体全对=稳健。")
    return res


# ---------- Part B: amazon 新域端到端 ----------
AMAZON = ROOT / "data/amazon_raw/products.json"


def build_edges(items, proposal, cap=100):
    """从发现的 schema 直接建无向边(模拟 config_driven),返回 (edge_set, per_rel_stats)。"""
    idf = proposal["id_field"]
    rels = proposal["relations"]
    lab = proposal["label_field"]
    # label 值
    def labval(it):
        return _resolve(it, lab)
    edges = set()
    rel_stats = {}
    for r in rels:
        gb = r["group_by"]
        groups = defaultdict(list)
        for it in items:
            iid = it.get(idf)
            if iid is None:
                continue
            v = it.get(gb)
            vals = v if isinstance(v, list) else [v]
            for x in vals:
                if x not in (None, "", []):
                    groups[x].append(iid)
        n_e = 0
        for g, members in groups.items():
            if len(members) < 2:
                continue
            # cap: 每组最多 cap 个成员参与
            m = members[:cap]
            for i in range(len(m)):
                for j in range(i + 1, len(m)):
                    a, b = sorted([m[i], m[j]])
                    if a != b:
                        edges.add((a, b))
                        n_e += 1
        rel_stats[r["name"]] = n_e
    return edges, rel_stats


def _resolve(it, field_ref):
    if field_ref is None:
        return None
    parts = re.split(r"[\[\]]", field_ref)
    base = parts[0]
    idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
    v = it.get(base)
    if idx is not None and isinstance(v, list):
        return v[idx] if idx < len(v) else None
    return v


def lp_ceiling(items, proposal, edges):
    """结构泄露上限(label propagation 简版): 用边连通同标签的传播准确率。"""
    idf = proposal["id_field"]
    # 节点 + label
    nodes = [it.get(idf) for it in items if it.get(idf) is not None]
    nid2idx = {n: i for i, n in enumerate(nodes)}
    labels = {}
    for it in items:
        iid = it.get(idf)
        if iid is None:
            continue
        lv = _resolve(it, proposal["label_field"])
        if lv not in (None, "", []):
            labels[iid] = str(lv)
    # 过滤掉没 label 的 + label 类太小
    cnt = Counter(labels.values())
    valid_classes = {c for c, n in cnt.items() if n >= 20}
    labeled = {n: l for n, l in labels.items() if l in valid_classes}
    if len(valid_classes) < 2:
        return None
    # 同质性
    e_lab = [(labels.get(a), labels.get(b)) for a, b in edges if a in labels and b in labels]
    homo = sum(1 for a, b in e_lab if a == b) / max(len(e_lab), 1)
    # LP-ceiling 近似: 每个无标签节点取邻居多数类(这里直接用全标签算同质性 + 多数类比)
    maj = cnt.most_common(1)[0][1] / sum(cnt.values())
    return {"n_nodes": len(nodes), "n_labeled": len(labeled), "n_classes": len(valid_classes),
            "n_edges": len(edges), "homophily": round(homo, 3), "majority_frac": round(maj, 3)}


def part_B(seeds=(1, 2, 3)):
    print("\n" + "=" * 72)
    print("Part B: amazon 新真实域端到端 (自治闭环 + 建图 + 同质性)")
    print("=" * 72)
    items = json.load(open(AMAZON))
    rnd = random.Random(1)
    sample = rnd.sample(items, min(300, len(items)))
    prof = profile_fields(sample)
    # 用 V3_structural(最强提示) 作为 skill 的 profile
    prof_str = prof_structural(prof, items, sample)
    print(f"amazon: {len(items)} items; 可用字段含 asin/main_cat/brand/price/category_path")
    print("人评参考: 合理 label=main_cat(8 类≥20), 合理关系=same_brand, 应避 same_main_cat(定义性)\n")
    runs = []
    for s in seeds:
        prop, trace = autonomous_discover(prof_str, items, seed=s)
        if prop is None:
            print(f"  seed={s}: PARSE FAIL"); continue
        meas = measure(items, prop)
        edges, rel_stats = build_edges(items, prop)
        lp = lp_ceiling(items, prop, edges)
        lab = prop["label_field"]
        rels = [r["group_by"] for r in prop["relations"]]
        has_defn = any(_base_field(r) == _base_field(lab) for r in rels)
        valid_rels = [r["name"] for r in meas["relations"] if not r["definitional"] and not r["degenerate"]]
        rec = {"seed": s, "label": lab, "label_valid": meas["label"]["valid"],
               "relations": rels, "has_definitional": has_defn,
               "n_valid_rel": len(valid_rels),
               "build_valid": meas["label"]["valid"] and len(valid_rels) >= 1 and len(edges) > 0,
               "n_edges": len(edges), "rel_stats": rel_stats,
               "lp": lp, "n_rounds": len(trace) - 1,
               "round0_label": (trace[0]["proposal"] or {}).get("label_field") if trace[0]["proposal"] else None}
        runs.append(rec)
        print(f"  seed={s}: round0_label={rec['round0_label']} -> label={lab} (valid={rec['label_valid']})")
        print(f"           relations={rels}  has_defn={has_defn}  build_valid={rec['build_valid']}")
        print(f"           n_edges={rec['n_edges']} rel_stats={rel_stats}")
        if lp:
            print(f"           图: {lp['n_nodes']} 节点 / {lp['n_edges']} 边 / {lp['n_classes']} 类 / "
                  f"homophily={lp['homophily']} / majority={lp['majority_frac']}")
    ok = sum(1 for r in runs if r["build_valid"] and not r["has_definitional"])
    print(f"\n amazon 硬门槛过(build_valid AND 无定义性): {ok}/{len(runs)}")
    return runs


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode in ("a", "both"):
        part_A()
    if mode in ("b", "both"):
        part_B()
