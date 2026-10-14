#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E4 — 自主性扩规模: agent 在 ≥5 个 held-out 域上自治发现 schema。
held-out = 不在 P4 金标准(spotify/wikiart/steam/tmdb)调参集内的真实源。
  amazon / reddit(产品/社交) + arxiv cs/math/cond-mat/astro-ph(论文, JSONL 流式采样)。
报每域: 发现的 label/relations + build_valid + decontam + 成本。复用 E2 成本插桩。
"""
import sys, json, os, re, time, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark" / "path_b"))
import probe_p4_autonomous_skill as P4
from probe_p1_schema import _load_env, profile_fields  # noqa

# held-out 域: (name, path, format)
HELDOUT = [
    ("amazon",      "./data/amazon_raw/products.json", "array"),
    ("reddit",      "./data/reddit_raw/posts.json",    "array"),
    ("arxiv_cs",    "./arxiv-cs.json",                 "jsonl"),
    ("arxiv_math",  "./arxiv-math.json",              "jsonl"),
    ("arxiv_condmat","./arxiv-cond-mat.json",         "jsonl"),
    ("arxiv_astroph","./arxiv-astro-ph.json",         "jsonl"),
]
SAMPLE = 300

COST = {}
def _reset(): return {"calls": 0, "prompt": 0, "completion": 0, "total": 0, "wall": 0.0}

def _patched(system, user, temperature=0.2, max_tokens=2048, seed=0):
    from openai import OpenAI
    _load_env()
    c = OpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("MODEL_BASE_URL"))
    model = os.getenv("MODEL_NAME", "qwen-plus")
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    t0 = time.time()
    try:
        r = c.chat.completions.create(model=model, temperature=temperature,
                                      max_tokens=max_tokens, timeout=90, messages=msgs, seed=seed)
    except Exception:
        time.sleep(4); return None, ""
    el = time.time() - t0
    txt = (r.choices[0].message.content or "").strip()
    u = getattr(r, "usage", None)
    pt = getattr(u, "prompt_tokens", 0) or 0; ct = getattr(u, "completion_tokens", 0) or 0
    COST["calls"] += 1; COST["wall"] += el
    COST["prompt"] += pt; COST["completion"] += ct; COST["total"] += (pt + ct)
    m = re.search(r"\{[\s\S]*\}", txt)
    if m:
        try: return json.loads(m.group(0)), txt
        except json.JSONDecodeError: pass
    return None, txt
P4._call_llm = _patched

def load_items(dom, path, fmt):
    """返回 (profile_sample=300, measure_items=大样本/full)。
    measure 必须在大样本上跑(否则稀有类 <min_class_size=20 误判 invalid, 关系共享度被低估)。"""
    p = ROOT / path
    if fmt == "array":
        items = json.load(open(p))
    else:  # jsonl: 读前 20000 行(够判 label 类数 + 关系共享)
        items = []
        with open(p, encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i >= 20000: break
                line = line.strip()
                if line:
                    try: items.append(json.loads(line))
                    except json.JSONDecodeError: continue
    rnd = random.Random(1)
    prof = rnd.sample(items, min(SAMPLE, len(items)))
    return prof, items, len(items)

def main():
    _load_env()
    model = os.getenv("MODEL_NAME", "qwen-plus")
    res = {}
    print(f"[E4] model={model}, held-out domains: {[d[0] for d in HELDOUT]}\n")
    for dom, path, fmt in HELDOUT:
        try:
            prof_sample, items, total = load_items(dom, path, fmt)
            prof = profile_fields(prof_sample)
            prof_str = "\n".join(f"- {p['name']} : dtype={p['dtype']}, n_unique={p['n_unique']}, "
                                 f"sample={p['sample_values']}" for p in prof)
            global COST; COST = _reset()
            # autonomous_discover 内部 measure 在 items 上跑(大样本, 正确); profile 只喂 300 给 LLM
            prop, trace = P4.autonomous_discover(prof_str, items, seed=1)
            if prop:
                meas = P4.measure(items, prop)
                valid_rels = [r for r in meas["relations"] if not r["definitional"] and not r["degenerate"]]
                build_valid = bool(meas["label"]["valid"] and len(valid_rels) >= 1)
                decontam = not any(r["definitional"] for r in meas["relations"])
            else:
                build_valid = decontam = False
            res[dom] = {"n_total": total,
                        "discovered_label": (prop or {}).get("label_field"),
                        "discovered_relations": (prop or {}).get("relations"),
                        "n_rounds": len(trace) - 1, "n_llm_calls": COST["calls"],
                        "total_tokens": COST["total"], "wall_s": round(COST["wall"], 1),
                        "build_valid": build_valid, "decontam_correct": decontam,
                        "label_n_classes": (meas["label"]["n_classes"] if prop else None)}
            print(f"  [{dom:14s}] n={total} label={res[dom]['discovered_label']} "
                  f"nrel={len(res[dom]['discovered_relations'] or [])} "
                  f"build_valid={build_valid} decontam={decontam} "
                  f"calls={COST['calls']} tok={COST['total']} {COST['wall']:.1f}s", flush=True)
        except Exception as e:
            res[dom] = {"error": str(e)[:160]}
            print(f"  [{dom:14s}] FAIL: {str(e)[:120]}", flush=True)
    (ROOT / "benchmark/results/e4_scale.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    ok = sum(1 for v in res.values() if v.get("build_valid") and v.get("decontam_correct"))
    print(f"\n[done] {ok}/{len(HELDOUT)} held-out 域 build_valid+decontam 全过 -> benchmark/results/e4_scale.json")

if __name__ == "__main__":
    main()
