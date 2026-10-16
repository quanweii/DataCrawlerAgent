#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E2 — Agent-vs-人工 成本量化.
插桩 autonomous_discover: 每次 LLM 调用记 prompt/completion tokens + wall-clock.
报 per-domain: #calls / 总 tokens / 端到端秒 / n_rounds / build_valid / decontam_correct.
诚实: 人工时间是"专家估计"(基于 config 复杂度 + 典型 schema 设计+去泄露调试耗时), 非用户实测.
"""
import sys, json, os, re, time, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark" / "path_b"))
import probe_p4_autonomous_skill as P4
from probe_p1_schema import _load_env, profile_fields, GOLD, DATA_FILE  # noqa

# ---- amazon / reddit 的 raw 数据(无 gold, 只测成本+build_valid) ----
EXTRA_DATA = {
    "amazon": "./data/amazon_raw/products.json",
    "reddit": "./data/reddit_raw/posts.json",
}
# 专家手写 schema(含去泄露调试)估计耗时(分钟) —— 透明估计, 非实测.
# 依据: 现有 config 复杂度(16-38 行)+ 作者经验, 新域需 inspect raw/选 label/枚举关系/查定义性泄露/迭代.
HUMAN_MIN_PER_DOMAIN = 25.0   # 熟练专家下界
HUMAN_MAX_PER_DOMAIN = 90.0   # 新域+去泄露调试上界

COST = {}
def _reset():
    return {"calls": 0, "prompt": 0, "completion": 0, "total": 0, "wall": 0.0}

def _patched_call_llm(system, user, temperature=0.2, max_tokens=2048, seed=0):
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
    pt = getattr(u, "prompt_tokens", 0) or 0
    ct = getattr(u, "completion_tokens", 0) or 0
    COST["calls"] += 1; COST["wall"] += el
    COST["prompt"] += pt; COST["completion"] += ct; COST["total"] += (pt + ct)
    m = re.search(r"\{[\s\S]*\}", txt)
    if m:
        try: return json.loads(m.group(0)), txt
        except json.JSONDecodeError: pass
    return None, txt

# 注入到 P4 命名空间(P4 从 probe_p1_schema import _call_llm)
P4._call_llm = _patched_call_llm

def _load_items(dom):
    p = ROOT / (EXTRA_DATA.get(dom) or DATA_FILE[dom])
    return json.load(open(p))

def run_domain(dom, seeds=(1, 2, 3)):
    items = _load_items(dom)
    rnd = random.Random(1)
    sample = rnd.sample(items, min(300, len(items)))
    prof = profile_fields(sample)
    prof_str = "\n".join(f"- {p['name']} : dtype={p['dtype']}, n_unique={p['n_unique']}, "
                         f"sample={p['sample_values']}" for p in prof)
    runs = []
    for s in seeds:
        global COST; COST = _reset()
        t0 = time.time()
        prop, trace = P4.autonomous_discover(prof_str, items, seed=s)
        wall_total = time.time() - t0
        # build_valid / decontam 直接从 measure() 算(不依赖 gold) —— 修 amazon/reddit 无 gold 的崩溃
        if prop:
            meas = P4.measure(items, prop)
            valid_rels = [r for r in meas["relations"] if not r["definitional"] and not r["degenerate"]]
            build_valid = bool(meas["label"]["valid"] and len(valid_rels) >= 1)
            decontam = (not any(r["definitional"] for r in meas["relations"]))
        else:
            build_valid = decontam = False
        label_correct = None
        if prop and dom in GOLD:   # gold 对比仅对有 gold 的域
            sc = P4.score(prop, GOLD[dom], items)
            label_correct = sc.get("label_correct")
        runs.append({
            "seed": s,
            "n_llm_calls": COST["calls"],
            "prompt_tokens": COST["prompt"], "completion_tokens": COST["completion"],
            "total_tokens": COST["total"],
            "llm_wall_s": round(COST["wall"], 1),
            "wall_total_s": round(wall_total, 1),   # 含 measure/parse 等
            "n_rounds": len(trace) - 1,
            "build_valid": build_valid,
            "decontam_correct": decontam,
            "label_correct": label_correct,
        })
    return runs

def main():
    _load_env()
    model = os.getenv("MODEL_NAME", "qwen-plus")
    domains = ["spotify", "wikiart", "steam", "tmdb", "amazon", "reddit"]
    out = {}
    print(f"[E2] model={model}  human est={HUMAN_MIN_PER_DOMAIN}-{HUMAN_MAX_PER_DOMAIN} min/domain (专家估计)\n")
    for dom in domains:
        try:
            runs = run_domain(dom)
        except Exception as e:
            print(f"  [{dom}] FAIL: {str(e)[:120]}"); continue
        out[dom] = runs
        # 汇总(3 seed 均值)
        nc = sum(r["n_llm_calls"] for r in runs) / len(runs)
        tok = sum(r["total_tokens"] for r in runs) / len(runs)
        wall = sum(r["wall_total_s"] for r in runs) / len(runs)
        bv = sum(1 for r in runs if r["build_valid"])
        dc = sum(1 for r in runs if r["decontam_correct"])
        print(f"  [{dom:8s}] calls={nc:.1f} tokens={tok:.0f} wall={wall:.1f}s "
              f"| build_valid {bv}/{len(runs)} decontam {dc}/{len(runs)}")
    res = {"model": model, "human_est_min_per_domain": [HUMAN_MIN_PER_DOMAIN, HUMAN_MAX_PER_DOMAIN],
           "human_est_basis": "expert estimate from config complexity (16-38 LOC) + schema-design+decontam-debug time; NOT a user study",
           "domains": out}
    (ROOT / "benchmark/results/e2_agent_cost.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print("\nsaved -> benchmark/results/e2_agent_cost.json")

if __name__ == "__main__":
    main()
