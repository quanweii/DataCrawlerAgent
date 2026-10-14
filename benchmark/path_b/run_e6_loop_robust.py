#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E6 — loop 鲁棒性定式化。证明"single-shot label 发现脆(prompt 变体会挂), 但 verifier 闭环总救回"。
每 (variant × domain × seed) 跑一次 autonomous_discover, 从 trace 同时取:
  - round0(single-shot) label + validity
  - final(loop 后) build_valid + decontam
  - recovered = round0 无效 而 final 合法
报 single-shot 合法率 vs loop 合法率 + 恢复事件数。
"""
import sys, json, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark" / "path_b"))
from probe_p4_autonomous_skill import autonomous_discover, measure
from probe_p5_robustness import VARIANTS
from probe_p1_schema import profile_fields, GOLD, DATA_FILE  # noqa

SEEDS = (1, 2)

def label_valid(items, prop):
    if not prop: return False
    m = measure(items, prop)
    return bool(m["label"]["valid"])

def build_ok(items, prop):
    if not prop: return False
    m = measure(items, prop)
    vr = [r for r in m["relations"] if not r["definitional"] and not r["degenerate"]]
    return bool(m["label"]["valid"] and len(vr) >= 1), (not any(r["definitional"] for r in m["relations"]))

def main():
    res = {"per_run": [], "by_variant": {}}
    for vname, vfn in VARIANTS.items():
        vshot_ok = vloop_ok = vrec = n = 0
        for dom in GOLD:
            items = json.load(open(ROOT / DATA_FILE[dom]))
            rnd = random.Random(1)
            sample = rnd.sample(items, min(300, len(items)))
            prof = profile_fields(sample)
            prof_str = vfn(prof, items, sample)
            for s in SEEDS:
                prop, trace = autonomous_discover(prof_str, items, seed=s)
                r0 = (trace[0]["proposal"] or {}) if trace else None
                r0_valid = label_valid(items, r0)
                fin_bv, fin_dc = build_ok(items, prop)
                recovered = (not r0_valid) and fin_bv
                res["per_run"].append({"variant": vname, "domain": dom, "seed": s,
                    "round0_label": (r0 or {}).get("label_field"), "round0_valid": r0_valid,
                    "final_label": (prop or {}).get("label_field"),
                    "final_build_valid": fin_bv, "final_decontam": fin_dc,
                    "recovered": recovered, "n_rounds": len(trace)-1})
                n += 1; vshot_ok += int(r0_valid); vloop_ok += int(fin_bv and fin_dc); vrec += int(recovered)
                tag = " <- RECOVERED" if recovered else ("" if r0_valid else " [still-invalid]")
                print(f"  {vname:13s} {dom:8s} s{s} | round0={'Y' if r0_valid else 'N'} "
                      f"final_bv={'Y' if fin_bv else 'N'} dc={'Y' if fin_dc else 'N'}{tag}", flush=True)
        res["by_variant"][vname] = {"n": n,
            "single_shot_label_valid": vshot_ok, "loop_build_valid_decontam": vloop_ok,
            "single_shot_rate": round(vshot_ok/n,3), "loop_rate": round(vloop_ok/n,3),
            "recovered_events": vrec}
        print(f"  -> {vname}: single-shot {vshot_ok}/{n} ({vshot_ok/n:.0%}) | "
              f"loop {vloop_ok}/{n} ({vloop_ok/n:.0%}) | recoveries={vrec}\n", flush=True)
    # 总
    tot = len(res["per_run"])
    sh = sum(r["round0_valid"] for r in res["per_run"])
    lp = sum(r["final_build_valid"] and r["final_decontam"] for r in res["per_run"])
    rc = sum(r["recovered"] for r in res["per_run"])
    res["overall"] = {"n": tot, "single_shot_valid": sh, "loop_valid": lp, "recovered": rc,
                      "single_shot_rate": round(sh/tot,3), "loop_rate": round(lp/tot,3)}
    print(f"=== 总: single-shot label 合法 {sh}/{tot} ({sh/tot:.0%}) | "
          f"loop build_valid+decontam {lp}/{tot} ({lp/tot:.0%}) | 恢复事件 {rc} ===")
    (ROOT/"benchmark/results/e6_loop_robust.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print("saved -> benchmark/results/e6_loop_robust.json")

if __name__ == "__main__":
    main()
