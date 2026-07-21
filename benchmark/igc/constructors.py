#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IGC constructors —— 把 spec 翻译成关系子集（真实关系名集合）。

确定性构造器（无 LLM，可 smoke）：
  - naive_keyword : spec token 匹配关系名/gloss
  - schema_rule   : 按 name 正则分桶(same_*→IDENTITY) + 策略表（脚本管线会部署的公平基线）
  - oracle        : 暴搜 2^k 子集，优化可测目标（上界）

LLM 构造器（见文末，依赖 prompts）：
  - single_shot_llm     : 一次 planner 调用（原"agent"，现降为消融）
  - closed_loop_multi_agent : planner↔critic 闭环（头条）

所有构造器返回 set[真实关系名]（盲模式下用 bmap 反映射回真实名）。
"""
from __future__ import annotations
import re
from itertools import chain, combinations

from benchmark.igc.graph_subset import measure_subset, measure_leakage, per_rel_lp_profile


def _powerset_nonempty(rels):
    s = list(rels)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


def _presented_to_real(names, bmap):
    """presented 名(可能盲) -> 真实关系名集合。"""
    if not bmap:
        return set(names)
    inv = {v: k for k, v in bmap.items()}
    return {inv.get(n, n) for n in names}


# ============ 1. naive keyword ============

_STOP = set("the a an of to for on and or that any via same shared "
            "item items two relation relations graph build include exclude".split())


def naive_keyword(spec, inventory, bmap):
    text = spec["text"].lower()
    toks = {w for w in re.split(r"\W+", text) if len(w) > 3 and w not in _STOP}
    picked = []
    for item in inventory:
        name_tokens = set(re.split(r"\W+", item["name"].lower())) | \
                      set(re.split(r"\W+", item["gloss"].lower()))
        if toks & name_tokens:
            picked.append(item["name"])
    if not picked:
        picked = [item["name"] for item in inventory]   # no-match → all (safe default)
    return _presented_to_real(picked, bmap)


# ============ 2. schema_rule (公平脚本基线) ============

def _bucket(name):
    """按 name 正则分桶。盲模式下 name=rel_A → 落入 OTHER（无 same_/shared_ 词）。"""
    n = name.lower()
    if n.startswith("same_") or n.endswith("_id") or "identical" in n:
        return "IDENTITY"
    if n.startswith("shared_") or n.startswith("common_") or "co_" in n:
        return "CATEGORICAL"
    return "OTHER"


def schema_rule(spec, inventory, bmap):
    ident = {item["name"] for item in inventory if _bucket(item["name"]) == "IDENTITY"}
    cat = {item["name"] for item in inventory if _bucket(item["name"]) == "CATEGORICAL"}
    alln = [item["name"] for item in inventory]
    sid = spec["id"]
    if sid in ("S1a", "S1b"):
        # 排除 IDENTITY；若 IDENTITY 为空(盲模式)→ 退化成 keep-all（这正是要展示的崩塌）
        kept = [n for n in alln if n not in ident] or alln
    elif sid == "S2":
        kept = list(ident | cat) or alln      # 高同质代理桶；本数据集即 all
    else:  # S5
        kept = alln
    return _presented_to_real(kept, bmap)


# ============ 3. oracle (暴搜上界) ============

def oracle(base, spec, split_seed=1, max_k=None):
    rels = base["all_relations"]
    if spec["id"] == "S5":
        return set(rels)
    target = spec["target"]
    best_set, best_score, best_meas = None, None, None
    candidates = list(_powerset_nonempty(rels))
    if max_k:  # 限制子集大小加速
        candidates = [c for c in candidates if len(c) <= max_k]
    for combo in candidates:
        meas = measure_subset(base, set(combo), split_seed=split_seed)
        if target == "min_lp_ceiling":
            score = -meas["lp_ceiling"]   # 越大越好 → LP-ceiling 越小越好
        elif target == "max_homophily":
            score = meas["homophily"]
        else:
            score = 0
        if best_score is None or score > best_score:
            best_set, best_score, best_meas = set(combo), score, meas
    return best_set


# ============ 4. greedy_removal (确定性测量驱动, 无 LLM) ============

def greedy_removal(base, spec, split_seed=1):
    """【decisive 消融】给脚本管线【同一个 measure_subset 工具】,但不给 LLM。
    每轮移除使 spec 目标最优的关系;终止于不再改善 / 达阈值 / 剩 1 条。

    返回【真实关系名】集合(blind-agnostic —— 它根本不看 bmap/名字,只看测量值)。
    这正是对照要点:closed_loop 的 critic 也只读 per-rel LP-ceiling(见 _critic_user),
    故本贪心与 closed_loop 用【完全相同的信息】。若 greedy_removal ≤ closed_loop
    → 多智能体在小关系数 regime 非必要(测量才是必要的,agent 只是更贵的载体)。
    """
    rels = base["all_relations"]
    if spec["id"] == "S5" or spec["target"] == "all":
        return set(rels)
    target = spec["target"]
    cur = set(rels)
    cur_meas = measure_subset(base, cur, split_seed=split_seed)
    while len(cur) > 1:
        if target == "min_lp_ceiling" and cur_meas["lp_ceiling"] <= cur_meas["majority_acc"] + 0.08:
            break
        if target == "max_homophily" and cur_meas["homophily"] >= 0.85:
            break
        # 试移除每个剩余关系,选使目标最优的(strictly better 才接受)
        best_r, best_meas = None, None
        for r in sorted(cur):
            m = measure_subset(base, cur - {r}, split_seed=split_seed)
            if target == "min_lp_ceiling":
                better = best_meas is None or m["lp_ceiling"] < best_meas["lp_ceiling"] - 1e-9
            else:  # max_homophily
                better = best_meas is None or m["homophily"] > best_meas["homophily"] + 1e-9
            if better:
                best_r, best_meas = r, m
        if best_meas is None:
            break  # 没有任何移除能改善 → 停(no-op,与 closed_loop 终止语义一致)
        cur = cur - {best_r}
        cur_meas = best_meas
    return cur


# ============ LLM 构造器（依赖 prompts/igc_*.txt）============

def _make_client():
    import os
    from openai import OpenAI
    _load_env()
    return OpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("MODEL_BASE_URL"))


def _load_env():
    import os
    from pathlib import Path
    envf = Path(__file__).resolve().parents[2] / ".env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _call_llm(client, model, system, user, temperature=0.2, max_tokens=2048, timeout=90):
    """调用 + JSON 解析。reasoning 模型可能把预算吃在思维链上 → finish_reason=length 截断 JSON，
    故用 max_tokens 阶梯(2048→4096→8192)重试。返回 (parsed_json|None, raw_text)。"""
    import re, json, time
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for max_tok in (max_tokens, 4096, 8192):
        for attempt in range(3):
            try:
                r = client.chat.completions.create(
                    model=model, temperature=temperature, max_tokens=max_tok,
                    timeout=timeout, messages=msgs)
                ch = r.choices[0]
                txt = (ch.message.content or "").strip()
                fr = ch.finish_reason
                if txt:
                    m = re.search(r"\{[\s\S]*\}", txt)
                    if m:
                        try:
                            return json.loads(m.group(0)), txt
                        except json.JSONDecodeError:
                            pass  # 截断的 JSON → 升档重试
                if fr != "length":
                    return None, txt          # 非截断的空/坏答 → 放弃此档
                break                        # length 截断 → 升 max_tokens
            except Exception:
                if attempt < 2:
                    time.sleep(4 * (2 ** attempt))
                else:
                    return None, ""
    return None, ""


def _load_prompt(name):
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "prompts" / name
    return p.read_text(encoding="utf-8")


def single_shot_llm(base, spec, inventory, bmap, client=None, model=None, seed=0):
    """一次 planner 调用 → 关系子集（消融：无 verifier）。"""
    import os
    _load_env()
    client = client or _make_client()
    model = model or os.getenv("MODEL_NAME", "mimo-v2-flash")
    sys_p = _load_prompt("igc_planner_prompt.txt")
    user = _planner_user(spec, inventory, critique=None)
    policy, raw = _call_llm(client, model, sys_p, user)
    picked = _parse_policy(policy, inventory)
    return _presented_to_real(picked, bmap), [{"round": 0, "raw": raw, "policy": policy}]


def closed_loop_multi_agent(base, spec, inventory, bmap, client=None, model=None,
                            seed=0, max_rounds=4, split_seed=1):
    """planner↔critic 闭环：终止于 critic 满意 或 max_rounds。

    executor 测量当前 G(S)（LP-ceiling / per-rel homo），critic 读测量值决定
    哪些关系可疑，planner 据此修订。盲模式下这是【唯一】能定位泄露关系的途径。
    """
    import os
    client = client or _make_client()
    model = os.getenv("MODEL_NAME", "mimo-v2-flash")
    planner_sys = _load_prompt("igc_planner_prompt.txt")
    critic_sys = _load_prompt("igc_critic_prompt.txt")

    trace = []
    critique = None
    prev_real = None
    # 域级泄露画像一次算好（与子集无关），每轮复用 → 省掉 per-rel LP 重算
    prlp = per_rel_lp_profile(base, split_seed=split_seed)
    for rnd in range(max_rounds):
        user = _planner_user(spec, inventory, critique)
        policy, raw = _call_llm(client, model, planner_sys, user)
        picked = _parse_policy(policy, inventory)
        real_set = _presented_to_real(picked, bmap)
        meas = measure_leakage(base, real_set, split_seed=split_seed, prlp=prlp)
        # 防退化：空图(LP=majority)是"删光"作弊 → 保留单独 LP 最低的关系
        prlp = meas.get("per_rel_lp_ceiling", {})
        if not real_set and prlp:
            keep = min(base["all_relations"], key=lambda r: prlp.get(r, 1.0))
            real_set = {keep}
            picked = [bmap.get(keep, keep)] if bmap else [keep]
            meas = measure_leakage(base, real_set, split_seed=split_seed)
        # 确定性终止判据（不依赖 critic 的 satisfied，避免误判 structural 提前退出）
        target = spec["target"]
        if target == "min_lp_ceiling":
            done = meas["lp_ceiling"] <= meas["majority_acc"] + 0.08
        elif target == "max_homophily":
            done = meas["homophily"] >= 0.85
        else:
            done = True
        converged = (prev_real is not None and real_set == prev_real)
        # critic 读测量值，给出可疑关系（多智能体：第二个 LLM 角色；用于指导下轮修订）
        cu = _critic_user(spec, meas, picked)
        critique, craw = _call_llm(client, model, critic_sys, cu)
        crit_satisfied = bool(critique.get("satisfied", False)) if critique else False
        trace.append({"round": rnd, "policy": picked, "raw_planner": raw,
                      "measured": {k: meas.get(k) for k in ("lp_ceiling", "homophily",
                                                           "n_edges", "per_rel_homo",
                                                           "per_rel_lp_ceiling")},
                      "critique": critique,
                      "done_threshold": done, "converged": converged,
                      "critic_satisfied": crit_satisfied})
        prev_real = set(real_set)
        # 终止：达标 / 策略不再变化(收敛) / 最大轮次
        if done or converged:
            break
    return real_set, trace


# ---- prompt 组装 ----

def _planner_user(spec, inventory, critique):
    inv_str = "\n".join(
        f"- {it['name']} : {it['gloss']} (edges: {it['edge_count']})" for it in inventory)
    crit_str = ""
    if critique:
        sus = critique.get("suspect_relations", [])
        reason = critique.get("critique", "")
        sus_str = ", ".join(sus) if sus else "（无）"
        crit_str = (f"\n\n【verifier 测量反馈】泄露画像显示【单独 LP-ceiling 最高的关系】: {sus_str}\n"
                    f"反馈: {reason}\n"
                    f"请据此修订：禁用上述高泄露关系，保留其余（至少保留 1 条关系，不要全删）。"
                    f"即使整体仍偏高，也要先移除最差的关系——下一轮会重新测量确认。")
    else:
        crit_str = ("\n\n【首轮】尚无 verifier 测量。请【启用全部关系】作为初始图，"
                    "以便 verifier 测量每个关系的泄露画像；后续轮次再据反馈移除泄露关系。"
                    "不要在没有任何测量证据时禁用关系。")
    return (f"【构建 spec】\n{spec['text']}\n\n"
            f"【可选关系类型】\n{inv_str}{crit_str}\n\n"
            f"严格只输出一个【紧凑】JSON（不要解释、reason 放思维链里即可）：\n"
            f'{{"enabled": {{"<关系名>": true/false, ...}}}}  '
            f"—— 把每个关系名映射到是否启用。")


def _critic_user(spec, meas, picked):
    import json as _json
    prlp = meas.get("per_rel_lp_ceiling", {})
    prlp_str = ", ".join(f"{k}={v:.3f}" for k, v in prlp.items())
    return (f"【spec】{spec['text']}\n\n"
            f"【planner 当前选择的关系】{picked}\n"
            f"【executor 测量(原始构造图)】当前图 LP-ceiling={meas['lp_ceiling']:.3f} "
            f"(多数类基线={meas['majority_acc']:.3f}), 同质性={meas['homophily']:.3f}, 边数={meas['n_edges']}\n"
            f"【泄露画像: 每个关系单独成图的 LP-ceiling】{prlp_str}\n"
            f"  (LP-ceiling 越高 = 该关系越泄露标签；接近多数类基线 = 不泄露)\n\n"
            f"判断: 若 spec 要去泄露——LP-ceiling 应接近多数类基线；高于基线说明仍有泄露，"
            f"应移除【单独 LP-ceiling 最高的那些】关系。若所有关系单独 LP-ceiling 都高，"
            f"则泄露是结构性的、关系选择无法消除，应 satisfied=true 并在 critique 注明。"
            f"严格以 JSON 输出: {{\"satisfied\": true/false, "
            f"\"suspect_relations\": [\"<关系名>\"], \"critique\": \"<理由>\"}}。")


def _parse_policy(policy, inventory):
    """从 planner JSON 提取 enabled 关系名。支持紧凑 {"enabled":{name:bool}} 与
    列表 {"relations":[{name,enabled}]} 两种格式。容错：无法解析 → 全选。"""
    valid = [it["name"] for it in inventory]
    vset = set(valid)
    if not policy:
        return valid
    # 紧凑格式
    en = policy.get("enabled")
    if isinstance(en, dict):
        picked = [n for n, on in en.items() if on is not False and n in vset]
        return picked or valid
    # 列表格式（兼容）
    rels = policy.get("relations", [])
    picked = [r.get("name") for r in rels if r.get("enabled", True) is not False and r.get("name")]
    picked = [p for p in picked if p in vset]
    return picked or valid
