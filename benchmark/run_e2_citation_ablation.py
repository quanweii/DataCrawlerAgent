#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E2 — arxiv 引用匹配: LLM vs SequenceMatcher(规则) 的诚实消融实验.

背景:
  论文头号贡献是"多智能体构建". 代码侦察证实多智能体不做任何语义建图决策——
  边由 crawlers/ 规则画, agent 只透传. 全仓库唯一真正需要语义判断、当前却是规则的
  环节 = arxiv 引用串 -> 论文 ID 匹配 (agents/reference_graph_builder.py:1117
  find_matching_paper, 用 difflib.SequenceMatcher, 阈值 0.90). OCR 乱码 / 标题变形
  恰好是 LLM 强、规则吃力的场景. 本实验在这里隔离出"语义匹配"决策, 做 LLM-vs-规则
  的公平对照, 作为多智能体价值的唯一诚实证据.

公平性设计 (核心, 防稻草人):
  - 金标准: resolved 边(reference_text ≈ target_title, 近乎逐字相同)是高置信正确答案;
            unresolved 是规则返回 None 的硬引用.
  - 两臂拿到 **同一份** top-K 候选短名单 (倒排索引词重叠检索), 只换"打分/选择"环节:
      * 规则臂 = SequenceMatcher ratio ≥ 0.90 + 关键词重叠门  (逐字复刻生产代码)
      * LLM 臂 = 标准 matching prompt, temp=0, 从 K 个候选里选
  - Part2 的 unresolved 上, 规则生产函数本就返回 None (by construction), 规则臂在
    同一短名单上也够不到 0.90 (否则不会 unresolved); LLM 若仍能正确识别 = 被隔离的
    语义优势.
  - 独立裁判: 严格 significant-word Jaccard + 作者姓重叠 (与 SequenceMatcher 不同算法),
            并对随机样本人工复核标定.

用法:
  python run_e2_citation_ablation.py --smoke            # 小样本冒烟 (~20 引用)
  python run_e2_citation_ablation.py                    # 正式
"""
import os, re, io, csv, json, time, argparse, pickle, random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

# ---- 路径 ----
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META_PATH = os.path.join(ROOT, "arxiv-cs.json")
CSV_PATH = os.path.join(ROOT, "data", "citation_edges_full", "cs_si_citation_edges_full.csv")
PAPERS_DIR = os.path.join(ROOT, "data", "cs_si_papers")
OUT_DIR = os.path.join(ROOT, "benchmark", "results", "e2_citation_ablation")
CACHE = os.path.join(OUT_DIR, "meta_index.pkl")

STOP = {'a','an','the','of','in','for','and','on','to','with','by','at','from',
        'is','are','was','were','be','been','as','or','not','no','but','its','it',
        'this','that','we','our','their','can','will','all','using','based'}

# ============ 标题标准化 (逐字复刻 reference_graph_builder._normalize_title) ============
def normalize_title(title: str) -> str:
    title = title.lower()
    title = re.sub(r'(\w)-\s+([a-z])', r'\1\2', title)   # 修 OCR 断字
    title = ' '.join(title.split())
    title = re.sub(r'[^\w\s]', '', title)
    return title.strip()

# ============ 关键词重叠验证 (逐字复刻 _validate_keyword_overlap) ============
def validate_keyword_overlap(ref: str, cand: str, min_overlap=0.30) -> bool:
    rw = set(ref.split()) - STOP
    cw = set(cand.split()) - STOP
    if not rw: return False
    inter = len(rw & cw)
    return inter / len(rw) >= min_overlap

# ============ 生产规则匹配器 (逐字复刻 find_matching_paper, 阈值0.90) ============
def rule_match_production(ref_title, ref_arxiv_id='',
                          title_to_id=None, arxiv_id_to_canonical=None,
                          word_index=None, threshold=0.90):
    """完整生产逻辑: ID精确->标题精确->倒排候选(top500)->双向子串->SequenceMatcher."""
    if ref_arxiv_id:
        can = arxiv_id_to_canonical.get(ref_arxiv_id)
        if can: return can
    nref = normalize_title(ref_title)
    if len(nref) < 10: return None
    if nref in title_to_id: return title_to_id[nref]
    sig = {w for w in (set(nref.split()) - STOP) if len(w) >= 3}
    if len(sig) < 2: return None
    cand_titles = defaultdict(int)
    for w in sig:
        if w in word_index:
            for t in word_index[w]:
                cand_titles[t] += 1
    cands = []
    for t, ov in cand_titles.items():
        if ov >= 2 and abs(len(t)-len(nref)) <= len(nref)*0.5:
            cands.append((t, title_to_id[t], ov))
    cands.sort(key=lambda x: x[2], reverse=True)
    best, best_s = None, 0
    for t, aid, _ in cands[:500]:
        if len(t) > 10 and len(nref) > 10:
            if nref in t or t in nref:
                ratio = min(len(nref),len(t))/max(len(nref),len(t))
                if ratio > 0.75 and validate_keyword_overlap(nref, t, 0.35):
                    return aid
        s = SequenceMatcher(None, nref, t).ratio()
        if s > best_s and s >= threshold and validate_keyword_overlap(nref, t, 0.30):
            best_s, best = s, aid
            if s >= 0.95: return best
    return best

# ============ 规则臂: 在给定短名单上打分 (隔离打分环节) ============
def rule_match_shortlist(ref_title, candidates, threshold=0.90):
    """candidates: list[(id,title,authors,score)]; 返回 SequenceMatcher 最高且>=阈值的 id."""
    nref = normalize_title(ref_title)
    if len(nref) < 10: return None, 0.0
    best, best_s = None, 0.0
    for aid, title, _au, _sc in candidates:
        nt = normalize_title(title)
        s = SequenceMatcher(None, nref, nt).ratio()
        if s > best_s and s >= threshold and validate_keyword_overlap(nref, nt, 0.30):
            best_s, best = s, aid
    return best, best_s

# ============ 倒排候选检索 (复刻检索逻辑, top-K) ============
def retrieve_candidates(ref_title, word_index, title_to_id, id_to_title, id_to_authors, K=20):
    nref = normalize_title(ref_title)
    sig = {w for w in (set(nref.split()) - STOP) if len(w) >= 3}
    cand_titles = defaultdict(int)
    for w in sig:
        if w in word_index:
            for t in word_index[w]:
                cand_titles[t] += 1
    out = []
    for t, ov in cand_titles.items():
        if ov >= 1 and abs(len(t)-len(nref)) <= len(nref)*0.8:
            aid = title_to_id.get(t)
            if aid:
                out.append((aid, id_to_title.get(aid,t), id_to_authors.get(aid,''), ov))
    out.sort(key=lambda x: x[3], reverse=True)
    return out[:K]

# ============ 独立裁判 (与 SequenceMatcher 不同算法: 词重叠+作者) ============
def judge_correct(ref_text, proposed_id, id_to_title, id_to_authors, jaccard_min=0.55):
    """判定 proposed_id 是否真是 ref_text 指向的论文. 用 significant-word Jaccard + 作者姓.
    返回 (correct: bool, jaccard: float). 保守 (宁可漏判)."""
    if proposed_id is None: return False, 0.0
    nref = normalize_title(ref_text)
    nt = normalize_title(id_to_title.get(proposed_id, ''))
    if not nref or not nt: return False, 0.0
    rw = {w for w in (set(nref.split()) - STOP) if len(w) >= 3}
    cw = {w for w in (set(nt.split()) - STOP) if len(w) >= 3}
    if not rw or not cw: return False, 0.0
    jac = len(rw & cw) / len(rw | cw)
    # 子串包含也算 (标题被截断的情况)
    contain = (nt in nref) or (nref in nt)
    ok = (jac >= jaccard_min) or (contain and jac >= 0.35)
    return ok, jac

# ============ LLM 臂 ============
def make_llm_client():
    from openai import OpenAI
    return OpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("MODEL_BASE_URL"))

_LLM_FAIL = {'n':0, 'reasons':defaultdict(int)}

def llm_match(client, model, ref_text, candidates, timeout=90, max_retries=4):
    """标准 matching prompt, temp=0. candidates: list[(id,title,authors,score)]. 返回 id|None.
    mimo-v2-flash 是推理模型: 思维链走 reasoning_content (单独计 token, 不占 max_tokens),
    真答案在 content. 保留推理 (关掉会退化成浅层关键词匹配, 见实验日志). 并发下 429/超时
    用指数退避重试; 失败计数."""
    if not candidates: return None
    lines = []
    for i, (aid, title, au, _sc) in enumerate(candidates, 1):
        au_short = (au[:60] if au else '')
        lines.append(f"{i}. {aid} | {title[:140]} | {au_short}")
    cand_block = "\n".join(lines)
    sys = ("You match academic citations to arXiv papers. You get a citation string "
           "(extracted from a reference list; may have OCR errors, missing spaces, merged "
           "words, author names) and a numbered list of candidate arXiv papers "
           "(id | title | authors). Decide which candidate (if any) is the cited paper. "
           "Match on SEMANTIC title equivalence tolerating OCR noise, word-order, and "
           "partial author info. Do NOT match on shallow keyword overlap alone. "
           "Reply with ONLY the candidate number (1-%d) or NONE. No explanation." % len(candidates))
    usr = f'Citation: "{ref_text[:300]}"\n\nCandidates:\n{cand_block}'
    msgs = [{"role":"system","content":sys},{"role":"user","content":usr}]
    txt = ''
    for max_tok in (3072, 6144):
        for attempt in range(max_retries):
            try:
                r = client.chat.completions.create(
                    model=model, temperature=0, max_tokens=max_tok, timeout=timeout, messages=msgs)
                ch = r.choices[0]
                txt = (ch.message.content or "").strip().lower()
                fr = ch.finish_reason
                break
            except Exception as e:
                _LLM_FAIL['n'] += 1
                _LLM_FAIL['reasons'][type(e).__name__] += 1
                if attempt < max_retries - 1:
                    time.sleep(5 * (2 ** attempt))   # 5,10,20,40s 退避 (端点并发限流)
                else:
                    txt, fr = '', 'error'
        if txt:
            break                       # 拿到答案
        if fr != 'length':
            break                       # 非截断的空 -> 放弃
        # length 截断 -> 加大 max_tok 再试
    if not txt:
        return None
    ints = re.findall(r'\d+', txt)       # 取第一个整数; NONE/空 -> 无整数 -> None
    if not ints:
        return None
    idx = int(ints[0]) - 1
    if 0 <= idx < len(candidates):
        return candidates[idx][0]
    return None

# ============ 元数据加载 (带 pickle 缓存) ============
def load_metadata(path, limit=None, rebuild_cache=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    if (not rebuild_cache) and os.path.exists(CACHE) and (limit is None):
        print(f"[meta] 从缓存加载 {CACHE}")
        with open(CACHE,'rb') as f: return pickle.load(f)
    title_to_id={}; id_to_title={}; id_to_authors={}; arxiv_id_to_canonical={}
    print(f"[meta] 加载 {path} (limit={limit}) ...")
    t0=time.time(); n=0
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            if limit and n>=limit: break
            try:
                p = json.loads(line)
            except Exception:
                continue
            title = (p.get('title') or '').strip().lower()
            aid = (p.get('id') or '').strip()
            if not (title and aid): continue
            nt = normalize_title(title)
            title_to_id[nt] = aid
            id_to_title[aid] = title
            arxiv_id_to_canonical[aid] = aid
            bid = aid.split('v')[0]
            if bid != aid: arxiv_id_to_canonical[bid] = aid
            # 作者: 优先 authors_parsed, 否则 authors 字符串
            au = ''
            ap = p.get('authors_parsed')
            if ap:
                try:
                    if ap and isinstance(ap[0], (list,tuple)):
                        au = '; '.join(' '.join(x for x in a if x) for a in ap[:8])
                    else:
                        au = '; '.join(str(x) for x in ap[:8])
                except Exception:
                    au = ''
            if not au:
                au = (p.get('authors') or '')[:200]
            id_to_authors[aid] = au
            n += 1
    # 倒排索引
    print(f"[meta] 建倒排索引 ...")
    word_index = defaultdict(set)
    for t in title_to_id:
        for w in ({w for w in (set(t.split())-STOP) if len(w)>=3}):
            word_index[w].add(t)
    print(f"[meta] {n} 篇, 耗时 {time.time()-t0:.1f}s")
    d = dict(title_to_id=title_to_id, id_to_title=id_to_title, id_to_authors=id_to_authors,
             arxiv_id_to_canonical=arxiv_id_to_canonical, word_index=dict(word_index))
    if limit is None:
        with open(CACHE,'wb') as f: pickle.dump(d, f)
    return d

# ============ 解析论文 References -> 引用列表 (复用 agent 解析器) ============
def get_agent_parser():
    """实例化 ReferenceGraphBuilderAgent (client=None) 仅用其解析方法."""
    import sys
    sys.path.insert(0, ROOT)
    from agents.reference_graph_builder import ReferenceGraphBuilderAgent
    return ReferenceGraphBuilderAgent(client=None, system_prompt="")

def parse_paper_refs(agent, md_path):
    try:
        md = open(md_path, encoding='utf-8', errors='replace').read()
    except Exception:
        return []
    sec = agent.extract_references_section(md)
    if not sec: return []
    try:
        refs = agent.parse_references(sec)
    except Exception:
        return []
    out = []
    for r in refs:
        t = (r.get('title') or '').strip()
        if len(t) >= 10:
            out.append({'title': t, 'arxiv_id': (r.get('arxiv_id') or '').strip(),
                        'author': (r.get('author') or '').strip()})
    return out

# ============ main ============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--meta-limit', type=int, default=None)
    ap.add_argument('--k', type=int, default=20, help='候选短名单大小')
    ap.add_argument('--part1-n', type=int, default=400)
    ap.add_argument('--parse-papers', type=int, default=400, help='Part2 解析论文数')
    ap.add_argument('--part2-cap', type=int, default=1500, help='Part2 送 LLM 的 unresolved 上限')
    ap.add_argument('--skip-part2', action='store_true', help='只跑 Part1 (放大 Part1 用)')
    ap.add_argument('--llm-workers', type=int, default=3)
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()
    if args.smoke:
        args.meta_limit = 80000; args.part1_n = 20; args.parse_papers = 8; args.part2_cap = 40
    random.seed(args.seed)
    os.makedirs(OUT_DIR, exist_ok=True)

    model = os.getenv("MODEL_NAME", "mimo-v2-flash")
    meta = load_metadata(META_PATH, limit=args.meta_limit)
    t2i = meta['title_to_id']; i2t = meta['id_to_title']; i2au = meta['id_to_authors']
    a2c = meta['arxiv_id_to_canonical']; wi = meta['word_index']
    print(f"[meta] 池大小 {len(t2i)}")

    client = make_llm_client()

    # ============ Part 1: resolved 金标准 (无伤害 + 难度桶) ============
    print("\n===== Part1: resolved 金标准 =====")
    raw = open(CSV_PATH,'rb').read().replace(b'\x00',b'').decode('utf-8','replace')
    rows = list(csv.DictReader(io.StringIO(raw)))
    # 只要 target_id 在池里的 (保证金标准可检索)
    gold_rows = [r for r in rows if r.get('target_id') and r.get('target_id') in i2t]
    random.shuffle(gold_rows)
    p1 = gold_rows[:args.part1_n]
    print(f"  resolved 总 {len(rows)}, 在池内 {len(gold_rows)}, 采样 {len(p1)}")

    def difficulty_bucket(ref_text, gold_title):
        s = SequenceMatcher(None, normalize_title(ref_text), normalize_title(gold_title)).ratio()
        if s >= 0.95: return 'clean(>=0.95)'
        if s >= 0.90: return 'medium(0.90-0.95)'
        return 'fuzzy(<0.90)'

    p1_rows=[]
    for r in p1:
        ref = r.get('reference_text','') or ''
        gold = r['target_id']; gtitle = i2t.get(gold,'')
        cands = retrieve_candidates(ref, wi, t2i, i2t, i2au, args.k)
        gold_in = gold in {c[0] for c in cands}
        rule_prod = rule_match_production(ref, '', t2i, a2c, wi, 0.90)
        rule_sl, _ = rule_match_shortlist(ref, cands, 0.90)
        p1_rows.append(dict(ref=ref[:160], gold=gold, gold_title=gtitle[:80],
                            bucket=difficulty_bucket(ref,gtitle), gold_in_candidates=gold_in,
                            rule_production=rule_prod, rule_prod_correct=(rule_prod==gold),
                            rule_shortlist=rule_sl, rule_sl_correct=(rule_sl==gold)))
    # LLM 臂 (并发)
    def run_llm(item):
        cands = retrieve_candidates(item['ref'], wi, t2i, i2t, i2au, args.k)
        return llm_match(client, model, item['ref'], cands)
    with ThreadPoolExecutor(max_workers=args.llm_workers) as ex:
        fmap = {ex.submit(run_llm, r): i for i,r in enumerate(p1_rows)}
        for fut in as_completed(fmap):
            i = fmap[fut]; p1_rows[i]['llm'] = fut.result()
            p1_rows[i]['llm_correct'] = (p1_rows[i]['llm']==p1_rows[i]['gold'])

    def agg(rows, key, predkey):
        by=defaultdict(lambda:[0,0])
        for r in rows:
            by[r['bucket']][0]+=1
            if r[predkey]: by[r['bucket']][1]+=1
        return {b:{'n':v[0],'correct':v[1],'recall':round(v[1]/v[0],3) if v[0] else None} for b,v in by.items()}
    p1_summary = dict(
        rule_production=agg(p1_rows,'bucket','rule_prod_correct'),
        rule_shortlist=agg(p1_rows,'bucket','rule_sl_correct'),
        llm=agg(p1_rows,'bucket','llm_correct'),
        gold_retrievable_rate=round(sum(r['gold_in_candidates'] for r in p1_rows)/len(p1_rows),3),
    )
    print("  Part1 recall by bucket:")
    for b in ['clean(>=0.95)','medium(0.90-0.95)','fuzzy(<0.90)']:
        rp=p1_summary['rule_production'].get(b,{}); rl=p1_summary['rule_shortlist'].get(b,{}); ll=p1_summary['llm'].get(b,{})
        print(f"    {b:18s} rule_prod={rp.get('recall')} rule_sl={rl.get('recall')} llm={ll.get('recall')} (n={rp.get('n')})")

    if args.skip_part2:
        out=dict(config=vars(args), model=model, meta_pool_size=len(t2i),
                 part1=p1_summary, part2={'skipped':True}, part1_rows=p1_rows)
        outf=os.path.join(OUT_DIR,'e2_part1_only.json')
        json.dump(out, open(outf,'w',encoding='utf-8'), ensure_ascii=False, indent=1)
        print(f"\n[done] --skip-part2, 保存 Part1(n={len(p1_rows)}) -> {outf}")
        return

    # ============ Part 2: unresolved 回收 (杀手结果) ============
    print("\n===== Part2: unresolved 回收 =====")
    agent = get_agent_parser()
    paper_ids = sorted(os.listdir(PAPERS_DIR))
    random.shuffle(paper_ids)
    unresolved = []   # list of dict(ref, from_paper)
    seen = set()
    for pid in paper_ids[:args.parse_papers*2]:
        if len(unresolved) >= 3000: break
        md = os.path.join(PAPERS_DIR, pid, f"{pid}.md")
        if not os.path.exists(md): continue
        refs = parse_paper_refs(agent, md)
        for r in refs:
            key = normalize_title(r['title'])
            if key in seen: continue
            seen.add(key)
            # 规则生产匹配
            m = rule_match_production(r['title'], r['arxiv_id'], t2i, a2c, wi, 0.90)
            if m is None:   # unresolved
                unresolved.append({'ref': r['title'], 'from': pid,
                                   'arxiv_id_hint': r['arxiv_id']})
        if len([p for p in paper_ids[:paper_ids.index(pid)+1] ]) and paper_ids.index(pid) >= args.parse_papers:
            # 已解析够多篇
            if len(unresolved) >= 500: break
    print(f"  解析得到 unresolved 引用: {len(unresolved)}")
    # 限制 LLM 调用量
    unresolved = unresolved[:args.part2_cap]

    # 给每个 unresolved 检索候选; 只保留有候选的
    p2_items=[]
    for u in unresolved:
        cands = retrieve_candidates(u['ref'], wi, t2i, i2t, i2au, args.k)
        if cands:
            p2_items.append(dict(ref=u['ref'], from_=u['from'], candidates=cands))
    print(f"  有候选的 unresolved: {len(p2_items)} (将送 LLM)")

    def process_one(it):
        cands = it['candidates']
        rule_sl, rule_score = rule_match_shortlist(it['ref'], cands, 0.90)
        ll = llm_match(client, model, it['ref'], cands)
        rule_judge, _ = judge_correct(it['ref'], rule_sl, i2t, i2au)
        ll_judge, ll_jac = judge_correct(it['ref'], ll, i2t, i2au)
        return dict(ref=it['ref'][:160], from_=it['from_'],
                    rule_shortlist=rule_sl, rule_sl_score=round(rule_score,3),
                    rule_correct=rule_judge,
                    llm=ll, llm_correct=ll_judge, llm_jaccard=round(ll_jac,3))
    p2_rows=[]
    with ThreadPoolExecutor(max_workers=args.llm_workers) as ex:
        fmap={ex.submit(process_one, it): i for i,it in enumerate(p2_items)}
        done=0
        for fut in as_completed(fmap):
            p2_rows.append(fut.result()); done+=1
            if done%50==0: print(f"    ...{done}/{len(p2_items)}")

    n=len(p2_rows)
    rule_n=sum(r['rule_correct'] for r in p2_rows)
    llm_proposed=sum(r['llm'] is not None for r in p2_rows)
    llm_corr=sum(r['llm_correct'] for r in p2_rows)
    p2_summary=dict(
        n_unresolved_with_candidates=n,
        rule_shortlist_correct=rule_n, rule_rate=round(rule_n/n,3) if n else None,
        llm_proposed=llm_proposed, llm_proposed_rate=round(llm_proposed/n,3) if n else None,
        llm_correct=llm_corr, llm_recovery_rate=round(llm_corr/n,3) if n else None,
        llm_precision=round(llm_corr/llm_proposed,3) if llm_proposed else None,
        net_new_edges_vs_rule=llm_corr-rule_n,
    )
    print("  Part2 结果:")
    for k,v in p2_summary.items(): print(f"    {k}: {v}")
    print(f"  [LLM 失败重试统计] 总失败={_LLM_FAIL['n']} 原因={dict(_LLM_FAIL['reasons'])}")

    # ============ 保存 ============
    out=dict(config=vars(args), model=model, meta_pool_size=len(t2i),
             part1=p1_summary, part2=p2_summary,
             part2_rows=p2_rows[:300],   # 留样本供人工复核
             part1_rows=p1_rows[:80])
    outf=os.path.join(OUT_DIR,'e2_results.json')
    json.dump(out, open(outf,'w',encoding='utf-8'), ensure_ascii=False, indent=1)
    # 也存一份 LLM 提议样本 csv 供人工核验裁判
    with open(os.path.join(OUT_DIR,'e2_llm_proposals_sample.csv'),'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['ref','llm_proposed_id','llm_proposed_title','judge_correct','jaccard'])
        for r in p2_rows:
            if r['llm']:
                w.writerow([r['ref'], r['llm'], (i2t.get(r['llm'],'') or '')[:90],
                            r['llm_correct'], r['llm_jaccard']])
    print(f"\n[done] 保存 {outf}")

if __name__=='__main__':
    main()
