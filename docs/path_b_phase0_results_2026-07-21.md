# 路径 B · Phase-0 证伪门结果 (2026-07-21)

> 三探针决定**路径 B 是否值得投入**。门规则:≥2 柱干净通过→投入;否则退回路径 A。
> 探针脚本:`benchmark/path_b/probe_{p1_schema,p2_citation,p3_audit}.py`。
> 模型 qwen-plus + DashScope text-embedding-v3(1024d)。

## 判定表

| Pillar | 探针 | 核心结果 | 廉价代理? | 判定 |
|---|---|---|---|---|
| **1 schema 发现** | 藏 YAML,LLM 单次推断 schema(4 域×3 seed) | spotify:LLM prec **0.50** vs B3-float 0.29(B2 0.11);都避开定义性泄露 shared_genre。但 wikiart/tmdb **过度去泄露**(丢掉金标准里的 shared_style/shared_genre);steam **2/3 seed 把 price 选成 label** | 部分(float/基数/名字启发式回收大部分,剩 ~1 个语义判断) | **弱正(条件性)** |
| **2 引用对齐** | n=60 同候选短名单打分(rule/emb/LLM) | fuzzy 桶:**emb 0.778 = LLM 0.778**(rule 0.111);全量 emb **0.967** > LLM 0.950 | **完全有**(embedding 全复制 LLM 语义匹配) | **❌ 死** |
| **3 数据审计** | 注 20 条语义脏数据(4 类)进 200 干净 wikiart | LLM:artist_variant/mislabel/field_swap **1.00**,semantic_dup 0.20(跨批漏);**clean FP 0.26**。embedding-dedup & centroid **全 0.00**(干净噪声底 > 注入 dup;style 不可文本质心分离) | **无**(mislabel 需领域知识,field_swap 词袋不变) | **✅ 正(但单次太噪)** |

## 门结论:**路径 B 不达标 → 退回路径 A**

仅 1 弱(P1)+1 条件正(P3),**未到"≥2 柱干净通过"**。与 IGC 翻车教训一致:3 柱里 2 柱(P2 确定、P1 大部分)有廉价确定性/embedding 代理,agent 不唯一赢。

## 银线(诚实记录,P3 是唯一"无廉价代理"区)
- **P3 是唯一 agent 真正不可替代的语义任务**:mislabel(知道画作属于哪派)、field_swap(读字段角色)——embedding 结构上做不到(干净文本噪声底 0.62 > 注入 dup;field_swap 词袋不变)。
- 但**单次 LLM 审计太噪**(clean FP 26%、跨批 dup 漏检)→ 不能直接做贡献点,需 verifier 闭环或检索增强把 FP 压下来。
- P1 spotify 的精度优势(0.50 vs 0.29)真实但薄,且 steam label 错误暴露可靠性问题。

## 数据(权威)
- P1:`benchmark/results/path_b_p1_probe.json`(注:gitignored)
- P2:fuzzy 桶 emb=llm=0.778;`benchmark/results/path_b_p2_probe.json`
- P3:LLM dirty_recall 0.80 / clean_FP 0.26;`benchmark/results/path_b_p3_probe.json`

## 对论文
确认**路径 A**:C1=4 个 leak-profiled MAG 数据集(+config_driven 引擎可复用性),C2=测量驱动去泄露协议(护城河),agent 诚实降级为"我们如何造数据"。P3 语义审计作为未来工作/诚实限制写入,不抬为 C1。

---
*门按设计工作:在投入 2-3 周前用 1.5 天证伪清楚。结论与 IGC 教训一致——agent 在"有廉价代理"的任务上不必要。*
