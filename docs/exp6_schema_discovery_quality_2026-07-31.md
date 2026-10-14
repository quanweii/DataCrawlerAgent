# Exp6 — Agent schema 发现质量评估（2026-07-31, P1）

> 核心问题：agent 自治发现的 label 不仅"合法(valid)"，是否"好(good benchmark)"？
> 结论：**validity 门 6/6 全过，但 quality 门只 ~2/6** → 报告卡质量门（Exp2）是必要补充；label 发现是 agent 的脆弱环节（e6: 88–94% vs 去泄露 100%）。
> 数据源（post-hoc，无需新跑 LLM）：`benchmark/results/e4_scale.json`（held-out 域 discovered schema）+ `e6_loop_robust.json`（loop 鲁棒性）+ arxiv-cs.json license 分布。

## 方法
post-hoc 审计 E4 的 6 个 held-out 域已发现的 schema（每域 discovered_label + label_n_classes + build_valid + decontam_correct 已在盘），用**报告卡质量透镜**（cardinality / 语义 / 平衡）逐域判"是不是好 benchmark"，对照 validity 门。

## 1. 发现质量审计表（6 held-out 域）

| held-out 域 | discovered label | n_classes | validity (build_valid+decontam) | **quality 判定** | 原因 |
|---|---|---|---|---|---|
| amazon | main_cat | 20 | ✅ | ✅ **合理** | 平衡、有意义的产品大类 |
| reddit | subreddit | 10 | ✅ | ✅ **合理** | 平衡、有意义的社区分区 |
| arxiv_cs | license | 9 | ✅ | ⚠️ **语义退化** | top-1 **59.7%** / top-2 **86.6%**（arxiv-cs.json 实算）；license 与论文内容无关 → NC signal_margin≈0 |
| arxiv_math | license | 7 | ✅ | ⚠️ **语义退化** | 同 license 退化 |
| arxiv_condmat | categories | **843** | ✅ | ❌ **过细** | 843 类远超 ≤100 阈值（avg 24 节点/类） |
| arxiv_astroph | categories | **275** | ✅ | ❌ **过细** | 275 类同上 |

- **validity 门（agent 当前保证）= 6/6**（build_valid + decontam_correct 全过）。
- **quality 门（报告卡透镜）≈ 2/6**（仅 amazon、reddit 非退化）。
- **gap = 4/6**：2 个过细（categories 843/275 类，竟被 validity 放行 = 当前门的具体漏洞）+ 2 个语义退化（license 与内容无关、长尾不平衡）。

## 2. label 发现 vs 去泄露：哪个是脆弱环节？（e6 复盘）
`e6_loop_robust.json` overall（4 profile × 4 金标准域 × 2 seed = 32 run）：
- **去泄露（decontam）= 32/32 = 100%** —— 任何变体/域/seed，终 schema 从不含定义性关系（C2 确定性测量，无懈可击）。
- **label 发现 = single-shot 28/32 (87.5%) → loop 30/32 (93.8%)**，+4 恢复事件（全在 steam V3/V4）；loop 非严格单调（V3 tmdb −2 回归）。
- **结论**：可靠核心是 **C2 去泄露（确定性，100%）**；**LLM label 发现是脆弱部分（88–94%）**——与 Exp1"合理但非最优 label 不被 loop 纠"一致。

## 3. 写论文怎么用（支撑 C2 + agent 诚实定位）
1. **validity ≠ quality**：agent 当前保证 valid+decontam（6/6 / 100%），但不保证"好 benchmark"（quality ~2/6）→ **C2 报告卡质量门是必要补充**，把"是不是好 benchmark"加入自治管线（当前只有 valid+decontam 两门）。
2. **agent 诚实定位加一条**：label 发现 = 脆弱环节（88–94%），去泄露 = 确定性可靠（100%）。框架的可靠性来自后者；前者的产出须经报告卡质量门复核。
3. **arxiv→license / categories→843 是诚实反例**：写进 limitation，证"自治 schema 发现需质量门兜底"，非"agent 万能"。
4. **不冲突红线**：Exp6 不声称"agent 决策>规则"（greedy=oracle 仍立）；只证 agent 发现的 label 多数合法但质量参差 → 质量门必要。

## 4. 残留 / future
- 843/275 类被 validity 放行的具体 gate 逻辑（`measure` 的 ≤100 是否在生产 e4 跑里没生效）值得查证；但观察到的"843 类 ship 成 build_valid=true"已是质量门必要性的充分证据。
- 金标准 4 域 agent 发现 = gold label（P4 12/12），其质量见 `report_cards.md`（wikiart 合理 / arxiv feature-saturated / tmdb·steam 图退化）——即 Exp1 结论，与本文一致。
- fresh 复跑 autonomous_discover（更多 seed / 新 prompt）需 LLM 端点（本环境无 .env、网络被墙），留开放机器做。

## 产物索引
- 数据：`benchmark/results/e4_scale.json`、`e6_loop_robust.json`、`e2_agent_cost.json`（皆已存）
- 本审计：本文档 + 逻辑线 §2.5 / §3 / §4.5、实验线 §20（待加）
