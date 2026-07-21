# NeurIPS D&B 论文逻辑线 — 最终确认版（FINAL, 2026-07-11）

> ⚠️ **本文档是论文逻辑的唯一权威来源（single source of truth）。**
>
> ✅ **2026-07-15 全域实验重跑完成（20:52）**：
> 1. **mean 融合修复**（GIN 训练崩溃）：已改 OpenMAG 式 mean 融合 + grad-clip + 10 seed；
> 2. **text 去泄露修复**（label 字段在 text 特征中）：wikiart/steam/tmdb 的 `Style:`/`Genres:` 等
>    label 明文字段已从 text 编码前 strip。wikiart 进一步清理了模板句/艺术家行，text 从
>    纯标题编码。
> 3. **全域实验重跑**：NC/LP/聚类/few-shot/crossdomain/modality/modmiss/robustness/labelnoise/
>    neighbors/edge_relation/fusion/per_class/steering 全部使用去泄露 text。旧 README 已清除。
>
> 本文下方若含旧绝对数字（如 +0.0007、n=5 p=0.0625、MLP 0.95、text≈multi），一律**以**
> [**`docs/experiment_results_FINAL_decontaminated_2026-07-16.md`**](experiment_results_FINAL_decontaminated_2026-07-16.md)
> **的去泄露后完整数字总表为准**（含 NC/LP 全模态消融、steering r=0.846、聚类、few-shot 等全部表 + 统一协议参数）。
> 泄露诊断与修复经过见 [`benchmark_text_decontamination_2026-07-15.md`](benchmark_text_decontamination_2026-07-15.md)。
> 旧 `benchmark_meanfusion_fix_2026-07-15.md`、`AMMAG_全域基准实验报告_2026-06-20.md` 等去泄露前结果表一律作废。
>
> 凡与本文档冲突的早期逻辑文档，**一律以本文档为准**（见 §6「已废止文档」）。
>
> 🔧 **2026-07-17 对抗审稿修订（4 处必改，已落实）**：投稿前 4 审稿人对抗审稿 + 实验核查（见 memory `adversarial-review-findings-2026-07-16`），修订如下：
> 1. **steering 降级措辞**（C4）：r≈0.85 仍成立但 **effective n≈3 域**（cluster-robust: within r=0.48, block-perm p=0.143）→ **删 "causal law" + 精确公式 `gap=0.364·homo−0.332`**，改 "**controlled cross-domain trend**"；共线性已被反驳（gap vs degree r=0.32 n.s.）→ 当**排他证据正面写**。
> 2. **LP-ceiling 改名** → `structure-leakage ceiling`：核查证实 LP=**Label Propagation**（NC 结构泄露上限）**非性链路预测**（详见 `lp_ceiling_clarification_2026-07-16.md`）；LP 链路预测 target-edge 泄露**不存在**（`run_newdomain_lp.py:252` msg_edges 只用 train_pos）。
> 3. **CLIP 预训练污染（三域系统性成立）**：frozen CLIP-L/14 image NC 比 ImageNet-only 高 ΔF1 **wikiart 0.21 > tmdb 0.15 > steam 0.11**（详见 `clip_contamination_finding_2026-07-16.md`）→ **C2 升级为含 "encoder pre-training contamination profiling" 的三层 leak-profiling**；主表加 ImageNet-only 对照列；撤 "image 全域转正" 干净叙事。
> 4. **§4.5 旧数清理**：P1-13 "wikiart mlp 0.933>gcn 0.900" 是去泄露前旧数，已更新为去泄露后真值（wikiart GCN 0.805>MLP 0.649）。

---

## 0. 最终决策（2026-07-11 用户确认，不可回退）

| 项 | 决策 |
|---|---|
| 目标 track | **NeurIPS D&B 2027**（放弃 ICLR main） |
| 四个域 | **arxiv / steam / tmdb / wikiart**（4 个主域，全由本框架多智能体管线构造）+ **spotify**（第 5 域，零代码首触构建，C3 可复用性证据） |
| 第 4 域 | **arxiv**：最大最丰富，引用边，**text+image+formula（formula 为 OpenMAG/MAGB 都没有的独有模态）**，homophily 0.933 |
| Agent 角色 | **仅在三处 load-bearing**（见 §4）。**不在「去泄露/建图决策」上声称 agent 强于规则** |
| 写作状态 | 逻辑线已锁；**先补全所有实验，暂不写初稿** |

---

## 1. Thesis（一句话）

> **四个由统一多智能体框架构建、泄漏可控的多模态属性图（MAG）benchmark 数据集（arxiv/steam/tmdb/wikiart，同质性谱 0.24→0.93，含独有的 formula 模态），配套一套测量驱动的去泄露构建协议与一个声明式 + skill 的可复用构建框架——让新域 MAG benchmark 从「专家数月手建、且常带隐性泄露」变成「一个 YAML、零代码、自带 leak-profile」。**

---

## 2. 四个贡献（C1–C4）× 证据 × 对标 OpenMAG 维度

| 贡献 | 内容 | 既有证据（已核实） | OpenMAG 维度 |
|---|---|---|---|
| **C1 四数据集（头条资产）** | 同质性谱**横跨自然源** tmdb 0.24→steam 0.53→wikiart 0.89→arxiv 0.93（注：仅 steering 因果受控；4 域同质性是**源选择**非调参——不得称"主动设计"）；arxiv 含 formula 模态（**当模态丰富度卖，非性能——消融 +0.0007**）；每数据集附 leak-profile 一等元数据 | 4 域 `*_multimodal_graph_agent.json`（字面=agent 产物）；同质性/规模/类数已核实 | Data Quality / Significance |
| **C2 测量驱动去泄露协议（护城河）—— 三层 leak-profiling** | ①**结构级**：NC 跨切分边剪枝 + **structure-leakage ceiling**（label-propagation 上限，非性链路预测）双报 ②**特征级**：text label 字段 strip（Style:/Genres:） ③**编码器级 🆕(2026-07-17)**：frozen CLIP image 预训练污染 profiling（ImageNet-only 对照）④LP 三档负采样 random/hard/hard_degree ⑤逐关系 homophily 检测定义性边；builder↔verifier 回路执行 | 剪枝后 4 域 structure-leakage ceiling 全≤多数类（wikiart 0.921→0.024）；**CLIP vs ImageNet-only ΔF1: wikiart 0.21 / tmdb 0.15 / steam 0.11（🆕 首次量化 frozen web-encoder 预训练泄露，领域忽视的维度）**；wikiart LP random 0.99→hard_degree 0.87 | Data Quality / Necessity |
| **C3 声明式框架 + skill（可复用/可扩展引擎）** | co-attribute schema（1 YAML=1 域）；新域 file-based 0 行 Python；build-mag-dataset skill 7 阶段零代码；co-attribute + citation 两类图族 | **可审计复用 manifest（2026-07-12 P1-11，SLOC 计）**：steam/tmdb/wikiart 复用 **94.6–97.2%（94% 成立，略保守）**；**arxiv 例外=68–78%**（被 2037 行 citation/fuzzy-OCR 引擎拖低，arxiv 独有图族）——94% 必须 footnote 到 3 个 co-attribute 域；共享骨架 ~5175 SLOC；builder 子类 31–32 raw wc -l/域；**builder 155→31 半可审计**：post-dedup 产物可核实，pre-dedup 155 为设计备注基线（未提交，不可重测）；**spotify 零代码 CONFIRMED**：grep 无任何 spotify 类定义，0 Python LOC/41 YAML 行=100% 复用，`build_graph_from_config` 从 YAML+数据建出；10000 节点/17440 边/homo 0.741/text-only，NC+LP+leak-profile 齐全 → C3 零代码可复用性**证据坐实**（仅注册：config choice-list 加 "spotify"，非新框架代码） | Reproducibility / Extensibility（D&B 核心加分项） |
| **C4 benchmark 科学价值** | steering **受控结构实验**（曾称"皇冠/因果律"，2026-07-17 降级为 controlled trend）+ GNN-MLP gap × 同质性耦合 + 模态×结构耦合 + 跨域迁移负结果 | steering：3 域同节点同特征只换边（MLP bit-identical 自检通过），GCN-only：**GCN-MLP gap vs 同质性 naive r=0.875, p=0.0002, n=12**。**2026-07-17 cluster-robust 核查**：①effective n≈3 域（within residual r=0.480 p=0.114, block-perm p=0.143）→ **不再称 "causal law"**；②**共线性反驳**（gap vs log(edges) r=0.444 n.s. / vs mean_degree r=0.322 n.s.）→ 同质性是主导、度/密度非混杂，**当排他证据正面写**；③LODO drop wikiart r=0.714 p=0.031（趋势稳健非崩溃）。**删精确公式 `gap=0.364·homo−0.332` 与 crossover@homo≈0.91 点估计（n=12 过拟合 + 尾端 CI 灾难）**。注：旧 r=0.463/n=16 已废止。跨域 linear-probe 12/12 负增益 | Effectiveness / Necessity / Robustness |

---

## 3. Agent 诚实定位（核心智力动作，不得违反）

| 任务 | LLM agent 是否 load-bearing | 证据 |
|---|---|---|
| 图构建 / 去泄露**决策** | ❌ **不是**（测量才是） | 2026-07-11 亲读 `benchmark/results/igc/*.json`：S1b(blind) 下 **greedy(无LLM) = oracle**（tmdb 0.300=0.300；wikiart 0.880=0.880）；schema_rule/naive_keyword 全线坍缩 |
| 噪声非结构化数据的**语义 wrangling** | ✅ **是（但边际）** | citation fuzzy OCR：**P1-8 功效增强（2026-07-12）后诚实重定位**——旧 n=7 fuzzy 头条（LLM 0.714 vs shortlist 0.000 / production 0.286）欠功效且夸大。McNemar 配对检验（全 n=60）：5/5 不一致对偏向 LLM vs shortlist **p=0.0625（边际）**，vs 生产规则 **p=0.625（不显著）**；combined non-clean(n=17) LLM 0.824[0.647,1.000] vs production 0.706 vs shortlist 0.529。→ LLM 对 fuzzy 引用匹配是**真实但 modest**的优势，非决定性 |
| **可复用 / 零代码 / 可扩展接口** | ✅ **是**（框架+skill 层） | 1 YAML=1 域；94% 复用；spotify 全程 skill 建出 |

**原则**：
- 「去泄露靠测量（**structure-leakage ceiling** / homophily / **encoder-contamination profile**）不靠 LLM」**当优点写**——协议 principled + robust，不依赖昂贵且不稳定的 LLM 决策。
- agent 的不可替代性 = 语义 wrangling（citation）+ 可复用接口（框架+skill）+ 端到端编排。
- **绝不在论文里声称「agent 去泄露/建图决策比规则强」**——greedy=oracle 消融会打穿。
- **IGC 诚实细节（2026-07-11 完备性批判核实）**：① closed_loop **反而比 greedy 差**（tmdb 0.4005 > greedy 0.3002）——回路是**审计机制非优化器**；② greedy=oracle **只对 tmdb/wikiart 干净**，steam 有例外（greedy 保留全部关系、oracle 能找 shared_genre），须披露；③ 去泄露归因 = NC 跨切分**边剪枝**（C2 协议），非 LLM、非单纯删 shared_style；④ verifier↔builder 回路只干净跑通 1 次（2/3 崩溃/跳过，日志 LLM 计数幻觉），称 "audit-capable pipeline" 而非 "self-verifying"。

---

## 4. 三条硬约束（不可违反）

1. **不碰 agent 决策优越性**（greedy=oracle 已证伪）。
2. **不用的弱结果**：G2Image 空目录 / G2Text 仅低质初步数 / Reddit-Amazon 异构 GNN（RGCN/HAN/HGT 有代码无结果）。
3. **agent 价值唯一诚实证据点**：citation fuzzy matching——**边际**（McNemar p=0.0625 vs shortlist / p=0.625 vs production，P1-8；非"LLM 0.714≫rule"的强优势）。

---

## 4.5 Reviewer 攻击面 + 诚实重定位（完备性批判产出，写论文逐条防）

| 攻击面 | 诚实重定位 |
|---|---|
| Formula 独有模态但消融 +0.0007（no-op） | 当**模态丰富度/完整性**卖，非性能；或找数学密集子集证其有用 |
| "agent 建了数据集"但 greedy=oracle | agent 仅在语义 wrangling(citation recall 0.714)挣位置；"agent-built"=编排非决策必要 |
| "自校验框架"但回路只成 1 次、closed_loop<greedy | 改称 **audit-capable pipeline**；披露脆弱；scope 到单次干净 wikiart 跑 |
| "零代码/94% 复用"但不可审计、spotify 未建 | 出复用 manifest；spotify 首触零代码建出前 extensibility 未证 |
| "leak-free 靠 C2"但逐域机制不同、0.045 应为 0.024 | 逐域精确归因；**structure-leakage ceiling**（旧名 LP-ceiling，=label-propagation 非性链路预测）改 **0.921→0.024**；三层 leak-profiling（结构/特征/编码器） |
| "6/6 跨域负"但仅 linear_probe | 报双协议：linear-probe 负（特征不迁移）、full_finetune 中性（模型重学） |
| "主动设计同质性谱"但 4 域是自然源选择 | 数据集用 "spanning natural sources"；"controlled/causal" **只**留给 steering |
| **P1-14**："多模态 benchmark"但 image 模态边际(drop_image≈0，tmdb 反 +0.007) | **text 主导**(drop_text wikiart −0.120)；卖"模态丰富度/灵活性+未来 trainable encoder 空间"，与 formula no-op 一致；**不**卖"多模态涨点"。**🆕 2026-07-17**：image NC 高分部分来自 **frozen CLIP LAION 预训练污染**（ImageNet-only 对照 ΔF1 wikiart 0.21/tmdb 0.15/steam 0.11）→ 撤 "去泄露后 image 全域转正" 干净叙事，改 limitation；C2 升级为 encoder-contamination profiling（详见 `clip_contamination_finding_2026-07-16.md`） |
| **P1-13**：leak-free NC 表上 GNN 不碾压 MLP | 去泄露后真值（2026-07-17 修正旧数 "mlp 0.933>gcn 0.900"）：wikiart GCN 0.805>MLP 0.649（GNN 赢）、arxiv +0.005（持平）、steam/tmdb GNN<MLP。这正是 leak-free 协议生效(剪掉结构捷径)；结构价值靠 **steering 受控实验**(r=0.875，已降级为 controlled trend 非因果律，见 C4)佐证，非头条 NC 表 |
| **P1-2/P1-3**：聚类/few-shot 与 steering 一致 | 高同质 GNN 助(wikiart few-shot gcn 0.823>mlp 0.773；聚类 gcnii 0.848≫raw)，低同质无效——多任务交叉验证结构×同质性耦合 |
| **P1-8**：agent 唯一证据 citation LLM 0.714 实为边际(McNemar p=0.0625) | agent 价值点缩为"边际语义 wrangling + 可复用接口 + 编排"；**不**卖"LLM 决定性优于规则" |
| **P1-4**："单一 CLIP 编码器"但未证编码器选择 | **域依赖**：clip_large 赢 arxiv(长文档，分块+pool) 但 siglip 赢 steam(短文本)；证 per-domain encoder profiling 必要，headline 选 clip_large 是 arxiv 最优非全局最优 |

---

## 5. Story Arc（章节，D&B 向）

§1 Intro（领域瓶颈：新域 benchmark 难建+易泄露，**含 frozen-encoder 预训练污染**）→ §2 Related（OpenMAG/MAGB/OGB + LLM-agent 数据管线 + gap）→ **§3 数据集（C1）** → **§4 构建协议（C2，三层 leak-profiling + greedy=oracle 诚实消融）** → **§5 框架+skill（C3）** → **§6 benchmark 价值（C4，steering 受控实验 + 三层 leak-profiling）** → §7 广度任务（NC/LP/聚类/few-shot/encoder/模态匹配/**CLIP-vs-ImageNet-only 污染对照**）→ §8 Discussion（局限+agent 诚实 scope+社区新域）→ §9 Ethics/Licensing/Reproducibility。

---

## 6. 标题候选（D&B 向，突出数据集+可复现）

- *MAG-Forge: Four Leak-Profiled Multimodal Attribute Graph Benchmarks and a Reusable Agent-and-Skill Construction Framework*
- *From Metadata to Trustworthy Benchmark: A Measurement-Grounded, Declarative Pipeline for Multimodal Attribute Graphs*
- *Building Multimodal Graph Benchmarks That Mean What They Say: A Self-Verifying Framework and Four New Domains*

---

## 7. 已废止文档（不再权威，避免污染后续思考）

以下文档的**逻辑排序/论点**已废止。其中的**原始实验数字**仍有效，仅须按本文档重新解读。

| 文档 | 状态 | 废止原因 |
|---|---|---|
| `docs/论文逻辑线与实验全景报告_2026-07-09.md` | ❌ 逻辑废止 | 旧排序 C1=benchmark 发现（GNN-MLP gap/LP 饱和）当头条、agent 降级为 C3「how we built it」；本 FINAL 反转：数据集+框架为头条，benchmark 发现降为 C4 价值证据 |
| `docs/paper_option1_method_analysis_2026-07-07.md` | ❌ 废止 | Option 1 方法分析（leakage/saturation 方法论主线） |
| `docs/paper_storyline.md` | ❌ 废止 | 早期 storyline |
| `docs/PAPER_THESIS_v2.md` | ❌ 废止 | 早期 thesis 稿，含旧 agent 定位 |
| `docs/positioning_and_scalability_2026-06-23.md` | ❌ 废止 | 早期 positioning |
| `docs/leakfree_main_tables_2026-07-03.md` | ⚠️ 数据有效，定位重解读 | 表数字有效，但不再是头条贡献，并入 C2 |
| `docs/AMMAG_全域基准实验报告_2026-06-20.md` / `Supplementary_Experiment_Report.md` / `igc_results_2026-07-03.md` / `all_experiment_results.json` | ✅ 数据有效 | 原始实验数据，按 C1–C4 重新归类 |

**后续会话如发现思路与本文档冲突，以本文档为准；不得回退到上述废止文档的论点。**

---

*生成于 2026-07-11，用户最终确认。实验路线图见同目录 `experiment_roadmap_2026-07-11.md`（Workflow 产出后写入）。*
