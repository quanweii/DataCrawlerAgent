# Related Work — 论文最终版（FINAL, 2026-08-06；对齐 07-31 新 thesis「开源自治框架」hero）

> **STATUS: STALE — 禁止直接用于投稿（2026-09-10）。** 本文件仍采用旧的“结构泄漏、LP-ceiling、四域剪枝、NC/LP 统一协议”叙事，与当前 `paper_logic_DnB_FINAL_2026-07-11.md` 的 provenance-confirmed defect / protocol condition / diagnostic signal 区分冲突。涉及他人数据集“泄漏”、`6/6 held-out`、`100% decontamination`、OpenMAG venue/status 及 first/none claims 的表述必须逐条用一手来源重新核验后重写；在此之前，只有当前逻辑线、实验线和 framework traceability 文件可作为权威来源。

> ⚠️ **本文是 §2 Related Work 的唯一权威来源（single source of truth）。** 所有数字以 [`experiment_results_FINAL_decontaminated_2026-07-16.md`](experiment_results_FINAL_decontaminated_2026-07-16.md) 为准；逻辑线以 [`paper_logic_DnB_FINAL_2026-07-11.md`](paper_logic_DnB_FINAL_2026-07-11.md)（2026-07-31 thesis 重定版）为准。
>
> 🆕 **2026-08-06 重写（本次，最高优先）—— 为什么重写**：旧版 `related_work_FINAL_2026-07-16.md` 是在**旧 thesis**（去泄露当护城河 + steering 当皇冠）下写的，与 07-31 新 thesis 冲突。本次：
> - **对齐 07-31 新 thesis**：hero = **开源自治多智能体 MAG 构造框架**（5 链环：①稀缺 ②自动化 ③高质量 ④正确 ⑤可复用）；**去泄露降级为 benchmark 正确性属性**（非头条）；**steering 降 appendix**（机制演示，非皇冠）；**报告卡质量门（Exp2 P0）升为 C2 一等贡献**。
> - **新增 6 个对手群**（07-16 完全缺失）：A2 agent 数据流水线、F 数据集文档/质量、G 单模态图 benchmark 族、H 可复现/声明式基建、I CLIP 记忆污染、J 合成 vs 真实数据；外加 **K Zhewei Wei 课题组**（重点核查，见 §4）。
> - **嵌两个新杀手锏**：① **generation-vs-extraction（E3）**——合成数据泄露语义级不可剥离 / 真实+C2 字段级可剥离（hero 范式钩）；② **valid ≠ good（Exp6）**——843 类 label 过 validity 门 → 报告卡质量门必要（P0 动机）。
> - **修正 1 处 load-bearing venue 错**：Platonov「Critical Look」= **ICLR 2023**（旧版/旧记忆误作 NeurIPS 2023）；OpenMAG 升格 **ICML 2026**（最强对手）。
> - **弃用旧 steering 钩**「They observe neighbor-density; we provide the knob」（steering 已降 appendix）。
>
> 🔴 **旧版 `related_work_FINAL_2026-07-16.md` 与调研底稿 `related_work_agent_dataset_construction_2026-07-13.md` 均已删除（被本文取代）**。

---

## 0. 一句话定位（07-31 thesis 版）

> **在「真实多模态源编排成 MAG × 测量驱动去泄露 × 自动报告卡质量门 × 声明式骨架复用」这个四叉路口，之前没有任何一站过——每个对手只占其中一条轴。**

精确地：
- 我们**不是**首个 MAG benchmark（**MAGB** KDD'25、**MM-Graph/Mosaic** CVPR'25、**OpenMAG** ICML'26 已在先，且**都不是** Zhewei Wei 课题组）——独占性来自 **真实源编排 ∩ 去泄露协议 ∩ 报告卡质量门 ∩ 声明式复用 ∩ 自治 schema 发现** 这一**组合**，而非 MAG benchmark 单一角。
- 图上的结构性泄露检测**已有 prior**（Platonov ICLR'23 / Zhu WSDM'24 / Li NeurIPS D&B'23 / **Wei 课题组 TGB-Seq ICLR'25**）——但它们**全在非多模态图、无 LP-ceiling 量化、无报告卡**。我们首次把 leak-profiling + LP-ceiling 量化 + 多模态 text 定义级泄露 + NC/LP 统一协议 + **valid≠good 质量门**带进 MAG。
- **范式独占**：generation agent（BenchAgents/AutoBencher/**Wei 课题组 GAG** ACL'25F）合成实例，**其泄露语义级不可剥离**；我们编排**真实**源、在**字段级**剥离泄露——**E3 实测**，非断言。

最接近的两类——agent-built benchmark（BenchAgents/DatasetAgent/GAG）与 multimodal/MAG benchmark（OpenMAG/MAGB/MM-Graph）——**前者有 agent 但造合成实例/知识图谱、无防泄露、无质量门；后者有 MAG 但手工建造、不审结构性泄露（Exp5b 证 OpenMAG 真图也 ship +0.50）、无 agent/声明式框架**。

---

## 0.5 方向定位（最重要的辨别：我们在「真实抽取→清洗→建图」方向，不在「生成」方向）

> ⚠️ **本文的方向 = 真实世界数据「抽取 → 清洗 → 建图」**：从真实多模态源（论文/游戏/电影/艺术/音乐/商品 API）**抽取**实体与属性、**清洗**（去重 / 去泄露 / 字段规整 / 同质性感知剪枝）、**建**多模态属性图 benchmark。
>
> 这与当下主流「agent 造 benchmark」文献是**同一终态（benchmark）、不同方向**。务必在 §1 Intro 与 §2.3 开头讲清，**否则审稿人会误把生成 agent 当直接竞品**。按「数据从哪来」分三个方向：

| 方向 | 代表工作 | 输入 | 产物 | 与本文关系 |
|---|---|---|---|---|
| **生成（合成）** | BenchAgents / AutoBencher / DatasetAgent / CAMEL / AgentInstruct / **GAG** | LLM（+少量种子） | 合成实例 / 合成图 | **不同方向**——LLM 合成新数据，适用于「真实数据不足/不存在」；我们 E3 证其合成数据带**语义级不可剥离**泄露，故不走此路 |
| **抽取/清洗/编排**（**同方向最近亲**） | **Data-Copilot / DS-Agent / OpenAgents / TaskWeaver / Data Interpreter** + Snakemake/Nextflow + DataComp | **真实数据** | 表格分析 / 可视化 / 通用 pipeline | **同方向**（共享「操作真实数据」哲学），但产物**非图 benchmark、无去泄露**——正是我们的空白 |
| 抽取 → **KG**（正交） | KARMA | 真实文献 | 知识图谱三元组 | 正交（非属性图 benchmark） |
| **抽取 → 清洗 → MAG**（**本文**） | **Ours** | **真实多模态源** | **去泄露 MAG benchmark + 报告卡** | **填补空白** |

> **三条要点**（写进 Intro/§2.3）：
> 1. **A1 生成 agent + GAG 不是直接竞品**——它们解「真实数据不足/不存在时如何合成」的**不同问题**；我们解「真实数据已存在时如何抽取清洗建图」。**同终态（benchmark）、不同方向（合成 vs 真实抽取）。**
> 2. **同方向最近亲**是 A2 抽取/编排 agent（Data-Copilot/DS-Agent/OpenAgents/TaskWeaver/Data Interpreter），但**没有一个建图 benchmark、没有一个去泄露**——这正是我们的独占空白。
> 3. **同产物的直接竞品**是 C 群（OpenMAG/MAGB/MM-Graph，都造 MAG），但它们**手工建造、不审泄露（Exp5b 证 OpenMAG 真图也 ship +0.50）、无 agent 框架**。
>
> 三方各缺一角 → 我们「真实抽取+清洗+建图 + 去泄露 + 报告卡」的**交集独占**。

---

## 1. 对手分类（11 群，mapped 到 5 链环；🆕=07-16 缺、本次新增）

| 群 | 代表工作（已联网核查） | 服务链环 | 07-16 状态 |
|---|---|---|---|
| **A1 生成式 agent**（LLM 合成实例；**不同方向·生成**） | BenchAgents (ICLR'25 ws)、AutoBencher (ICLR'25)、DatasetAgent (arXiv'25)、CAMEL (NeurIPS'23)、AgentInstruct (arXiv'24)、Benchmark Self-Evolving (COLING'25) | ②（**不同方向·生成**，非直接竞品） | ✅ |
| **A2 🆕 数据流水线/ETL agent**（编排真实源；**同方向·抽取**） | **Data-Copilot** (Zhang, ICLR'24)、**DS-Agent** (Guo, ICML'24)、**OpenAgents** (Xie, COLM'24)、TaskWeaver (arXiv'23)、Data Interpreter (Hong, ACL'25F) | ②（**同方向·抽取**，最近亲） | ❌ 新 hero 核心对照 |
| A3 多智能体骨架 | AutoGen (Wu, COLM'24)、MetaGPT (Hong, ICLR'24) | ②（substrate，一句带过） | ❌ |
| **B agent 建 KG** | KARMA (NeurIPS'25 Spotlight)、Bian LLM-KG survey (arXiv'25)、Neo4j Graph Builder | ②（正交） | ✅ |
| **C 多模态图 benchmark**（最强对手） | **OpenMAG (ICML'26)**、MAGB (KDD'25)、MM-Graph/Mosaic (CVPR'25) | ①+③ | ✅（OpenMAG 升 ICML'26） |
| **D LLM 基准污染** | AntiLeakBench (ACL'25)、BDC survey (arXiv'24)、Emperor (ICML'25)、DataSAIL (NatComm'25) | ④（正交机制） | ✅ |
| **E 图结构性泄露（非多模态）** | Platonov (ICLR'23)、Zhu (WSDM'24)、Li (NeurIPS D&B'23)、Brière (bioRxiv'25) | ④（直接 prior） | ✅（修 venue） |
| **F 🆕 数据集文档/质量** | Datasheets (CACM'21)、Data Cards (FAccT'22)、Nutrition Label (NeurIPS'18ws)、Pervasive Label Errors (NeurIPS'21)、Underspecification (JMLR'22)、Recht fresh-test (ICML'19) | ③（**报告卡 prior**） | ❌ P0 无支撑 |
| **G 🆕 单模态图 benchmark 族** | OGB (NeurIPS'20)、Shchur "Pitfalls of" (NeurIPS'18ws)、Dwivedi "Benchmarking GNNs" (JMLR'23)、Lim non-homophilous (NeurIPS'21)、Zheng/Pan heterophily survey (TKDE'26) | ①（**量化 MAG 稀缺**） | ❌ |
| **H 🆕 可复现/声明式基建** | Pineau reproducibility (JMLR'21)、Snakemake (Bioinf.'12)、Nextflow (NBT'17) | ⑤ | ❌ C4 无支撑 |
| **I 🆕 CLIP/VLM 记忆污染** | Carlini (USENIX'21/'23)、LAION-5B (NeurIPS'22)、MetaCLIP/Demystifying CLIP Data (ICLR'24) | ④（encoder profiling prior） | ❌ E5 无依据 |
| **J 🆕 合成 vs 真实数据** | Model collapse (Shumailov, Nature'24)、DataComp (NeurIPS'23) | ②（**E3 prior**） | ❌ |
| **K 🆕 Zhewei Wei 课题组**（重点核查） | TGB-Seq (ICLR'25)、GAG/LLM-Multi-Agent-Graph-Generative (ACL'25F)、GDGB (ICLR'26)、Rethinking-LP-Directed (TPAMI'26)、TAG-Robustness (ICLR'26)、PolyGCL (ICLR'24) | ②③④（LP 去污 + agent 建图 + TAG eval） | ❌ 全缺 |

---

## 2. 核心对比表（直接进论文/PPT；按 5 链环排，🆕=新维度/新对手）

> 列：每群挑 1–2 代表。✅=有 ❌=无 🟡=部分/不同机制

| 能力维度（链环） | BenchAgents<br>(A1 生成) | Data-Copilot<br>(A2 🆕流水线) | GAG<br>(K Wei·生成图) | OpenMAG<br>(C MAG) | Platonov /<br>TGB-Seq<br>(E/K 泄露) | Data Cards<br>(F 🆕文档) | Snakemake<br>(H 🆕基建) | **Ours** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ② 真实源编排→**MAG**（非合成） | ❌ 合成实例 | ✅ 真实**表格** | ❌ **合成图** | ❌ 手工 | ❌ | ❌ | ❌ 通用引擎 | **✅ 真实多模态→MAG** |
| ② **生成 vs 抽取泄露轴实证**（E3） | ❌（其合成泄露不可剥） | ❌ | ❌（合成图，无泄露讨论） | ❌ | ❌ | ❌ | ❌ | **✅ 合成不可剥/真实可剥** |
| ② 自治 schema 发现 + 端到端 | 🟡 半自动+人工 | ✅ 但无 benchmark | ✅ 但造合成图 | ❌ 手工 | ❌ | ❌ | ❌ | **✅ E2/E4 6/6 held-out** |
| ③ **报告卡质量门（valid≠good）** | ❌ agent 当 load-bearing | ❌ | ❌ | ❌ | ❌ | 🟡 **人写**散文卡 | ❌ | **✅ 自动生成+机器门(Exp2/6)** |
| ③ 同质性谱 + 诚实退化标注 | ❌ | ❌ | ❌ | 🟡 多域无谱/无标注 | 🟡 单模态 | ❌ | ❌ | **✅ 0.24→0.93 谱 + 标红** |
| ④ **MAG 上去泄露 + LP-ceiling 量化** | ❌ | ❌ | ❌ | ❌（**Exp5b 自家 +0.50**） | 🟡 **非多模态**/无 LP-ceiling | ❌ | ❌ | **✅ 四域剪枝后全≤majority** |
| ④ 外部铁证（公开/他人 MAG 也漏） | ❌ | ❌ | ❌ | ❌ | 🟡 仅自检/同机制 | ❌ | ❌ | **✅ E1 七公开+0.43~0.68** |
| ⑤ 声明式骨架复用（1 YAML=1 域） | ❌ 需人工编辑 | ❌ | ❌ | ❌ | ❌ | ❌ | 🟡 通用 DSL | **✅ 94%复用/spotify 0 LOC** |

**读表结论（§2.8 Positioning 骨架）**：**没有任何一站同时覆盖「真实多模态源编排成 MAG × 测量驱动去泄露 × 自动报告卡质量门 × 声明式骨架复用」。** A2 群最接近「编排真实源」但产物是表格分析、无图无 benchmark 无去泄露；C 群最接近「MAG」但手工、不审泄露、无框架；F 群最接近「质量文档」但是人写散文卡；E+K 群最接近「图泄露」但全在非多模态图、无 LP-ceiling、无 MAG 实例、无质量门。

---

## 3. 贡献 & 独特性体现在哪（逐链环，审稿人真正问的）

### 链环②（hero C1 自动化）—— **最大、最该重写的一环**

> **方向前置（务必讲清）**：本文在「**真实抽取 → 清洗 → 建图**」方向（见 §0.5）。A1 生成 agent 与 K·GAG 是**不同方向（合成）、非直接竞品**；A2 抽取/编排 agent 才是**同方向最近亲**；C 群是**同产物（MAG）直接竞品**但手工。三方各缺一角 → 我们的独占。

- **vs A1 生成式 agent + K·GAG（不同方向·合成）**：它们解「真实数据不足/不存在时如何合成」，我们解「真实数据已存在时如何抽取清洗建图」——**不同问题**。我们刻意不走生成路：E3 证 LLM 在「已知 label」下合成 text，泄露**语义级、不可剥离**（strip=no-op，acc≈1.0）；真实抓取+C2 剥离**字段级、可剥离**（acc 崩到≈majority）。**「generation 路线泄露不可修，extraction 路线可修」= 我们选 extraction 方向的实证依据**。GAG（Wei 组）合成图同样落入生成范式，正好坐实区分。
- **vs A2 🆕数据流水线 agent**：Data-Copilot/DS-Agent/OpenAgents 编排真实数据做**表格分析/可视化**，无图结构、无多模态源、无去泄露、不产出 benchmark →「编排真实多模态源 → MAG benchmark」是真空白。
- **agent 价值诚实 scope**（守红线① greedy=oracle）：load-bearing 在**自动化（自治 schema 发现 + 端到端编排 + 真实源获取）**，不在构造决策。E6 量化「可靠核心=C2 确定性测量（去泄露 100%）；脆弱环节=LLM label 发现（88–94%）」。

### 链环③（C3 高质量 + 报告卡 P0）—— **新贡献面，07-16 完全没覆盖**
- **valid ≠ good（Exp6 核心洞察）**：843 类 label 过了 validity 门 →「合法+去泄露」不够，需「是不是好 benchmark」门。**这是报告卡的理论动机。**
- **vs F 🆕数据集文档**：Datasheets/Nutrition Label/Data Cards 是**创作者手写散文卡**；本文是**机器自动生成、量化、benchmark-统计驱动、且产出 pass/fail 门**——可防守差异「first auto-generated machine quality gate for MAG」。
- **vs C 群**：OpenMAG/MAGB/MM-Graph 域更多但手工、无同质性谱、不诚实标注退化域。本文 portfolio = **少而精 + 同质性谱 + 报告卡标红**（红线④：绝不硬凑「4 域都图有用」）。
- **难度/饱和轴 prior**：Underspecification (JMLR'22) + Pervasive Label Errors (NeurIPS'21) + Recht fresh-test (ICML'19) 支撑报告卡「饱和/标签噪声」两轴。

### 链环④（C2 正确性——降级为属性，不再当护城河）
- **vs E + K·TGB-Seq 图结构性泄露**：诚实收窄——「我们不发现图泄露，Platonov/TGB-Seq 在非多模态/时序图先做了；我们首次把 leak-profiling + **LP-ceiling 量化** + 多模态 **text 定义级泄露** + NC/LP 统一协议带进**静态 MAG**」。TGB-Seq 在**时序 LP** 删重复边防记忆；本文在**静态 MAG** 用跨切分剪枝 + LP-ceiling——同精神、不同图族，互为印证。
- **外部铁证堵「自证循环」**：E1（7 公开 PyG benchmark 全 ship +0.43~0.68）+ Exp5b（OpenMAG 真图 books-nc +0.50 / ele-fashion +0.47）= **领域金标准系统性漏掉这维度**。
- **vs I 🆕CLIP 记忆污染**：MetaCLIP (ICLR'24) 在 CLIP 预训练数据去污染（pipeline-hygiene prior，正可引「负责任 CLIP pipeline 已这么做」）；本文 E5 不同——**profile 冻结 encoder 在 benchmark 域上的残余污染**（任务依赖、集中 style/genre），做成报告卡一轴 =「encoder-contamination profiling as a benchmark quality axis」新角度。

### 链环⑤（C4 可复用）
- **vs A1**：BenchAgents 需大量人工编辑；AutoBencher「声明式」是 QA-topic spec，非建图骨架。
- **vs H 🆕流水线基建**：Snakemake/Nextflow 是通用 workflow 引擎；本文是 **MAG 域特定声明式骨架**（1 YAML=1 域）+ 可审计 reuse manifest（spotify 0 Python LOC / 94.6–97.2%）。Pineau (JMLR'21) 兜底「严谨」。

### 链环①（稀缺/动机）—— 最弱、只当动机
- **vs G 🆕单模态图 benchmark 族**：OGB/Shchur/Dwivedi/Lim 证单模态图 benchmark 成熟丰沛；MAG 仅 3 个（OpenMAG/MAGB/MM-Graph）全手工、零审泄露。**Exp0 landscape 表量化「少、手工、未审、域窄」**。非 novelty claim，是 motivation。

---

## 4. 🆕 Zhewei Wei 课题组重点核查（用户指定；结论：非直接竞品，但 2 篇 MUST-CITE）

> **核查结论**：Wei 课题组**无多模态/MAG（text+image）论文**，也无 benchmark-leakage 框架；其 attributed-graph 线**全为 text-attributed graphs (TAGs)**。故**非直接竞品**。但有**两篇与本文核心贡献强重叠的 MUST-CITE**，且 Wei 是顶刊图 ML 学者/大概率审稿人，必须正确引用其组工作。**注意：MAGB (KDD'25) 与 MM-Graph (CVPR'25) 均非 Wei 组**（联网核查确认，曾有 web 摘要误挂名）。

| # | 工作 | 元数据（核查） | 相关轴 | Cite | 与本文关系 |
|---|---|---|---|---|---|
| 1 | **TGB-Seq** | Yi, Peng, Zheng, Mo, **Wei**, Ye, Yue, Huang. *TGB-Seq Benchmark: Challenging Temporal GNNs with Complex Sequential Dynamics*. **ICLR 2025**, arXiv:2502.02975 | LP 泄露/去污 | **MUST** | **本文 LP 去污的最强概念锚**：其在时序 LP 删重复边防「记忆作弊」，与本文静态 MAG 跨切分剪枝 + LP-ceiling 同精神、不同图族 |
| 2 | **GAG / LLM-Multi-Agent Graph Generative** | Ji, Lei, Bi, **Wei***, Chen, Lin, Pan, Li, Ding. *LLM-Based Multi-Agent Systems are Scalable Graph Generative Models*. **ACL 2025 Findings**, arXiv:2410.09824 | agent 建图 | **MUST** | **本文 agent-MAG-构造的最接近先例**：其多智能体**合成**图；本文从**真实**源构造 benchmark MAG——正好坐实 generation-vs-extraction（E3）区分 |
| 3 | **GDGB** | Peng, Ji, Lei, **Wei***, Liu*, Hong. *GDGB: A Benchmark for Generative Dynamic Text-Attributed Graph Learning*. **ICLR 2026**, arXiv:2507.03267 | 图 benchmark | MUST | 其组最接近的 benchmark 同型物：**动态 text-only** DyTAG vs 本文**静态多模态** MAG——须显式区分 |
| 4 | **Rethinking LP for Directed Graphs** | He, Guo, Zheng, **Wei**, Günnemann, Xiao. *Rethinking Link Prediction for Directed Graphs*. **TPAMI 2026**, arXiv:2502.05724 | LP 数据视角 | MUST | 其组 LP 数据视角旗舰；讨论 LP 评估方法处引用 |
| 5 | **TAG-Robustness / TGRB** | Lei, Yi, He, Qiu, **Wei***, Liu*, Hong. *Robustness in Text-Attributed Graph Learning: Insights, Trade-offs, and New Defenses*. **ICLR 2026**, arXiv:2510.17185 | TAG eval | MUST | 与本文 GNN-vs-encoder 评估轴重叠；TAG 鲁棒性大规模评测，与 MAGB 并列 |
| 6 | PolyGCL | Chen, Lei, **Wei***. *PolyGCL: Graph Contrastive Learning via Learnable Spectral Polynomial Filters*. **ICLR 2024 (Spotlight)**, OpenReview:y21ZO6M86t（无 arXiv） | 同/异质 | nice | 跨同质性谱谱滤波，支撑 homophily-spectrum framing |
| 7 | Beyond Over-smoothing | Peng, Lei, **Wei**. *Beyond Over-smoothing: Uncovering Trainability Challenges in Deep GNNs*. **CIKM 2024** | GNN 失效 | nice | 支撑「去泄露后 GNN≲MLP 非构图 bug」框定 |

**写法建议**：把 TGB-Seq 并入 §2.6（图泄露），把 GAG 并入 §2.3（generation-vs-extraction，作 agent 建图对照），GDGB + TAG-Robustness 并入 §2.1（MAG vs TAG benchmark 对比）。这样 Wei 组 5 篇 MUST-CITE 自然分散到对应小节，不堆砌。

---

## 5. 【可直接贴论文】英文 Related Work 正文（07-31 thesis 版；引用 key 见 §6）

### 2.1 Multimodal Attributed Graph Benchmarks

MAGB [Yan et al. 2025] introduced a systematic multimodal attributed graph (MAG) benchmark and studied GNN-as-predictor versus VLM-as-predictor paradigms over text and visual attributes. Mosaic of Modalities (MM-Graph) [Zhu et al. 2025] assembled seven text+visual graph datasets spanning classification, link prediction, and knowledge-graph completion. OpenMAG [Wan et al. 2026] expanded this line to 19 datasets, 16 encoders, and 24 models, establishing the breadth baseline. **All three are hand-curated, and none performs homophily-aware structural leakage analysis, reports an LP-ceiling, attaches leak-profile metadata, or ships a construction framework.** We do **not** claim the first MAG benchmark; our contribution is *agent-driven construction, MAG-level leak profiling with LP-ceiling quantification, a report-card quality gate, and declarative reuse* layered on this benchmark family. A parallel line targets **text-attributed graphs (TAGs)** — single-modality text+structure — including GDGB [Peng et al. 2026] (dynamic TAG benchmark) and TGRB [Lei et al. 2026] (TAG robustness); our setting adds the visual modality that these TAG works do not address.

### 2.2 Graph Benchmarks and GNN Evaluation Pitfalls

Single-modal graph benchmarks are mature and abundant: OGB [Hu et al. 2020] and its large-scale challenge [Hu et al. 2021] provide standardized datasets, while Dwivedi et al. [2023] and Lim et al. [2021] establish benchmarking protocols and non-homophilous benchmarks. Shchur et al. [2018] and Platonov et al. [2023] show that GNN rankings are highly sensitive to evaluation protocol (splitting, features, repeats), a lesson we adopt in our leak-free protocol. Surveys of heterophily [Zheng et al. 2026] formalize the homophily spectrum that our portfolio deliberately spans. **By contrast, the MAG niche is small (three families, all hand-built, none leak-audited)** — quantifying this gap is our Exp0 landscape contribution.

### 2.3 We Take the Extraction Direction, Not the Generation Direction

Our work is an **extraction–cleaning–construction** pipeline: it fetches real multimodal sources, cleans and decontaminates them, and builds attribute-graph benchmarks. This is a *different direction* from the dominant **generation** line of LLM-agent benchmark construction, which *synthesizes* instances when real data is unavailable or insufficient; we relate to each line differently.

**Generation agents (a different direction, not direct competitors).** BenchAgents [Butt et al. 2025] (four agents synthesizing multimodal task benchmarks, load-bearing agent loop, developer-in-the-loop editing), AutoBencher [Li et al. 2025] (declarative QA-topic generation), DatasetAgent [Sun et al. 2025] (image datasets), CAMEL [Li et al. 2023], AgentInstruct [Mitra et al. 2024], Benchmark Self-Evolving [Wang et al. 2025], and GAG [Ji et al. 2025] (multi-agent *synthetic* graph generation) all synthesize new data rather than curate real data. They solve a different problem — generating instances when real data is scarce — and we deliberately avoid this direction: an LLM that synthesizes text *given the label* embeds leakage that is **semantic and unstrippable** (E3: field-stripping is a no-op, accuracy stays ≈1.0).

**Extraction/curation agents (our direction-mates).** Data-Copilot [Zhang et al. 2024], DS-Agent [Guo et al. 2024], OpenAgents [Xie et al. 2024], TaskWeaver [Qiao et al. 2023], and Data Interpreter [Hong et al. 2025] share our philosophy of operating on *real* data — yet none targets graph structure, multimodal attributes, or structural leakage control, and none produces a benchmark. **We fill exactly this gap**, orchestrating real multimodal sources into leak-profiled MAGs, where leakage is **field-level and strippable** (E3: accuracy collapses to ≈majority after C2 decontamination). Our greedy-without-LLM ablation further shows the *decontamination decision itself does not require an LLM* — measurement (LP-ceiling, homophily) suffices — which we frame as robustness, not weakness. (KG-construction agents such as KARMA, §2.4, also extract but to entity–relation triples — a different graph family; multi-agent substrates AutoGen [Wu et al. 2024] and MetaGPT [Hong et al. 2024] are orthogonal infrastructure.)

### 2.4 LLM-Agent Knowledge-Graph Construction

KARMA [Lu et al. 2025] (NeurIPS 2025 Spotlight) enriches knowledge graphs from scientific literature via multi-agent entity/relation extraction and conflict resolution; Bian [2025] surveys LLM-driven KG construction. **Knowledge graphs (entity–relation triples) are a different graph family from the co-attribute MAGs we build**, and these works release no benchmarks, evaluate no GNNs, and address no leakage. We position our co-attribute MAG family as orthogonal to KG construction.

### 2.5 Dataset Documentation and Quality Assessment

Datasheets for Datasets [Gebru et al. 2021], the Dataset Nutrition Label [Holland et al. 2018], and Data Cards [Pushkarna et al. 2022] established transparency artifacts that dataset creators fill in by hand; Sambasivan et al. [2021] document how neglected "data work" compounds into quality failures. On the benchmark side, Northcutt et al. [2021] show pervasive label errors in test sets, D'Amour et al. [2022] show that a single metric under-specifies credibility, and Recht et al. [2019] show benchmark scores drop on fresh test sets. **These motivate our report-card quality gate but are creator-authored prose; ours is auto-generated, quantitative, benchmark-statistics-driven, and emits a pass/fail verdict per domain.** Crucially, our Exp6 shows a benchmark can be *valid* (passes split + leakage checks) yet *not good* (e.g., an 843-class label sailed through validity) — exactly the gap the quality gate closes.

### 2.6 Benchmark Leakage: Graph-Structural, Temporal, LLM-Contamination, and Encoder Memorization

Closest to our decontamination contribution is **structural leakage in (non-multimodal) graphs**: Platonov et al. [2023] show heterophily benchmarks leak via duplicate nodes; Zhu et al. [2024] define *implicit test leakage* from target edges left in the message-passing graph; Li et al. [2023] survey LP evaluation pitfalls; Brière et al. [2025] benchmark leakage in biomedical KG link prediction; and **TGB-Seq [Yi et al. 2025] curates temporal LP datasets to remove repeated edges so models cannot memorize**. These establish graph leakage as real and studied — we claim no discovery. Our distinct contribution is to (i) bring leak-profiling and decontamination into the **multimodal** attribute-graph setting for the first time, (ii) introduce the **LP-ceiling** as a quantifiable, constructible degeneracy metric on **static** MAGs (TGB-Seq's decontamination spirit, ported from temporal LP to static MAG), (iii) expose a previously undocumented **definition-level leakage in multimodal text attributes**, and (iv) unify NC and LP leakage into one protocol. For **LLM** contamination, AntiLeakBench [Wu et al. 2025], the BDC survey [Xu et al. 2024], the Emperor's-New-Clothes audit [Sun et al. 2025], and DataSAIL [Joeres et al. 2025] target test-in-train contamination or i.i.d. entity-pair similarity — a different mechanism, orthogonal to message-passing cross-split edges. For **encoder** contamination, Carlini et al. [2021; 2023] extract training data from LLMs/diffusion models, LAION-5B [Schuhmann et al. 2022] is the suspected pretraining corpus, and MetaCLIP [Xu et al. 2024] already performs benchmark-decontamination in CLIP data curation; our E5 profiles the *residual* contamination of a frozen encoder on benchmark domains (task-dependent, concentrated in style/genre) and exposes it as a report-card axis. Finally, synthetic-data scaling itself is not free — model collapse [Shumailov et al. 2024] and DataComp [Gadre et al. 2023] — reinforcing our real-fetch choice.

### 2.7 Reproducible and Declarative Pipelines

Pineau et al. [2021] established the ML reproducibility checklist; declarative workflow engines Snakemake [Köster & Rahmann 2012] and Nextflow [Di Tommaso et al. 2017] make reproducible pipelines declarative across scientific domains. **These are general-purpose engines; our contribution is a domain-specific declarative *skeleton* for MAG construction — one YAML yields one benchmark domain, audited via a reuse manifest (Spotify: 0 Python LOC; Steam/TMDB/WikiArt: 94.6–97.2% reuse).**

### 2.8 Positioning Statement

**We are the first work to simultaneously satisfy (i) agent-driven, declaratively reusable construction of MAG benchmarks from real multimodal sources; (ii) a measurement-grounded structural-decontamination protocol with LP-ceiling quantification on MAGs, externally validated (public benchmarks and published MAGs both leak); (iii) an auto-generated report-card quality gate that operationalizes "valid ≠ good"; and (iv) a portfolio spanning a natural homophily spectrum, with honest per-domain quality annotation.** Where OpenMAG observes that excessive neighborhood density dilutes node features, we *measure and decontaminate* leakage; where GAG *synthesizes* graphs, we *extract* them — and show the leakage consequences.

---

## 6. 参考文献（bibtex，2026-08-06 联网逐篇核查）

```bibtex
% ===== A1 生成式 agent =====
@inproceedings{butt2025benchagents,
  title={BenchAgents: Automated Benchmark Creation with Agent Interaction},
  author={Butt, Natasha and Chandrasekaran, Varun and Joshi, Neel and Nushi, Besmira and Balachandran, Vidhisha},
  booktitle={ICLR 2025 Workshop on Data Problems for Foundation Models (DATA-FM)}, year={2025}, note={arXiv:2410.22584}}
@inproceedings{li2025autobencher,
  title={AutoBencher: Towards Declarative Benchmark Construction},
  author={Li, Xiang Lisa and Kaiyom, Farzaan and Liu, Evan Zheran and Mai, Yifan and Liang, Percy and Hashimoto, Tatsunori}, booktitle={ICLR}, year={2025}}
@article{sun2025datasetagent,
  title={DatasetAgent: A Novel Multi-Agent System for Auto-Constructing Datasets from Real-World Images},
  author={Sun, Haoran and Bian, Haoyu and Zeng, Shaoning and Rao, Yunbo and Xu, Xu and Mei, Lin and Gou, Jianping}, journal={arXiv:2507.08648}, year={2025}}
@inproceedings{li2023camel,
  title={CAMEL: Communicative Agents for ``Mind'' Exploration of Large Language Model Society}, author={Li, Guohao and Hammoud, Hasan Abed Al Kader and others}, booktitle={NeurIPS}, year={2023}}
@inproceedings{wang2025selfevolving,
  title={Benchmark Self-Evolving: A Multi-Agent Framework for Dynamic LLM Evaluation}, author={Wang, Siyuan and Long, Zhuohan and Fan, Zhihao and Huang, Xuanjing and Wei, Zhongyu}, booktitle={COLING}, year={2025}}
@article{mitra2024agentinstruct,
  title={AgentInstruct: Toward Generative Teaching with Agentic Flows}, author={Mitra, Arindam and others}, journal={arXiv:2407.03502}, year={2024}}

% ===== A2 数据流水线/ETL agent (NEW) =====
@inproceedings{zhang2024datacopilot,
  title={Data-Copilot: Bridging Billions of Data and Humans with Autonomous Workflow}, author={Zhang, Wenqi and Shen, Yongliang and Lu, Weiming and Zhuang, Yueting}, booktitle={ICLR}, year={2024}, note={arXiv:2306.07209}}
@inproceedings{guo2024dsagent,
  title={DS-Agent: Automated Data Science by Empowering Large Language Models with Case-Based Reasoning}, author={Guo, Siyuan and Deng, Cheng and Wen, Ying and Chen, Hao and Chang, Yongshao and Wang, Jian}, booktitle={ICML}, year={2024}, note={arXiv:2402.17453}}
@inproceedings{xie2024openagents,
  title={OpenAgents: An Open Platform for Language Agents in the Wild}, author={Xie, Tianbao and others}, booktitle={COLM}, year={2024}, note={arXiv:2310.10634}}
@article{qiao2023taskweaver,
  title={TaskWeaver: A Code-First Agent Framework}, author={Qiao, Bo and others}, journal={arXiv:2311.17541}, year={2023}}
@inproceedings{hong2025datainterpreter,
  title={Data Interpreter: An LLM Agent For Data Science}, author={Hong, Sirui and others}, booktitle={Findings of ACL}, year={2025}, note={arXiv:2402.18679}}

% ===== A3 多智能体骨架 =====
@inproceedings{wu2024autogen,
  title={AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation Framework}, author={Wu, Qingyun and Bansal, Gagan and Zhang, Jieyu and others}, booktitle={COLM}, year={2024}, note={arXiv:2308.08155}}
@inproceedings{hong2024metagpt,
  title={MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework}, author={Hong, Sirui and Zhuge, Mingchen and Chen, Jonathan and others}, booktitle={ICLR}, year={2024}, note={arXiv:2308.00352}}

% ===== B agent 建 KG =====
@inproceedings{lu2025karma,
  title={KARMA: Leveraging Multi-Agent LLMs for Automated Knowledge Graph Enrichment}, author={Lu, Yuxing and Wu, Wei and Zhao, Xukai and Peng, Rui and Wang, Jinzhuo}, booktitle={NeurIPS (Spotlight)}, year={2025}, note={arXiv:2502.06472}}
@article{bian2025llmkg,
  title={LLM-empowered Knowledge Graph Construction: A Survey}, author={Bian, Haonan}, journal={arXiv:2510.20345}, year={2025}}

% ===== C 多模态图 benchmark =====
@inproceedings{wan2026openmag,
  title={OpenMAG: A Comprehensive Benchmark for Multimodal-Attributed Graph}, author={Wan, Chenxi and Li, Xunkai and Zuo, Yilong and Deng, Haokun and Li, Sihan and Fan, Bowen and Qin, Hongchao and Li, Ronghua and Wang, Guoren}, booktitle={ICML}, year={2026}, note={arXiv:2602.05576}}
@inproceedings{yan2025magb,
  title={When Graph Meets Multimodal: Benchmarking and Meditating on Multimodal Attributed Graph Learning}, author={Yan, Hao and Li, Chaozhuo and Yin, Jun and Yu, Zhigang and Han, Weihao and Li, Mingzheng and Zeng, Zhengxin and Sun, Hao and Wang, Senzhang}, booktitle={KDD (Datasets and Benchmarks)}, year={2025}, doi={10.1145/3711896.3737404}, note={arXiv:2410.09132}}
@inproceedings{zhu2025mosaic,
  title={Mosaic of Modalities: A Comprehensive Benchmark for Multimodal Graph Learning}, author={Zhu, Jing and Zhou, Yuhang and Qian, Shengyi and He, Zhongmou and Zhao, Tong and Shah, Neil and Koutra, Danai}, booktitle={CVPR}, year={2025}, note={arXiv:2406.16321 (= MM-Graph)}}

% ===== D LLM 基准污染 =====
@inproceedings{wu2025antileak,
  title={AntiLeakBench: Preventing Data Contamination by Automatically Constructing Benchmarks with Updated Real-World Knowledge}, author={Wu, Xiaobao and Pan, Liangming and Xie, Yuxi and others}, booktitle={ACL}, year={2025}, note={arXiv:2412.13670}}
@article{xu2024bdc,
  title={Benchmark Data Contamination of Large Language Models: A Survey}, author={Xu, Cheng and Guan, Shuhao and Greene, Derek and Kechadi, M-Tahar}, journal={arXiv:2406.04244}, year={2024}}
@inproceedings{sun2025emperor,
  title={The Emperor's New Clothes in Benchmarking? A Rigorous Examination of Mitigation Strategies for LLM Benchmark Data Contamination}, author={Sun, Yifan and Wang, Han and Li, Dongbai and Wang, Gang and Zhang, Huan}, booktitle={ICML}, year={2025}, note={arXiv:2503.16402}}
@article{joeres2025datasail,
  title={Data Splitting to Avoid Information Leakage with DataSAIL}, author={Joeres, Roman and Blumenthal, David B. and Kalinina, Olga V.}, journal={Nature Communications}, volume={16}, pages={3337}, year={2025}, doi={10.1038/s41467-025-58606-8}}

% ===== E 图结构性泄露（非多模态） =====
@inproceedings{platonov2023critical,
  title={A Critical Look at the Evaluation of {GNNs} Under Heterophily: Are We Really Making Progress?}, author={Platonov, Oleg and Kuznedelev, Denis and Diskin, Michael and Babenko, Artem and Prokhorenkova, Liudmila}, booktitle={ICLR}, year={2023}, note={arXiv:2302.11640; 旧版误作 NeurIPS 2023，已修正; 同组 TPAMI 2024 扩展待最终复核}}
@inproceedings{zhu2024pitfalls,
  title={Pitfalls in Link Prediction with Graph Neural Networks}, author={Zhu, Yanqiao and others}, booktitle={WSDM}, year={2024}, note={arXiv:2306.00899}}
@inproceedings{li2023lpeval,
  title={Evaluating {GNNs} for Link Prediction: Current Pitfalls and New Benchmarking}, author={Li, Juanhui and others}, booktitle={NeurIPS Datasets and Benchmarks}, year={2023}}
@article{briere2025leakage,
  title={Benchmarking Data Leakage on Link Prediction in Biomedical Knowledge Graph Embeddings}, author={Bri{\`e}re, others}, journal={bioRxiv}, year={2025}, doi={10.1101/2025.01.23.634511}}

% ===== F 数据集文档/质量 (NEW) =====
@article{gebru2021datasheets,
  title={Datasheets for Datasets}, author={Gebru, Timnit and Morgenstern, Jamie and Vecchione, Briana and Vaughan, Jennifer Wortman and Wallach, Hanna and Daum{\'e} III, Hal and Crawford, Kate}, journal={Communications of the ACM}, volume={64}, number={12}, pages={86--92}, year={2021}, doi={10.1145/3458723}, note={arXiv:1803.09010}}
@inproceedings{holland2018nutrition,
  title={The Dataset Nutrition Label: A Framework To Drive Higher Data Quality Standards}, author={Holland, Sarah and Hosny, Ahmed and Newman, Sarah and Joseph, Joseph and Chmielinski, Kasia}, booktitle={NeurIPS Workshop on Trustworthy ML}, year={2018}, note={arXiv:1805.03677}}
@inproceedings{pushkarna2022datacards,
  title={Data Cards: Purposeful and Transparent Dataset Documentation for Responsible AI}, author={Pushkarna, Mahima and Zaldivar, Andrew and Kjartansson, Oddur}, booktitle={FAccT}, year={2022}, doi={10.1145/3531146.3533231}, note={arXiv:2204.01075}}
@inproceedings{sambasivan2021datacascades,
  title={``Everyone wants to do the model work, not the data work'': Data Cascades in High-Stakes AI}, author={Sambasivan, Nithya and Kapania, Shivani and Highfill, Hannah and Akrong, Diana and Paritosh, Praveen and Aroyo, Lora M.}, booktitle={CHI}, year={2021}, doi={10.1145/3411764.3445518}, note={arXiv:2103.09291}}
@inproceedings{northcutt2021pervasive,
  title={Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks}, author={Northcutt, Curtis G. and Athalye, Anish and Mueller, Jonas}, booktitle={NeurIPS Datasets and Benchmarks}, year={2021}, note={arXiv:2103.14749}}
@article{damour2022underspec,
  title={Underspecification Presents Challenges for Credibility in Modern Machine Learning}, author={D'Amour, Alexander and Heller, Katherine and Moldovan, Dan and others}, journal={JMLR}, volume={23}, number={226}, pages={1--61}, year={2022}, note={arXiv:2011.03395}}
@inproceedings{recht2019imagenet,
  title={Do ImageNet Classifiers Generalize to ImageNet?}, author={Recht, Benjamin and Roelofs, Rebecca and Schmidt, Ludwig and Shankar, Vaishaal}, booktitle={ICML}, year={2019}, note={arXiv:1902.10811}}

% ===== G 单模态图 benchmark 族 (NEW) =====
@inproceedings{hu2020ogb,
  title={Open Graph Benchmark: Datasets for Machine Learning on Graphs}, author={Hu, Weihua and Fey, Matthias and Zitnik, Marinka and Dong, Yuxiao and Ren, Hongyu and Liu, Bowen and Catasta, Michele and Leskovec, Jure}, booktitle={NeurIPS Datasets and Benchmarks}, year={2020}, note={arXiv:2005.00687}}
@inproceedings{hu2021ogblsc,
  title={OGB-LSC: A Large-Scale Challenge for Machine Learning on Graphs}, author={Hu, Weihua and Fey, Matthias and Ren, Hongyu and Nakada, Makoto and Glescher, Matthias and Dong, Yuxiao and Leskovec, Jure}, booktitle={NeurIPS Datasets and Benchmarks}, year={2021}, note={arXiv:2103.09430}}
@article{shchur2018pitfalls,
  title={Pitfalls of Graph Neural Network Evaluation}, author={Shchur, Oleksandr and Mumme, Maximilian and Bojchevski, Aleksandar and G{\"u}nnemann, Stephan}, journal={arXiv:1811.05868}, year={2018}, note={NeurIPS 2018 RRL Workshop}}
@article{dwivedi2023benchmarking,
  title={Benchmarking Graph Neural Networks}, author={Dwivedi, Vijay Prakash and Joshi, Chaitanya K. and Luu, Anh Tuan and Laurent, Thomas and Bengio, Yoshua and Bresson, Xavier}, journal={JMLR}, volume={24}, pages={22--0567}, year={2023}, note={arXiv:2003.00982}}
@inproceedings{lim2021nonhomophilous,
  title={Large Scale Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple Methods}, author={Lim, Derek and Hohne, Felix and Li, Xiuyu and Huang, Sijia Linda and Gupta, Vaishnavi and Bhalerao, Omkar and Lim, Ser-Nam}, booktitle={NeurIPS}, year={2021}, note={arXiv:2110.14446}}
@article{zheng2026heterophily,
  title={Graph Neural Networks for Graphs with Heterophily: A Survey}, author={Zheng, Xin and Wang, Yi and Liu, Yixin and Li, Ming and Zhang, Miao and Jin, Di and Yu, Philip S. and Pan, Shirui}, journal={IEEE TKDE}, year={2026}, doi={10.1109/TKDE.2026.3680353}, note={arXiv:2202.07082; 非 Wei 组}}

% ===== H 可复现/声明式基建 (NEW) =====
@article{pineau2021reproducibility,
  title={Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)}, author={Pineau, Joelle and Vincent-Lamarre, Philippe and Sinha, Koustuv and Larivi{\`e}re, Vincent and Beygelzimer, Alina and D'Alch{\'e}-Buc, Florence and Fox, Emily and Larochelle, Hugo}, journal={JMLR}, volume={22}, pages={1--26}, year={2021}, note={arXiv:2003.12206}}
@article{koster2012snakemake,
  title={Snakemake---a scalable bioinformatics workflow engine}, author={K{\"o}ster, Johannes and Rahmann, Sven}, journal={Bioinformatics}, volume={28}, number={19}, pages={2520--2522}, year={2012}, doi={10.1093/bioinformatics/bts480}}
@article{ditommaso2017nextflow,
  title={Nextflow enables reproducible computational workflows}, author={Di Tommaso, Paolo and Chatzou, Maria and Floden, Evan W. and Prieto Barja, Pablo and Palumbo, Evan and Notredame, Cedric}, journal={Nature Biotechnology}, volume={35}, pages={316--319}, year={2017}, doi={10.1038/nbt.3820}}

% ===== I CLIP/VLM 记忆与污染 (NEW) =====
@inproceedings{carlini2021extracting,
  title={Extracting Training Data from Large Language Models}, author={Carlini, Nicholas and Tram{\`e}r, Florian and Wallace, Eric and others}, booktitle={USENIX Security Symposium}, year={2021}, note={arXiv:2012.07805}}
@inproceedings{carlini2023diffusion,
  title={Extracting Training Data from Diffusion Models}, author={Carlini, Nicholas and Hayes, Jamie and Nasr, Milad and Jagielski, Matthew and Shmatikov, Vitaly and Nita-Rotaru, Cristina and Tram{\`e}r, Florian}, booktitle={USENIX Security Symposium}, year={2023}, note={arXiv:2301.13188}}
@inproceedings{schuhmann2022laion5b,
  title={LAION-5B: An open large-scale dataset for training next generation image-text models}, author={Schuhmann, Christoph and others}, booktitle={NeurIPS}, year={2022}, note={arXiv:2210.08402}}
@inproceedings{xu2024metaclip,
  title={Demystifying CLIP Data}, author={Xu, Hu and Xie, Saining and Tan, Xiaoqing Ellen and others}, booktitle={ICLR}, year={2024}, note={arXiv:2309.16671 (MetaCLIP)}}

% ===== J 合成 vs 真实数据 (NEW) =====
@article{shumailov2024collapse,
  title={AI models collapse when trained on recursively generated data}, author={Shumailov, Ilia and Shumaylov, Zakhar and Zhao, Yiren and Gal, Yarin and Papandreou, Papandreou and others}, journal={Nature}, volume={631}, pages={755--759}, year={2024}, doi={10.1038/s41586-024-07566-y}, note={arXiv:2305.17493}}
@inproceedings{gadre2023datacomp,
  title={DataComp: In search of the next generation of multimodal datasets}, author={Gadre, Samir Yitzhak and Ilharco, Gabriel and Fang, Alex and others}, booktitle={NeurIPS Datasets and Benchmarks}, year={2023}, note={arXiv:2304.14108}}

% ===== K Zhewei Wei 课题组 (NEW; 用户指定重点核查) =====
@inproceedings{yi2025tgbseq,
  title={TGB-Seq Benchmark: Challenging Temporal GNNs with Complex Sequential Dynamics}, author={Yi, Lu and Peng, Jie and Zheng, Yanping and Mo, Fengran and Wei, Zhewei and Ye, Yuhang and Yue, Zixuan and Huang, Zengfeng}, booktitle={ICLR}, year={2025}, note={arXiv:2502.02975}}
@inproceedings{ji2025gag,
  title={LLM-Based Multi-Agent Systems are Scalable Graph Generative Models}, author={Ji, Jiarui and Lei, Runlin and Bi, Jialing and Wei, Zhewei and Chen, Xu and Lin, Yankai and Pan, Xuchen and Li, Yaliang and Ding, Bolin}, booktitle={Findings of ACL}, year={2025}, note={arXiv:2410.09824 (GAG)}}
@inproceedings{peng2026gdgb,
  title={{GDGB}: A Benchmark for Generative Dynamic Text-Attributed Graph Learning}, author={Peng, Jie and Ji, Jiarui and Lei, Runlin and Wei, Zhewei and Liu, Yongchao and Hong, Chuntao}, booktitle={ICLR}, year={2026}, note={arXiv:2507.03267}}
@article{he2026rethinkinglp,
  title={Rethinking Link Prediction for Directed Graphs}, author={He, Mingguo and Guo, Yuhe and Zheng, Yanping and Wei, Zhewei and G{\"u}nnemann, Stephan and Xiao, Xiaokui}, journal={IEEE TPAMI}, year={2026}, note={arXiv:2502.05724}}
@inproceedings{lei2026tagrobustness,
  title={Robustness in Text-Attributed Graph Learning: Insights, Trade-offs, and New Defenses}, author={Lei, Runlin and Yi, Lu and He, Mingguo and Qiu, Pengyu and Wei, Zhewei and Liu, Yongchao and Hong, Chuntao}, booktitle={ICLR}, year={2026}, note={arXiv:2510.17185}}
@inproceedings{chen2024polygcl,
  title={{PolyGCL}: Graph Contrastive Learning via Learnable Spectral Polynomial Filters}, author={Chen, Jingyu and Lei, Runlin and Wei, Zhewei}, booktitle={ICLR (Spotlight)}, year={2024}, note={OpenForum:y21ZO6M86t; 无 arXiv}}
@inproceedings{peng2024beyond,
  title={Beyond Over-smoothing: Uncovering the Trainability Challenges in Deep Graph Neural Networks}, author={Peng, Jie and Lei, Runlin and Wei, Zhewei}, booktitle={CIKM}, year={2024}}
```

---

## 7. 诚实的 gap（别人更强的地方，必须承认）

1. **广度不如 OpenMAG**（19/6/16/24/8）：定位 = **「少而精 + 防泄露 + 质量门 + 可控」**互补型，不抢广度王。
2. **不是首个 MAG benchmark**：MAGB/Mosaic/OpenMAG 在先。补的是 agent 构建 + MAG 去泄露 + 报告卡，**不是 MAG benchmark 本身**。
3. **agent 通用性不如 BenchAgents**：scope agent 价值到「自治 schema 发现 + 端到端编排 + 真实源获取 + 可复用接口」。
4. **图结构性泄露非首创**：Platonov/TGB-Seq 等先做；新颖性在「MAG + LP-ceiling 量化 + 多模态 text 泄露 + 报告卡 + 统一协议」。
5. **Wei 课题组的 TAG 线（GDGB/TGRB）比本文 text 维度更深**：本文补**多模态**（image+formula）与**去泄露**，不与 TAG 深度争。
6. **encoder frozen + per-domain profiling**：scope out trainable（诚实），提供 per-domain profiling（clip_large 赢 arxiv 长文档，siglip 赢 steam 短文本）。
7. **agent 在去泄露决策上不 load-bearing**（greedy=oracle）：红线，**绝不**夸成「agent 防泄露/建图决策比规则强」。
8. **报告卡是质量尺非质量保证**：label 发现仍 88–94%（Exp6），报告卡暴露问题、不消灭问题。

---

## 8. 红线对齐 + 写作执行清单

**写作红线（与逻辑线一致，不可违反）：**
1. **去泄露 = 正确性属性**，措辞删「独有护城河」；改「benchmark 正确性属性 + 外部可证（E1/Exp5b）」。
2. **steering 降 appendix**：Related Work 正文**不放** r=0.846 knob 钩子；§2.6 一句「机制演示见 appendix」足矣。
3. **不声称 agent 决策 > 规则**（greedy=oracle）：agent 价值只 scope 到自动化/schema 发现。
4. **不声称全域图有用/多模态必需**（红线④）：用报告卡 + 同质性谱诚实标注退化域。
5. **valid≠good 是新主张，但别过头**：报告卡是质量尺，非质量保证。

**新杀手锏三连（写进 Intro / §2.8）：**
1. **范式钩**：*"Generation agents synthesize instances whose leakage is semantic and unstrippable; we orchestrate real sources and strip leakage at the field level — measured (E3)."*
2. **质量钩**：*"A benchmark that passes validity and leakage checks is valid, not necessarily good — we auto-generate a report card (Exp6: an 843-class label passed validity)."*
3. **正确性钩**：*"The field ships structural leakage even in gold-standard public benchmarks (E1) and published real MAGs (Exp5b: OpenMAG +0.50) — we decontaminate and prove it externally."*

**核查修订记录（2026-08-06）：**

| 项 | 旧 | 新（本次） | 依据 |
|---|---|---|---|
| Platonov venue | NeurIPS 2023 (+TPAMI'24) | **ICLR 2023**（TPAMI 扩展待复核） | OpenReview tJbbQfw-5wv / arXiv 2302.11640 |
| OpenMAG venue | arXiv 2602.05576 | **ICML 2026** | arXiv + 核查 |
| Shchur 标题 | "Pitfalls **in**" | "Pitfalls **of**" | arXiv 1811.05868 |
| Wei 组 MAGB/MM-Graph | （疑似关联） | **均非 Wei 组** | DBLP + 联网 |
| heterophily survey | （误挂 Wei） | **Zheng/Pan, TKDE 2026** | arXiv 2202.07082 |
| 新增 6 群 + Wei 组 | — | A2/F/G/H/I/J + K | 本文 §1 |
| 旧 steering 钩 | 章末头条 | **弃用**（降 appendix） | 07-31 thesis |

---

## 9. 与其他文档的关系

| 文档 | 关系 |
|---|---|
| `paper_logic_DnB_FINAL_2026-07-11.md`（07-31 thesis 重定版） | 逻辑线权威；本文执行其 §2 Related 定位 |
| `experiment_results_FINAL_decontaminated_2026-07-16.md` | 数字总表权威；本文所有数字以它为准 |
| `related_work_FINAL_2026-07-16.md` | 🔴 **已删除，被本文取代**（旧 thesis 产物） |
| `related_work_agent_dataset_construction_2026-07-13.md` | 🔴 **已删除**（旧调研底稿，被本文取代） |

---

*2026-08-06 重写：对齐 07-31「开源自治框架」hero + 5 链环；新增 6 对手群（A2/F/G/H/I/J）+ Zhewei Wei 课题组（§4）；修正 Platonov→ICLR'23、OpenMAG→ICML'26；弃旧 steering 钩。所有引文经联网逐篇核查（5 个并行验证 agent）。*
