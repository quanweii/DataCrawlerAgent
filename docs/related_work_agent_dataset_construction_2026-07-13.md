# 相关工作调研：多智能体构建数据集 — 我们的论文优势在哪（2026-07-13）

> 任务：找"也用多智能体/LLM-agent 构建数据集或 benchmark"的工作，定位我们论文（DataCrawlerAgent / MAG-Forge）的优势。
> 调研覆盖四条相关轴：① agent-built dataset/benchmark ② agent-built graph（KG 构造）③ multimodal graph benchmark ④ benchmark leakage。
> 结论先行：**没有任何一个工作同时覆盖"多智能体构造 + 多模态属性图 benchmark + 结构性防泄露协议 + 声明式可复用框架"这四点**——每个相关工作只占其中一条轴。这是我们清晰、可防守的定位。

---

## 0. 一句话定位

> **我们是唯一一个"用多智能体 + 声明式 skill 构造多模态属性图 benchmark，并自带结构性防泄露协议（LP-ceiling / 跨切分边剪枝）"的工作。**
> 最接近的两类——agent-built benchmark（BenchAgents/DatasetAgent）和 multimodal graph benchmark（OpenMAG/MAGB）——一个有 agent 没有图与防泄露，一个有图与多模态但手工建造且不防泄露。

---

## 1. 相关工作全景（4 个 cluster）

### Cluster A — Agent-built dataset / benchmark（最直接对标"多智能体建数据集"）

| 工作 | 年份/会议 | 做了什么 | 缺什么（相对我们）|
|---|---|---|---|
| **BenchAgents** (Butt et al.) | 2024 / ICLR-2025 DATA-FM workshop, 23 引 | 4 agent（Planning/Generation/Verification/Evaluation）自动造 NLP benchmark（日历调度、约束文本生成）| ① 造的是**合成文本实例**，无图结构 ② Verification 是 clarity/feasibility 质量检查，**非结构性泄露检测** ③ 重度依赖 human-in-the-loop 编辑（其 Table 4 显示 DIL 大量修改）④ 无多模态、无属性图 |
| **DatasetAgent** (Sun et al.) | 2025 / arXiv:2507.08648 | 4 个 MLLM agent + 图像工具，从真实图像自动造**图像**数据集（分类/检测/分割）| ① 纯图像，**无图结构** ② 无泄露协议 ③ 无 benchmark 评测广度 |
| **AutoBencher** (Li et al.) | ICLR 2025, 28–30 引 | LLM 自动生成 QA benchmark（salient/novel/difficult）| ① QA/文本，**无图** ② "declarative" 指对 QA 主题的声明式 spec，非"1 YAML=1 图域"的可复用骨架 ③ 无泄露 ④ 无多模态 |
| **CAMEL** (Li et al.) | NeurIPS 2023 | 角色扮演双 agent 生成**对话训练数据** | 合成对话，非结构化 benchmark，无图无泄露 |
| **Benchmark Self-Evolving** (Wang et al.) | ACL 2025 | 多 agent 动态造 LLM eval benchmark | 动态 eval，非图 benchmark |
| **AgentInstruct** (Mitra et al.) | 2024 | agent flow 造教学数据 | 训练数据，非 benchmark，无图 |

### Cluster B — Agent-built graph / KG construction（最接近"agent 建图"）

| 工作 | 年份/会议 | 做了什么 | 缺什么 |
|---|---|---|---|
| **KARMA** | NeurIPS 2025, 53 引, arXiv:2502.06472 | 多 agent LLM 从文献自动做**知识图谱充实**（实体/关系抽取）| ① 是知识图谱（实体-关系三元组），**非属性图 benchmark** ② 不发布 benchmark ③ 无泄露协议 ④ 非多模态属性图 |
| **LLM-empowered KG Construction (Survey)** | 2025, arXiv:2510.20345, 25 引 | 综述 LLM 做 KG 构造管线 | 综述，确认"多模态属性图 + agent"是空白 |
| Neo4j LLM Graph Builder | 2025 工业工具 | 文本→图构造 | 工业工具，非学术 benchmark，无泄露/无多模态属性图 |

### Cluster C — Multimodal graph benchmark（我们造的"东西"这一轴）

| 工作 | 年份/会议 | 做了什么 | 缺什么 |
|---|---|---|---|
| **OpenMAG** (Wan et al.) | 2026, arXiv:2602.05576, 7 引 | 19 数据集/6 域/16 encoder/24 模型，广度王 | ① **无防泄露协议** ② 手工建造，**无可复用 agent/skill 框架** ③ 无声明式建图 |
| **MAGB** (Yan et al.) | WWW 2025, arXiv:2410.09132, 13 引 | 首个综合 MAG benchmark（text+visual）| 同上：无防泄露、手工建造 |
| **MM-Graph** | 2024 | 多模态图 benchmark（5 数据集）| 同上 |
| **Mosaic of Modalities** | CVPR 2025 | 7 数据集多模态图 | 同上 |

### Cluster D — Benchmark leakage / data contamination（我们"防泄露"这一轴）

| 工作 | 年份/会议 | 做了什么 | 缺什么 |
|---|---|---|---|
| **AntiLeak-Bench** | 2024, 40 引 | 用更新知识自动造 benchmark 防 contamination | ① 防的是 **test-in-train 污染**（LLM），**非图的结构性泄露**（跨切分边/LP-ceiling）② 非图、非多模态属性图 |
| **Benchmark Data Contamination Survey** | 2024, arXiv:2406.04244 | BDC 综述 | 确认 leakage 研究集中在 LLM contamination，图结构性泄露是空白 |
| **DataSAIL** | 2024 (PMC) | 分割算法最小化 inter-split 信息泄露 | 通用分割，未针对图属性图 benchmark + 无 LP-ceiling 量化 |
| **Emperor's New Clothes** | ICML 2025 | BDC 缓解策略审查 | LLM contamination，非图 |

---

## 2. 核心对比表（可直接进论文 / PPT）

| 能力维度 | BenchAgents | DatasetAgent | AutoBencher | KARMA | OpenMAG / MAGB | AntiLeak-Bench | **Ours (MAG-Forge)** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 多智能体建造 | ✅ 4 agent | ✅ 4 agent | 🟡 单 LLM | ✅ 多 agent | ❌ 手工 | 🟡 自动 | ✅ 多 agent + skill |
| **多模态属性图** (MAG) | ❌ 文本 | 🟡 仅图像 | ❌ QA | ❌ 知识图谱 | ✅ | ❌ | ✅ |
| **结构性防泄露** (LP-ceiling/跨切分剪枝) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌¹ | ✅ |
| 声明式可复用 (1 YAML=1 域) | ❌ 需 DIL 重编辑 | ❌ | 🟡 QA-spec | ❌ | ❌ | ❌ | ✅ 94.6% 复用, spotify 零代码 |
| 防泄露量化 (leak-profile 元数据) | ❌ | ❌ | ❌ | ❌ | ❌ | 🟡 contamination | ✅ 每域一等元数据 |
| 因果结构实验 (steering) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ r=0.846 |
| Benchmark 评测广度 (NC/LP/聚类/few-shot/…) | ❌ | ❌ | ❌ | ❌ | ✅ 最广 | ❌ | ✅ |
| 公开 benchmark 资产 | 🟡 2 个 NLP | 🟡 图像 | 🟡 QA | ❌ | ✅ 多 | 🟡 | ✅ 4+1 域 |

¹ AntiLeak-Bench 防的是 LLM test-in-train 污染，**与图的跨切分边结构性泄露是不同机制**——这是我们独有的护城河。

---

## 3. 我们的 5 个独特优势（带证据，可直接写进论文）

### 优势 1：唯一"多智能体 + 声明式 skill 构造多模态属性图 benchmark"
- **对标**：BenchAgents/DatasetAgent 有 agent 但造文本/图像（无图结构）；KARMA 有 agent 但造知识图谱（非属性图 benchmark，不发布）；OpenMAG/MAGB 造 MAG 但手工、无 agent。
- **证据**：4 个 `*_multimodal_graph_agent.json`（字面=agent 产物）+ formula 独有模态 + 同质性谱 0.24→0.93。

### 优势 2：结构性防泄露协议是图 benchmark 里独有的护城河
- **对标**：整个 leakage 文献（AntiLeak-Bench / BDC Survey / Emperor's New Clothes）研究的是 **LLM 的 test-in-train 污染**；**没人做图的"跨切分边结构性泄露"**。OpenMAG/MAGB 不报告也不剪这种泄露。
- **证据**：LP-ceiling 坍塌 wikiart 0.921→0.024，4 域剪枝后全 ≤ 多数类；每域附 leak-profile 一等元数据。
- **关键论点**：我们顺带指出 OpenMAG/MAGB 报告的精度里**很可能带着这种结构泄露**——这是个可写的、对领域有贡献的发现。

### 优势 3：去泄露靠测量，不靠 LLM（methodological integrity，反 BenchAgents）
- **对标**：BenchAgents 的核心卖点是"agent 是 benchmark 质量的 load-bearing 环节"。我们用消融证明：去泄露决策靠 LP-ceiling/homophily **测量**，**greedy(无LLM)=oracle**。
- **论点**：我们不跟 BenchAgents 抢"agent 多能干"，反而把这个**当成优点**——协议 principled+robust，不依赖昂贵不稳定的 LLM 决策。这是诚实，也是差异化。

### 优势 4：可审计的声明式可复用（94.6% 复用 + spotify 零代码）vs 别人需重度人工编辑
- **对标**：BenchAgents 的 Table 4 显示它需要 developer-in-the-loop 大量编辑（normalized Levenshtein 0.14–0.41）；AutoBencher 的"declarative"是对 QA 主题，不是可复用建图骨架。
- **证据**：可审计 code-reuse manifest（每个文件列 SLOC）；spotify 41 行 YAML、0 Python LOC 建出 10k 节点真图——**比"agent 能造 benchmark"更强的可复用性证据**。

### 优势 5：steering 因果实验——没人做过的结构×同质性因果隔离
- **对标**：所有相关工作都没做"同节点同特征只换边"的因果对照。
- **证据**：GCN−MLP gap vs 同质性 r=0.846 [0.60,0.96], p=0.0005, n=12，MLP 跨关系 bit-identical 自检。
- **论点**：这是 benchmark 本身的**科学发现**（structure×homophily 耦合律），不只是"我们造了个数据集"。

---

## 4. 诚实的 gap（别人更强的地方，写论文要承认）

1. **广度不如 OpenMAG/MAGB**：他们 19/6 域、24 模型、8 任务；我们 4+1 域、10 模型。定位是"少而精 + 防泄露"，不是广度王。
2. **agent 通用性不如 BenchAgents**：BenchAgents 的 4-agent 分解（plan/gen/verify/eval）对任意 NLP benchmark 通用；我们的 agent 贡献更窄（语义 wrangling + 可复用接口 + 编排），诚实 scope 到这三点。
3. **KG 构造成熟度不如 KARMA**：KARMA 在"agent 抽取知识建图"上更深；我们不声称做 KG，只做 co-attribute 属性图。
4. **encoder 只做 frozen**：OpenMAG 做 frozen+trainable；我们 scope out trainable（诚实）。
5. **agent 在去泄露决策上不是 load-bearing**：这是诚实红线，不能夸大成"agent 防泄露比规则强"。

---

## 5. Related Work 章节怎么写（给现成骨架）

建议分 4 小节，每节先述、再定位差异：

> **§2.1 Multimodal Attributed Graph Benchmarks.** MAGB [Yan et al. 2025] 首次系统化 MAG benchmark；OpenMAG [Wan et al. 2026] 扩到 19 数据集/16 encoder，是广度基准。**两者均为手工建造，且不报告或剪除跨切分边导致的结构性泄露**——这正是我们 C2 协议的切入点。
>
> **§2.2 LLM-Agent Pipelines for Dataset/Benchmark Construction.** BenchAgents [Butt et al. 2024] 用 4 agent（plan/gen/verify/eval）自动造 NLP benchmark；DatasetAgent [Sun et al. 2025] 用 MLLM agent 造图像集；AutoBencher [Li et al. 2025] 声明式造 QA benchmark；CAMEL [Li et al. 2023] 角色扮演生成对话数据。**这些工作均面向文本/图像 QA，不涉及图结构、不做结构性防泄露、且（如 BenchAgents）依赖重度人工编辑**。我们把 agent 用在多模态**属性图**构造上，并用声明式 skill 把单域增量降到 ~280 LOC（94.6% 复用）。
>
> **§2.3 LLM-Agent Knowledge Graph Construction.** KARMA [2025] 等用多 agent 做知识图谱充实。**KG（实体-关系三元组）与我们造的属性图（co-attribute 关系）图族不同**，且这类工作不发布 benchmark、不做防泄露。
>
> **§2.4 Benchmark Leakage & Data Contamination.** AntiLeak-Bench、BDC survey [2024]、Emperor's New Clothes [ICML 2025] 研究的是 **LLM 的 test-in-train 污染**。**图的"跨切分边结构性泄露"（LP-ceiling）是不同机制，据我们所知尚无 MAG benchmark 系统量化并剪除它**——我们的 leak-profile 一等元数据填补此空白。
>
> **定位句（章末）**：综上，我们是首个**同时**满足（i）多智能体 + 声明式可复用构造、（ii）多模态属性图 benchmark、（iii）结构性防泄露协议、（iv）因果结构实验的工作。

---

## 6. 参考文献（核心）

- BenchAgents — Butt et al., 2024, arXiv:2410.22584 (ICLR-2025 DATA-FM). https://arxiv.org/abs/2410.22584
- DatasetAgent — Sun et al., 2025, arXiv:2507.08648. https://arxiv.org/abs/2507.08648
- AutoBencher — Li et al., ICLR 2025, arXiv:2407.08351. https://openreview.net/forum?id=ymt4crbbXh
- CAMEL — Li et al., NeurIPS 2023, arXiv:2303.17760. https://arxiv.org/abs/2303.17760
- Benchmark Self-Evolving — Wang et al., 2024, arXiv:2402.11443.
- AgentInstruct — Mitra et al., 2024, arXiv:2407.03502.
- KARMA — 2025, arXiv:2502.06472 (NeurIPS 2025). https://arxiv.org/html/2502.06472v1
- LLM-empowered KG Construction Survey — Bian et al., 2025, arXiv:2510.20345.
- OpenMAG — Wan et al., 2026, arXiv:2602.05576. https://arxiv.org/abs/2602.05576
- MAGB — Yan et al., WWW 2025, arXiv:2410.09132. https://arxiv.org/abs/2410.09132
- MM-Graph — 2024, https://mm-graph-benchmark.github.io/
- Mosaic of Modalities — Zhu et al., CVPR 2025.
- AntiLeak-Bench — 2024 (Semantic Scholar).
- Benchmark Data Contamination Survey — 2024, arXiv:2406.04244. https://arxiv.org/html/2406.04244v1
- Emperor's New Clothes in Benchmarking — ICML 2025. https://openreview.net/forum?id=TuvDxubEfE
- DataSAIL — 2024, PMC11978981.

---

*所有相关工作均为公开可查；本调研用于论文 §2 Related Work 与导师汇报的"对照 OpenMAG / 优势"页。*
