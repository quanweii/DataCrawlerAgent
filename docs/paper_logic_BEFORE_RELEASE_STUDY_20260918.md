# MAG-Forge 论文逻辑线（FINAL CLEAN，2026-09-15）

> 本文件是论文叙事的唯一权威来源；核心概念统一为 provenance-confirmed defect、protocol condition 与 diagnostic signal。

## 1. 一句话 Thesis

MAG-Forge 是一个面向真实多模态来源的自治多智能体框架：它自动发现或确认 schema、构造多模态属性图、审计可验证的完整性缺陷、生成有真实 provenance 支持的部署协议视图，并发布带 integrity card 的可复现 benchmark。

论文卖点不是“删边让分数下降”，而是：

1. 把真实来源自动转化为可用 MAG；
2. 把数据正确性建立在 provenance 和明确协议上；
3. 同一数据集同时发布多种真实部署假设下的 benchmark，而不是只给一个容易误读的随机切分分数；
4. 用结构化报告卡公开每个数据集的适用范围和失败边界。

## 2. 六模块最终框架

1. **Real-world Multimodal Sources**
2. **Autonomous Schema Discovery**
3. **Schema-Guided MAG Construction**
4. **Protocol-Aware Integrity Auditing**
5. **Integrity-Guided Representation Encoding**
6. **Protocol-Conditioned Benchmarking & Integrity Reporting**

### 2.1 模块 1：Real-world Multimodal Sources

输入是论文、游戏、电影和艺术品的真实文本、图片、公式、属性与关系，不用 LLM 合成 benchmark 样本。四个成熟域是 arxiv、steam、tmdb、wikiart；spotify/amazon/reddit 只作为复用或 held-out 证据，不混入四域主表。

### 2.2 模块 2：Autonomous Schema Discovery

生产流水线中 `SchemaDiscoveryAgent` 是 Collector 与 Builder 之间的强制质量门。成熟域使用 YAML schema 作为待确认 proposal；新域可让 LLM 根据字段 profile 提议 label、关系与采集 schema。两种模式最终都由确定性 validator 检查 ID 唯一性、label 覆盖与类别退化、关系字段有效性，以及关系是否与 label 字段同源。未经验证的 proposal 不会发布给 Builder。LLM 的价值是减少人工 schema 编写与编排成本，不声称其微观关系选择优于所有规则方法。

### 2.3 模块 3：Schema-Guided MAG Construction

Collector 加载并校验真实条目；Builder 按已确认 schema 构图并控制密度；Formatter 输出节点、边、模态路径、metadata 和 provenance。raw MAG 是不可变原始产物，后续审计产生 derived views，不直接覆盖 raw graph。

### 2.4 模块 4：Protocol-Aware Integrity Auditing

模块 4 是“体检与协议生成”，不是看到高同质或跨 split 边就删。

真正可以判为 integrity defect 的情况：

- feature/relation 经字段 lineage 证实由 target 直接派生；
- 重复实体或近重复内容跨 split；
- temporal 训练阶段使用了未来信息；
- group-inductive 的同一声明 group 跨 split；
- 实际数据暴露违反论文声明的 protocol。

下列现象只能作为诊断，不能单独叫 leakage：

- transductive 图中存在 train–test 普通边；
- LP/topology-only probe 高于 majority；
- relation homophily 很高；
- 完整图比 split-disjoint 图表现更好。

审计后为每个数据集物化四个派生视图：

| Protocol | Node split | Edge exposure | 研究问题 |
|---|---|---|---|
| Transductive | stratified 60/20/20 | 完整 sanitized topology | 标准转导条件下的上限 |
| Split-disjoint | 与 transductive 相同 masks | 仅 split 内部边 | 对跨 split 结构暴露的敏感性 |
| Temporal-inductive | 按真实时间排序，同一时间桶不拆 | 仅时间段内部边 | 面向未来数据的稳健性 |
| Group-inductive | 真实 primary group 整体分配 | 仅 split 内部边 | 面向未见实体组的稳健性 |

四域 provenance：arxiv=`arXiv ID 月份 + 首作者`；steam=`release_date + primary developer`；tmdb=`release_date + primary director`；wikiart=`year + artist`。

### 2.5 模块 5：Integrity-Guided Representation Encoding

先删除 provenance-confirmed target-derived 输入，再进行编码。当前 text/image/formula encoder 冻结，逐模态 L2 normalize 后做 element-wise mean fusion，再归一化。因为冻结编码与 mean fusion 没有从 split 学习参数，同一节点表示可以安全复用于四个 protocol。

如果未来加入 PCA、数据标准化、learned fusion、adapter 或其他 fitted state，必须在每个 protocol 中只用 train nodes 拟合，再作用于 val/test。

Encoder provenance 必须记录 checkpoint 和预训练来源。CLIP-vs-ImageNet 的性能差只能说明 encoder sensitivity；只有 sample-level overlap 检索加 clean/overlap performance gap 才能支持预训练污染结论。

### 2.6 模块 6：Protocol-Conditioned Benchmarking & Integrity Reporting

主实验现为完整的四协议六任务矩阵：node classification、link prediction、node clustering、modality retrieval、modality alignment 与 modality matching。所有任务消费同一批 sanitized views，但任务边界不同：LP 的消息图只含训练正边；聚类表示由 label-free graph-autoencoder 学习，KMeans 在 train embeddings 拟合、test 评分；retrieval/alignment 只改变评估 cohort；matching 的 train/val/test 实体完全隔离。不能机械地把 NC 的 edge exposure 规则套到 node-local 模态任务，也不能用 test NMI 挑聚类模型。

六任务的定位是验证 benchmark 产物可被多类任务一致消费，并刻画 applicability；它不把性能分数当作构图正确性的证明。正确性仍由 provenance、重复/实体检查、raw/derived 分离和 protocol invariants 决定。

每域 integrity card 同时包含 correctness 与 applicability：前者检查 provenance/duplicate/protocol invariants；后者报告饱和度、图价值、模态价值、时间/group coverage 和 open-set 风险。

## 3. 论文贡献

### C1：真实来源上的自治 MAG 构造框架

框架把字段 profiling、schema discovery、采集、构图、审计、编码与 benchmark 串成可执行流水线。E2 显示 schema discovery/audit 阶段约 1–2 次 LLM 调用、约 1.1k–3.0k tokens、约 1.4–4.3 秒；人工 25–90 分钟仅为专家估计，不作为已验证 user-study 结论。

### C2：基于 provenance 的完整性审计与四协议发布

正确性来自确定性 lineage/duplicate/protocol checks，而不是 LLM 自我声明。四个成熟域均通过当前 integrity card；Steam/TMDB 的 `shared_genre` 和 WikiArt 的 `shared_style` 因 target provenance 被排除，arxiv citation 保留。

### C3：四域、四协议、六任务的 MAG benchmark portfolio

同一个 MAG 在 transductive、split-disjoint、temporal 与 group-inductive 条件下表现不同；这种差异已在六类图与模态任务中量化。框架把协议敏感性作为 benchmark 内容，而不是把一个协议包装成唯一正确答案。

### C4：可复用、可审计的发布接口

新域通过 YAML 声明 label、relations、time field 和 group field。生产复用审计已覆盖 Steam、TMDB、WikiArt 与 held-out Spotify，四者均通过 `YAML → deterministic schema audit → mandatory five-agent plan → existing graph contract`。Spotify 没有可信时间字段，因此只发布 transductive、split-disjoint、group-inductive；这说明框架会拒绝伪造不适用协议，而不是机械凑齐四项。

## 4. 核心研究问题

- **RQ1 Automation**：框架能否减少真实 MAG schema 与构图的人工工作？
- **RQ2 Integrity**：能否识别并移除可验证的 target-derived/duplicate/future-information defects？
- **RQ3 Protocol sensitivity**：模型性能、模型排序和 GNN−MLP gap 是否随部署 protocol 改变？
- **RQ4 Representation robustness**：text/image/formula 融合收益是否跨 protocol 稳定？
- **RQ5 Reusability**：未见领域能否通过声明式配置生成带完整性报告的 benchmark？
- **RQ6 Task portability**：同一份经审计的 protocol views 能否在分类、预测、聚类、检索、对齐和匹配中保持清晰且可复现的训练/评估边界？

## 5. 当前核心证据

四域完整矩阵已完成：64 个 dataset×protocol×modality cells、384 个 model cells、1,920 个 seeded training runs。最终审计脚本重新计算均值、聚合表、模态差值、temporal label support、protocol invariants 与 topology null 元数据，共 1,739/1,739 checks 通过。融合模态 headline 如下：

| Dataset | Transductive | Split-disjoint | Temporal | Group-inductive |
|---|---:|---:|---:|---:|
| arxiv | **0.9165** | 0.8999 | 0.8664 | 0.9028 |
| steam | **0.6846** | 0.6621 | 0.6179 | 0.6569 |
| tmdb | **0.5915** | 0.5892 | 0.5470 | 0.5775 |
| wikiart | **0.8895** | 0.7920 | 0.0101 | 0.5665 |

固定 transductive winner 后，split-disjoint 下降 arxiv 1.69pp、steam 3.29pp、tmdb 0.89pp、wikiart 9.75pp；temporal 下降分别为 5.35pp、11.19pp、4.45pp、88.29pp；group-inductive 下降分别为 1.37pp、4.12pp、1.40pp、32.35pp。

WikiArt temporal 是 open-set 特例：year coverage=85.99%，3,078 个无年份节点不进入 temporal view。训练期缺少全局 64 类中的 19 类，test 实际包含 18 个训练未见类；训练可见标签只覆盖 1,495/3,848=38.85% 的测试节点，因此 train-support oracle ceiling 为 0.3885，最大训练可见 test 类本身占 0.0990。模型 accuracy 0.0101 仅为该覆盖上限的约 2.6%，所以 label-space emergence 只解释天花板，不能单独解释全部下降。可见类条件下 train/test prior TVD=0.965、JS=0.646 nats，支持 severe prior shift 的诊断；剩余差距仍可能包含 covariate shift 与模型失配。以上均不是 leakage 证据，也不能与普通 closed-set temporal drop 混为一谈。

四协议模态消融进一步显示：Steam 与 TMDB 的 full fusion 在四种 view 均优于最好单模态；Arxiv 的 text 在三种 view 更好；WikiArt 由 image 主导，且 group-inductive 下 full fusion 比 image 低 3.34pp。因此 RQ4 的答案不是“融合稳定增益”，而是融合收益受域与部署协议共同控制。

六任务补齐实验又增加 480 个 label-free graph-autoencoder seeded runs（同时评估 LP 与 clustering）、120 个 modality-matching seeded runs，以及 24 个 retrieval/alignment protocol×pair cells。输入哈希、边拆分、实体隔离、逐 seed 聚合、指标范围与最终文档转录共 2,665/2,665 checks PASS。NC 1,920 runs 加上述 600 个训练 runs 后，当前四协议核心共有 2,520 个 seeded training runs。

图任务按 validation AP 选择统一表示模型。LP AUC 从 transductive 到 temporal 分别为 arxiv 0.9286→0.8594、steam 0.9081→0.6963、tmdb 0.9174→0.7503、wikiart 0.9921→0.6750；这是 protocol difficulty，不是 transductive leakage 裁决。模态结果显示 Steam/TMDB text–image retrieval R@1 为 0.6223/0.8496，WikiArt 为 0.1329 且 temporal 降至 0.0849；Arxiv text–image R@1=0.1074，而 image–formula 仅 0.0006。Matching 可以显著高于零样本 retrieval，因此二者测量的不是同一种能力：前者是可学习的配对可分性，后者是冻结空间的精确实例排序。

## 6. 写作红线

1. 不把普通跨 split 边统一叫结构泄露。
2. 不把 LP/topology-only 高分作为单独泄露证据。
3. 不把高 homophily 自动当定义性关系；必须看 lineage。
4. 不把 CLIP-vs-ImageNet 差值写成污染已证。
5. 不声称所有域都 GNN>MLP、所有域都需要双模态。
6. 不声称 greedy/LLM 关系选择优于 oracle；已有消融不支持该结论。
7. 不把专家估计时间写成真人 user study。
8. 不把 WikiArt temporal 0.0101 简写成一般时间泛化失败。

## 7. 论文 Story Arc

1. 真实 MAG benchmark 稀缺，手工构建昂贵，且数据完整性与评估条件常被混为一谈。
2. MAG-Forge 自动完成真实来源到 MAG 的 schema、构图与发布。
3. provenance-based auditor 先处理可验证缺陷，再生成四种明确部署协议。
4. 四域六任务实验说明单一随机切分无法完整描述 benchmark：结构暴露、时间漂移、未见实体组和模态对兼容性产生不同难度与模型排序。
5. integrity card 使数据集的正确性、适用条件与失败边界成为正式发布内容。

## 8. 尚未闭合的证据

- 四协议模态消融：已完成，支撑“融合价值依赖 domain 与 protocol”的 RQ4 结论。
- topology label-permutation + degree-preserving null：已完成；真实拓扑高于标签置换 null，但仅用于解释结构信号，不做泄露裁决。
- 六任务四协议 benchmark：已完成；LP/聚类、检索/对齐/匹配均已按当前 sanitized views 重跑，2,665/2,665 新审计项通过。
- sample-level pretraining overlap：缺少可查询训练语料索引，当前不能完成强污染结论。
- Exp3/Exp4b 真人效率与外部复用研究：需要真实受试者，不能由服务器模拟。
- 生产 YAML 复用合同：已完成；held-out Spotify 证明声明式接入，但不等同于真实外部用户研究。

## 9. 与现有 MAG benchmarks 的定位

- [MAGB](https://arxiv.org/abs/2410.09132)（KDD 2025）提供多领域 text/image MAG，并比较 GNN/VLM learning paradigms。
- [MM-GRAPH](https://arxiv.org/abs/2406.16321)（CVPR 2025）提供 7 个 text/image graph 数据集与统一模型研究。
- [OpenMAG](https://arxiv.org/abs/2602.05576)（arXiv 2026）扩展到 19 数据集、16 encoders、24 models 与 8 tasks。
- MAG-Forge 的正交贡献是从真实来源到 schema、构图、provenance audit、四协议 derived views 和 integrity card 的自动发布链。论文不声称这些现有 benchmark 存在泄漏，也不以模型/任务数量与 OpenMAG 竞争。

## 10. 权威产物

- 数字总表：`docs/experiment_results_FINAL_decontaminated_2026-07-16.md`
- 四协议矩阵：`benchmark/results/newdomain_nc/protocol_matrix/protocol_matrix_all.{json,md}`
- 六任务矩阵：`benchmark/results/four_protocol_tasks/six_task_summary.{json,md}`
- 六任务 runners/audit：`benchmark/run_four_protocol_graph_tasks.py`、`benchmark/run_four_protocol_modality_tasks.py`、`benchmark/aggregate_validate_four_protocol_tasks.py`
- Protocol manifests：`benchmark/results/protocol_views/*_protocol_manifest.json`
- Manifest 同步保护：`benchmark/sync_protocol_manifests.py`
- Integrity cards：`benchmark/results/report_cards.{json,md}`
- 最终数字审计：`benchmark/results/final_evidence_audit.{json,md}`
- 生产复用审计：`benchmark/results/reusability_audit.{json,md}`
- 自动测试：`tests/test_protocols.py`、`tests/test_framework_integrity.py`

## 11. 框架机制证据与写作更新（2026-09-15）

论文采用 *MAG-Forge: Multi-Agent Construction of Auditable Multimodal Graph Benchmarks* 标题。六模块图保持现有顺序；图中模块 4 简写 **Integrity Auditing**，模块 5 为 **Protocol-Safe Representation**。完整职责仍包括来源审计、四协议派生、输入过滤与冻结编码。arXiv 使用专门的引用图路径，声明式 co-attribute 路径的 schema 确认顺序不能泛化为所有领域均自主发现 schema。

### 11.1 Intro 的实证依据

Section 3 采用“问题 → 观察 → 结论/启发 → 模块设计”的链条：历史 schema 的合法性不等于适用性；固定 GraphSAGE 的四域 edge-only drop 为 1.26/2.29/0.89/9.75pp；拓扑 null 不能替代来源证据。Introduction 提前引用本节，电影 genre/同导演例子解释为什么这些决策需要跨阶段共享。

原 NC/fusion 表按 mean test accuracy 选架构，是 **descriptive maxima**。固定 transductive winner 也由 transductive test mean 选择，不能描述成完全独立于测试的选模。Intro 的协议证据改为跨域统一 GraphSAGE 的回顾性分析。五个初始化 seed 的区间只描述固定划分下的初始化变动。

### 11.2 已完成的 schema 角色对照

Steam/TMDB/WikiArt/Spotify × 3 generation seeds；两臂共享初始 proposal、300-item profile 和 full-data validator feedback。分别由原智能体继续修订、独立 schema reviewer 在新上下文修订。模型 qwen-plus，temperature=0.2，每次输出上限 1,200 tokens；各臂逻辑调用数和输出上限相同，实际 token 成本不完全相同。

两臂均为 12/12 schema-level publishability、12/12 ID agreement、9/12 base-label agreement。该 publishability 是实验检查条件：production validator 有效且至少一个非 target-derived relation 的共享覆盖率不低于 5%，并非完整构图或科研价值认证。

Spotify 两臂都选 playlist_subgenre，参考配置为 playlist_genre；这是任务定义分歧。两次 TMDB 原智能体选择 genres，独立审阅选择 genres[0]；基础字段一致也不代表标签语义完全一致。

**结论：该小样本实验未观察到独立角色在 publishability 上的优势，也不能证明两种方式等价。** 不把它称为端到端多智能体优越性实验。多智能体贡献继续定位为职责组织、共享产物、任务依赖及可复用接口。

### 11.3 已完成的确定性质量门消融

四域各注入 5 种可控缺陷，共 20 个；另有 8 个 clean/harmless-rename controls。在内存副本上调用生产 validator，用 harness 分别执行或绕过 schema rejection 与 lineage exclusion。

| Decision policy | 阻止缺陷 | 误拒绝 controls |
|---|---:|---:|
| Full checks | 20/20 | 0/8 |
| Without schema rejection | 4/20 | 0/8 |
| Without lineage exclusion | 16/20 | 0/8 |

它支持两类确定性检查在已注入缺陷上的互补作用。不能扩大成任意语义缺陷、近重复、所有协议错误或端到端发布可靠性已获证明。

新核验为 114/114 checks；schema/protocol 回归测试 8/8。真实 API 调用 36 次，共 52,891 tokens；没有新增下游训练，2,520 个 seeded training runs 保持不变。

产物：`benchmark/run_framework_mechanisms.py`、`benchmark/validate_framework_mechanisms.py`、`benchmark/results/framework_mechanisms_20260915/`。仍缺少端到端同工具/预算的单智能体、确定性流水线与多智能体对照，以及外部用户复用证据。
