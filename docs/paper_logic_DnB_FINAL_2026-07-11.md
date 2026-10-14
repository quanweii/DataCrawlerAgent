# MAG-Forge 论文逻辑线（FINAL CLEAN，2026-09-15）

## 2026-10-14 更新：Cleveland 新多模态来源完整生命周期案例

本节覆盖此前“新增来源仅验证配置／构图”的证据边界，但不把新实验倒推为旧实现既有能力。没有使用音乐源作为本次案例。

- 来源是 Cleveland Museum of Art Open Access API（https://www.clevelandart.org/open-access-api）。作者在当前对话实际确认“认可，按这个任务执行”：以馆藏对象 type 为目标，creator_ids/culture 为关系字段。确认记录保存在 `benchmark/results/cleveland_e2e_20261014/researcher_confirmation.json`。
- 预先固定 Painting、Print、Drawing、Sculpture、Ceramic、Textile 六类，每类按 API 返回顺序保留150个通过 CC0/图像解码/唯一图像哈希检查的对象，总计900；是类均衡便利样本，不是自然分布或按模型分数筛选。900张真实图像均有来源URL与hash，898个对象有馆方叙述文本，文本词数中位数100。
- 新增 `tools/cleveland_tools.py`、`domains/cleveland/domain.yaml` 和案例运行／验证入口。发现通用 Formatter 原来只透传三个内置域，需要通用同构bundle扩展；这是本次补齐的工程能力，不称为旧系统的零代码能力。Builder、协议生成、CLIP及NC/LP/clustering/retrieval/alignment/matching训练/指标函数直接复用生产实现。
- 完整五角色 Collector→Schema→Builder→Formatter→Verifier 实际执行并success，保存 28 次 qwen-plus 调用、24,179 tokens 的输入/响应记录。900节点、3,505边；接受 researcher-confirmed schema，不能写成自动发现目标或自动生成网站爬虫。
- 每种模态900个非零、有限768d CLIP向量；无空图片目录或随机占位图。Trans/Split/Group均生成并落盘；creation-year bounds不构成collection/observation chronology，Temporal unavailable。creator-ID coverage为60.6667%，354个缺失值用singleton；不能声称所有测试对象都对应严格未见的真实创作者。13个重复文本组保留并披露，900张保留图像的exact hash不重复。
- 本次新增375次训练：270 NC（3协议×3输入×6架构×5seed），90 LP/clustering共享（3×6×5），15 matching。四成熟域主网格仍为2520，本次案例新增375；旧NC重执行1920、第二partition480和release288依各自历史账目另列。没有重新训练或修改四成熟域数值。
- NC checkpoint与architecture按validation accuracy；每个cell架构由五seed平均validation选择；LP/clustering共用validation AP所选模型；matching按validation AP选checkpoint。数据卡也按validation选模，并只报告声明支持的三个协议，不能回退到test maxima或伪造Temporal。
- Fused NC的Trans/Split使用MLP，test mean=.9367；Group使用GraphSAGE，=.9326。相应validation-selected image-only为.9267/.8729。完整6任务、逐seed/all-model结果见下面的新实验报告及CSV/JSON；新论文只用简洁案例表避免超过篇幅。
- Matching存在初始化敏感性，AUC标准差为.108--.128，论文附录已明确披露；不以均值证明稳定配对能力。
- 导出三份audited benchmark bundles、十五个selected NC checkpoints、十五份test prediction CSV、accepted schema、protocol manifest、完整网格、integrity card及可加载的tar.gz。独立检查4684项PASS，包括重新加载十五个真实checkpoint、用保存的view/features重新推理并逐项复算validation/test Acc./Macro-F1，均与训练记录完全一致。
- 人工边界：作者确认目标/关系意义；适配器、YAML、配额、输入字段白名单和artifact路径注册是实现设置；分类体系适用性、代表性、匿名/集合归属及语义重复仍需专家判断。馆方叙述可以提到对象类型，本例不宣称纯视觉推理评测。LLM自动总结保留作溯源，完成性依据真实导出文件/解码/预测见证，不依据其总结。
- 论文§5.6改为完整新源案例；附录补齐六任务结果及来源/配置/人工作业边界。Abstract/Introduction、原四域表和四幅原图不改动。本次不推导agent相对同预算single-agent/script的因果优势，也不声称任意网站可自治接入。
- 最终正文8页，全文12页；PDF已逐页渲染检查，无Overfull/未解析引用。论文与实测结果101项一致性检查PASS。执行时源码快照保留在execution_snapshot；完成目录由入口防覆盖，实际执行版本与新增防覆盖版本分别保存。

原始及可用结果目录：`/home/dev_qw/DataCrawlerAgent/benchmark/results/cleveland_e2e_20261014/`；raw/cache/images位于data/cleveland_raw、data/cleveland_cslo；源图位于data/cleveland_datasets；固定编码位于benchmark/processed/cleveland。论文修改前备份在 `/home/dev_qw/www2027_paper/revisions/new_multimodal_source_20261014/before/`。

> 下方保留历史记录；涉及旧source-only边界和旧页码时，以本节和当前稿为准。

## 2026-10-14 更新：验证集选模，测试集报告（覆盖此前 NC/test-max 口径）

- 根据审稿意见，正文 NC 主表、融合比较和六任务 NC 面板统一改为 validation-only selection。每个 seed 的 checkpoint 按 validation accuracy 选；每个 domain × protocol × modality 的架构按 5 seeds 的 mean validation accuracy 选；架构并列按 MLP/GCN/GAT/GraphSAGE/GIN/GCNII 固定顺序，不使用 test 指标打破并列。
- 单模态基线也按 validation 选择：先在各单模态内选架构，再在 text/image/formula 中比较验证均值。full fusion 独立按相同规则选架构。表中报告测试差值及五个配对初始化差值的 population std；不再使用 test-max 或 root-sum-square dagger 标记。
- 原生产 NC JSON 未保留 validation 分数，因此按原 split_seed=1、保存的 masks/edges、冻结特征和原训练设置完整补跑 1,920 次（64 个 modality/protocol cells × 6 models × 5 seeds）。重跑结果替代论文中原 NC 数值；旧结果不覆盖，作为历史记录保留。本次增加 1,920 次训练执行，不能写成“没有新训练”。核心新主表训练网格仍为 1,920 NC + 480 LP/clustering + 120 matching = 2,520；历史首轮 NC、第二划分 480 次及 release 288 次分别保留。
- 新结果与旧结果之间的差异同时包含架构选择变化和重执行的数值波动，不把全部差异归因为 test-selection bias。所有协议的重生成 masks/edges 均与已保存视图逐 tensor 一致。
- 原始逐 seed validation/test/epoch 记录位于 `benchmark/results/validation_selected_nc_20261014/nc_*.json`；唯一的当前 NC 汇总是 `validation_selected_summary.json/.md`，验证选模清单是 `validation_selection_manifest.json`。`validation_selection_audit.json` 为 PASS：5,186 项校验、1,920 次训练。任意改动所有 test 分数不会改变架构或 single-modality 选择；并列不看 test。
- 论文正文 Table 4 同时报验证分数与测试 Acc./Macro-F1；Δ_G 是按验证集在五个 GNN 中选择后，再减 MLP 的测试分数，不是 test-best GNN。Table 5 用验证选出的单模态基线。附录 Table 8 保留全部 96 个融合模型/cell 的测试均值和标准差，加粗的是 validation winner，可以不是 test winner。
- Figure 3 的固定 GraphSAGE 主划分数据同步使用重跑：arXiv/Steam/TMDB/WikiArt Trans−Split 为 1.30/2.54/0.31/9.82pp；第二划分保持 1.04/0.75/−0.17/10.02pp。Figure 4 NC 面板与 Table 4 一致。LP/clustering、retrieval、alignment、matching 的数值不变；原来它们的选模/检查点规则已使用 validation 或为冻结诊断。
- 结论相应收敛：不能再称 Trans 在所有域都给最高选中模型测试分数；Steam Split 的 validation winner 为 GraphSAGE、Temporal 为 MLP；WikiArt Temporal 选 GraphSAGE，test accuracy 0.0065，约为 seen-class reference 的 1.7%。融合在 Steam/TMDB 的四协议均为正，平均 +2.04/+2.76pp；arXiv 四协议均为负，WikiArt 混合，不能泛化成“多模态总是更好”。
- Abstract/Introduction 保留用户恢复的原文，只把固定 GraphSAGE 的 9.75pp 同步为 9.82pp；主图不变。本次是已有划分的回顾性报告修正，不等于新的从未见过的 test cohort，不宣称消除所有选择偏差或保证录用。
- 后续生产 NC runner 新增 validation 分数与 best epoch 持久化；两个 protocol 汇总器改为 validation selection，缺少 validation 数据时报错，不回退到 test selection。新独立运行器不改生产视图或历史 JSON。

### 当前验证选模主结果


> 以下记录保留历史实验与修订过程；涉及旧 NC 数值、test-max、7/16 一致率和旧图表估计时，以本节的新汇总为当前论文口径。

## 2026-10-13 更新：恢复原摘要和Introduction，调整方法/实验篇幅

- 摘要与Introduction从篇幅压缩前的长版逐字恢复，依据revisions/page_compression_20261013/extended_record/www2027_conference.tex；逐段内容一致性检查通过。新用户提供的framework主图继续使用，图片没有改动。
- 仅进一步精简方法中的重复状态/反馈说明、正文release验证解释和card用途；次要selection/ranking数值分析移入附录，实验数值表和全部四幅核心图均保留。
- 补齐schema gate公式的D、类别集合和n_c定义，补齐fusion公式的z向量定义及非空modality条件；将Algorithm的retry重置pending/继续循环写清；图注说明图中简化notation及repair指声明输入排除，不代表已经验证的语义恢复能力。
- 正文仍为第1–8页，参考文献第9页，附录第10–12页，全文12页。Framework Figure 2因恢复前两部分后的重排位于第4页页首，复现Figure 3在第6页，六任务Figure 4在第7页。此记录覆盖之前章节/图表页码。
- 摘要和Introduction之外的精简没有新增训练、修改实验数值或削弱解释边界；32项数据/引用/位置核对及编译、PDF排版检查通过，无undefined引用或Overfull。修改前正式稿、figures和两份docs备份在revisions/restore_opening_20261013/before/。

## 2026-10-13 更新：WWW 2027 Research Track 投稿篇幅压缩版

- 官方长文篇幅：正文最多8页；参考文献和可选附录合计后全文最多12页。核实来源：https://www2027.thewebconf.org/research-track-papers/。
- 服务器正式稿现为8页正文、1页参考文献（第9页）、3页附录（第10–12页），总计12页。此记录覆盖之前关于章节/图表页码的描述；全部历史实验记录保留。
- 第三章与第四章文字连续：方法章从第2页开始，framework Figure 2在下一页（第3页）页首。结构暴露复现Figure 3在第6页，六任务Figure 4在第7页。
- 核心结果均在正文：NC Table 4，模态收益Table 5，完整LP/clustering Table 6，text–image概览Table 7。附录完整融合NC Table 8保留全部96个模型/协议均值与标准差；retrieval/alignment Table 9和matching Table 10保留全部24个实验单元的指标及cohort计数，仅合并完全相同的Trans/Split行。
- 故障效应、时间标签支持、第二partition复现分别为Tables 11–13。主要图、结果数值、源文件与全部已引用文献不变；本次没有新训练或新增实验数据。
- 按审稿阅读逻辑精简引言/相关工作/重复协议解释，保留agent决策权限、schema gate、lineage、exposure、任务契约、fault设计与解释边界；非核心历史schema timing、topology探针等完整长版内容归档到revisions/page_compression_20261013/extended_record/，不作为12页投稿PDF的一部分。
- 未解决的证据边界仍显式保留：未证明多agent相对预算/工具匹配的单agent或脚本的因果优势，已知注入故障拒绝不等于自然错误恢复或未见故障鲁棒性；NC architecture test maxima仍为描述性结果。
- 编译无未定义引用或Overfull；12页渲染检查与32项正文/数值/引用保留核对通过。修改前正式稿与docs完整备份在revisions/page_compression_20261013/before/。

## 2026-10-13 更新：方法主图与实验图位置

- Framework主图固定在Methodology章标题后、方法解释之前，PDF第4页；随正文顺序重编号为Figure 2（label仍为fig:framework）。
- 结构暴露与两partition复现图移到§5.3 Protocol sensitivity，第8页，重编号为Figure 3（label仍为fig:construction_diagnostics）。
- §3 Q2缩为动机观察及指向§5.3/实验图的引用。六任务Figure 4继续保留。图片、数据及所有旧表内容不改动，只有章节位置、交叉引用和动机文字调整。
- 使用方法章全宽开头使主图与标题同页，避免双栏浮动把主图延后到其他页面；编译与全PDF渲染检查通过。备份在revisions/figure_placement_20261013/before/。

## 2026-10-12 更新：高优先级实验主图与审稿论证同步

本轮只改论文呈现，原始数据、配置、模型与训练结果不变，没有新增训练。

- `fig:construction_diagnostics`改为`figures/exposure_replication.pdf`：两partition固定GraphSAGE、frozen full-modality，分别使用相同Trans/Split masks；五个paired initialization差值全显示，均值与95% paired-seed bootstrap CI（10000次、seed20260915）按partition分别计算。右侧小效应放大复用同一批观察，不是额外证据。WikiArt 9.75/10.02pp保持，Steam变化与TMDB符号反转可见；CI不估计partition总体不确定性。
- `fig:task_response`改为`figures/six_task_applicability.pdf`：六任务独立分面，primary benchmark suite的4域×4协议；NC取原test-mean descriptive maxima，LP/Clustering共用validation-LP-AP架构。模态面板限定text–image；CKA诊断／R@1检索冻结，Matching entity-disjoint。四个训练任务的bar为五初始化population std，冻结项没有伪造error bar。
- Steam高亮使用原始未舍入值：CKA变化+0.08628899，R@1变化−16.80203533pp；图示+.0863/−16.80pp。两图库分别560/558合格实体，变化不解释为孤立时间因果效应，也不把CKA数值当检索质量。
- 表格完整保留：arXiv其他模态对、R@5/10/MRR、cosine gap、Matching AP及全协议矩阵仍在附录。大图提前声明，放在分析附近，不在Conclusion后另起空白图页。
- 设置前置Steam/TMDB genres[0]单标签目标、WikiArt主要title文本；方法明确benchmark/release扩展在调度之后分别调用，opt-in发布门没有被描述为当前scheduler已自动接通。
- 补充使用建议：选择符合研究设置的暴露视图，时间结果先检查label support，exact counterpart任务以retrieval为端点；Matching/CKA提供互补诊断。仍没有新增同预算multi-agent/single-agent/script对照，不将图表改进写成架构优越性证据。
- 科学作图源码`figures/make_priority_figures.py`及`figures/priority_figure_data.json`保留输入SHA-256、逐cell值、五初始化差值和新图的重生成路径；高优先级修订备份在`/home/dev_qw/www2027_paper/revisions/priority_figures_20261012/before/`。

## 2026-10-12 更新：六任务结果完整呈现与论文证据边界同步

本节更新论文呈现及结论口径；下方历史实验记录保留。**没有新增训练或改动原始实验结果。** 核心训练总量仍为 NC 1,920 + LP/Clustering 共享 480 + Matching 120 = 2,520；Retrieval/Alignment 为24个协议—模态对单元的确定性评估。

- Framework 最后一个 benchmark 模块的六类任务均已有实际结果。NC 由正文 `tab:nc`、`tab:fusion` 与完整 per-model 表支持；LP/Clustering 由 `tab:graph` 覆盖4域×4协议的16个单元，增加 validation-AP 所选架构与五种子 mean/std。
- 正文 `tab:modality_overview` 新增4域×Trans/Temp的8个 text–image 单元，同时呈现 R@1、CKA、cosine gap、Matching AUC；§5.5按问题、设置、结果、复用含义组织，不再遗漏 Alignment 数值。
- 附录 `tab:modality_tasks` 完整列出24个单元的 paired test gallery 数、双向平均 R@1/R@5/R@10/MRR、CKA、cosine gap；`tab:matching_full` 完整列出同24个单元的 train/val/test paired entity 数与 Matching AUC/AP mean/std。arXiv 包含 text–image/text–formula/image–formula，其余三域包含 text–image，四协议全部覆盖。
- Alignment 是冻结表示诊断：centered linear CKA衡量整体表示依赖，cosine gap对比同实体与单行循环错配。它不使用人工语义对齐标注，也不证明语义等价或统计显著。共享CLIP坐标只是融合前提，不能保证同实体检索质量。
- Steam 的 Trans→Temp CKA .4071→.4934，而 R@1 .6223→.4543，说明整体表示关系不能代替实例检索。WikiArt .2106→.1281 CKA、.0732→.0445 gap、.8150→.6679 matching AUC只描述当前保留的title–image输入及变化后的cohort。
- arXiv image–formula Trans R@1 .0006、Matching AUC .7558说明冻结全图库排名与学习后的负例配对判别具有不同适用性，不能直接比较为同一任务的性能提升。
- Retrieval/Alignment过滤缺失、非有限或零向量，使用整个合格test gallery，recall/MRR双向取平均，ties保守计rank。Matching每实体正负各一，优先同标签不同实体；单例类回退其他标签，validation AP选checkpoint。cosine gap的循环错配不是Matching的同标签负例。
- LP仅用training-positive message edges，非transductive held-out端点无邻居；Clustering用协议允许的拓扑推理、train节点拟合K-means，共用validation-LP-AP所选编码器，并未针对NMI/ARI独立择优。不同任务的协议响应不等同于单一拓扑干预的因果收益。
- Trans/Split frozen结果一致源于相同node masks与frozen features，是一致性检查；Temporal还改变cohort、gallery size与label support，不是重建历史特征可用性。Time buckets：arXiv/Steam/TMDB为月，WikiArt为年。缺失primary group使用逐实体singleton key，并保留覆盖率限制。
- 同步此前核查口径：schema eligibility为12/12、base target agreement为9/12，不能替代完整嵌套路径/目标类型验证；held-out reuse是图构造与声明配置核查，未验证全套模态编码和协议视图落盘；六模型Spearman相关是描述性结果；seed标准差及root-sum-square标记不是显著性检验；第二split seed中的Temporal仍为同一时间partition的重训练。
- 现有证据支持可审计构造、协议依赖和任务适用性；multi-agent相对同预算单agent/script的因果优势、自然缺陷自动修复收益及外部用户收益尚未由这些实验确立。论文不以任务数量或高AUC代替这些证据。

数据来源：`benchmark/results/four_protocol_tasks/six_task_summary.json` 与该目录各域 `modality/*_modality_tasks.json`；所有新增表格逐单元核对原始数据，不修改原JSON。服务器备份位于 `/home/dev_qw/www2027_paper/revisions/benchmark_completion_20261012/before/`。

## 2026-10-11 更新：方法完整算法与调度状态同步

WWW 方法部分新增完整算法，连接角色局部 Observe/Think/Act、检查失败重试、重处理状态失效、下游再生成与停止条件。题目保持不变。

- 原反馈调度追加同名 builder，且使用接收队列名作为前驱；最小控制流复现中，带反馈上下文的 builder 可能执行晚于 formatter/verifier，导致下游产物早于最后一次重建。
- 修正为重用原步骤及原前驱，传入 reprocess_context；撤销 target 和已完成依赖后继的完成标志，从完成集合和 retained stage results 中移除受影响角色。先重建，再重新运行 formatter/verifier，重新生成包装、审计、protocol views/manifests。
- 每次运行重处理计数重新初始化。每个 target 最多两轮；耗尽后保留 unresolved_reprocessing，并将运行结果标记 partial。
- schema 阶段必须 published 且 schema_audit.valid，非法或未发布 schema 的 partial 结果改判 execution error，阻止 builder 绕过 schema gate。
- 关键阶段失败或无就绪步骤且还有 pending/running 时为 failed；非关键失败、partial 结果或未解决反馈为 partial。调度终止与完整性验收不等价。
- 下游固定 benchmark procedures 在构建之后消费当前产物。输入图/cohort/features 改变时，相关任务输入、学习模型、指标和 cards 要重算；冻结逐节点表示仅在 ID、保留模态内容和 encoder 配置不变时可复用。
- 新增无 API 的控制流回归；它验证调度行为，不是自然缺陷的语义修复成功率、多智能体比较收益或历史 benchmark 重新运行。现有 2,520 training runs、114/114 历史 checks、schema 比较和所有模型数字不改变。


## 2026-09-18 更新：正文 Q1 改为工件发布边界实验

本节覆盖此前关于 empirical Q1 的安排；历史实验记录保留，不把历史数字归因于新增功能。

- 新 Q1：**Is a runnable artifact sufficient for release?** 不是“框架优于人工”的人类实验。
- Intro 对应动机：可定制构造的各阶段决定需要成为可执行的发布约束，而不是仅生成可运行文件。直接引用 §3 Q1 的 120/120、0/120 和 36/36 对照结果。
- 样本来自 arXiv、Steam、TMDB、WikiArt 已有真实图的固定子集；10 种构造/评估契约缺陷 × 4 域 × 3 seeds。三种策略处理相同候选，homophily 策略随后选择最高训练同质性关系。
- 两种无审计策略均发布 120/120 含注入缺陷工件；新增 audited release 拒绝 120/120，接受 36/36 正常对照。结果指实际落盘工件的残留缺陷，不只是验证器返回值。
- 下游不仅保留虚高：NC 目标派生边 +23.95 pp、显式目标特征 +32.90 pp、特征版本错配 -36.76 pp；其余零、小幅或负效应均记录。含义是“分数不能替代契约检查”，不是“任何完整性问题都会提高分数”。
- 方法对应：`benchmark/artifact_release.py` 是本轮新增、可选的发布门。原 `save_protocol_suite` 仅序列化，不把新门的零逃逸倒推为旧框架既有性质。此实验拒绝缺陷，未证明自动修复或完成重新发布。
- 原耗时 Q1 移入附录，未丢弃历史记录。摘要不变；正文仍在第 10 页结束。
- 结论边界：已知故障族、可信 contract、人工注入、固定子集；不代表自然缺陷率、人工错误率、多智能体相对单智能体收益，也不能声称优于认真审计的人工构建。目标特征是 one-hot 注入，不是原始文本经 CLIP 的标签残留实验。
- Appendix F 保留完整 3×10 逃逸表、10 类 NC/LP signed effects、实验参数与局限。结果目录：`benchmark/results/artifact_release_20260918/`；正式新增训练 288 次，独立于原 2520 次。


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
