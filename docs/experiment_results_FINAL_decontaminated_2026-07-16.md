# MAG-Forge 实验线（FINAL CLEAN，2026-09-15）

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

# Cleveland 新多模态来源完整案例

真实来源：Cleveland Museum of Art Open Access API。900 个保留对象，六类各 150 个；900 张真实图像；898 个有馆方叙述文本。固定类额的便利样本，未按模型结果选择。

作者已确认以对象类型为目标、创作者／文化来源为图关系。适配器、YAML、输入白名单和结果路径注册是实现设置；不能写成 Agent 自动生成任意网站爬虫。

375 次训练（NC 270、LP/聚类共享 90、Matching 15）完成。4684 项独立校验通过，十五个已导出 NC 检查点重新加载后的验证／测试预测与原记录完全一致。

| View | NC model | Val Acc. | Test Acc. ± std | Macro-F1 | LP model | LP AUC ± std | NMI ± std | R@1 | CKA | Match AUC ± std |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| transductive | mlp | 0.9456 | 0.9367 ± 0.0044 | 0.9366 | graphsage | 0.9539 ± 0.0013 | 0.4255 ± 0.0233 | 0.7167 | 0.6966 | 0.7231 ± 0.1279 |
| split_disjoint | mlp | 0.9456 | 0.9367 ± 0.0044 | 0.9366 | gat | 0.9324 ± 0.0010 | 0.4835 ± 0.0191 | 0.7167 | 0.6966 | 0.7231 ± 0.1279 |
| group_inductive | graphsage | 0.9478 | 0.9326 ± 0.0180 | 0.9322 | gcn | 0.8750 ± 0.0061 | 0.4268 ± 0.0750 | 0.7099 | 0.7165 | 0.6839 ± 0.1083 |

NC 架构按五 seed 平均 validation accuracy 选择；LP/Clustering 共用按 validation AP 选择的模型；Matching 的 checkpoint 以 validation AP 选择。原始全网格及所有指标在 CSV/JSON 中保留。

人工边界：作者确认目标与关系语义；配置阶段确定 CC0、六类配额、文本字段和协议。354 个对象缺少创作者 ID，使用 singleton，Group 不能保证所有未知归属互为不同人；13 个重复文本组保留，图像哈希不重复；叙述文本可能直接描述对象类型。分类体系适用性、样本代表性与归属语义需要领域专家判断。

资料已导出：三份 audited benchmark .pt（含标签映射／特征／边／mask），十五个 selected NC checkpoint .pt，十五份 test prediction CSV，schema、完整 API/LLM 记录、protocol manifest、integrity card、全模型结果。流水线 LLM 摘要保留用于溯源，不能代替真实图像和检查点的检查。

复现：在 DataCrawlerAgent 根目录加载 domains/cleveland/domain.yaml；benchmark/run_cleveland_case.py 复用现有 Agent 和六任务函数。已有案例不可覆盖发表工件；独立重跑使用 --output-dir 指向新目录并保留 researcher_confirmation.json。源码与数据无需音乐源。


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

| Domain | View | Selected model | Validation acc | Test acc mean 卤 std | Test macro-F1 | Selected GNN鈭扢LP |
|---|---|---|---:|---:|---:|---:|
| arxiv | transductive | gin | 0.9166 | 0.9161 卤 0.0015 | 0.9149 | +0.0247 |
| arxiv | split_disjoint | gin | 0.9035 | 0.9006 卤 0.0010 | 0.8998 | +0.0092 |
| arxiv | temporal | gcnii | 0.8828 | 0.8645 卤 0.0049 | 0.8577 | +0.0089 |
| arxiv | group_inductive | gin | 0.9051 | 0.9005 卤 0.0013 | 0.8999 | +0.0121 |
| steam | transductive | gcnii | 0.6964 | 0.6904 卤 0.0098 | 0.5068 | +0.0282 |
| steam | split_disjoint | graphsage | 0.6788 | 0.6614 卤 0.0089 | 0.4486 | -0.0007 |
| steam | temporal | mlp | 0.5679 | 0.6122 卤 0.0067 | 0.4455 | +0.0047 |
| steam | group_inductive | graphsage | 0.6333 | 0.6569 卤 0.0171 | 0.4449 | +0.0337 |
| tmdb | transductive | graphsage | 0.6061 | 0.5876 卤 0.0035 | 0.3872 | -0.0017 |
| tmdb | split_disjoint | mlp | 0.5992 | 0.5892 卤 0.0030 | 0.3619 | -0.0099 |
| tmdb | temporal | graphsage | 0.5534 | 0.5470 卤 0.0035 | 0.3165 | +0.0009 |
| tmdb | group_inductive | graphsage | 0.5916 | 0.5797 卤 0.0024 | 0.3803 | +0.0069 |
| wikiart | transductive | graphsage | 0.8937 | 0.8898 卤 0.0015 | 0.8723 | +0.2421 |
| wikiart | split_disjoint | graphsage | 0.7910 | 0.7917 卤 0.0026 | 0.7402 | +0.1439 |
| wikiart | temporal | graphsage | 0.1636 | 0.0065 卤 0.0004 | 0.0075 | -0.0001 |
| wikiart | group_inductive | gcn | 0.5805 | 0.5663 卤 0.0087 | 0.4376 | +0.1268 |

## Validation-selected modality comparison

| Domain | View | Full model | Single modality | Single model | Full鈭抯ingle paired mean 卤 std (pp) |
|---|---|---|---|---|---:|
| arxiv | transductive | gin | text | gin | -0.55 卤 0.16 |
| arxiv | split_disjoint | gin | text | gin | -0.54 卤 0.15 |
| arxiv | temporal | gcnii | text | gcnii | -2.74 卤 0.58 |
| arxiv | group_inductive | gin | text | gcnii | -0.29 卤 0.14 |
| steam | transductive | gcnii | text | graphsage | +2.71 卤 0.88 |
| steam | split_disjoint | graphsage | text | graphsage | +2.43 卤 1.44 |
| steam | temporal | mlp | text | graphsage | +0.43 卤 0.81 |
| steam | group_inductive | graphsage | text | graphsage | +2.57 卤 1.41 |
| tmdb | transductive | graphsage | image | graphsage | +2.43 卤 0.52 |
| tmdb | split_disjoint | mlp | image | gcnii | +3.08 卤 0.46 |
| tmdb | temporal | graphsage | image | gcnii | +2.35 卤 0.49 |
| tmdb | group_inductive | graphsage | image | graphsage | +3.16 卤 0.50 |
| wikiart | transductive | graphsage | image | graphsage | +0.20 卤 0.44 |
| wikiart | split_disjoint | graphsage | image | graphsage | +0.20 卤 0.30 |
| wikiart | temporal | graphsage | image | graphsage | -0.12 卤 0.13 |
| wikiart | group_inductive | gcn | image | graphsage | -2.25 卤 0.63 |


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

## 2026-09-18 新完成：Q1 工件发布缺陷逃逸实验

本研究的参数独立于下方原始 full-benchmark 设置；原 2520 次训练与正式图/特征不改动。

### 设计与审计

- 真实图子集：arXiv 2048 节点/218 有向边；Steam 1387/18504；TMDB 2005/8132；WikiArt 2006/12090。固定 per-class sampling seed=20260918；3 个扰动/训练 seeds=0,1,2。
- 10 故障 × 4 域 × 3 seeds=120 故障候选/策略；clean、关系重命名、边重排共 36 正常对照/策略。3 策略共 468 次发布决策。
- 策略：训练标签同质性最高关系后直接发布；未经审计直接发布；**新增可选审计发布门**。前两者是程序策略，不是实测人类行为。原 serializer 并不具有本轮新门的 fail-closed 保证。
- 结果从落盘文件重新加载，通过独立 injection witnesses 判断残留；保留 candidate、clean comparator、contract、receipt、SHA-256 和 epoch curves。
- 1980/1980 校验通过，包括源图/embedding 哈希不变、发布存在性、残留缺陷和从曲线复算选择分数。不是独立模型实现或 prediction-level 重训练复现。
- 另训练 NC 144 + LP 144=288 次：GraphSAGE 2 层，hidden64，dropout0.2，Adam lr0.005，wd0.0005，60 epochs；LP 输出32维。NC validation accuracy、LP validation AP 选 checkpoint（故障 F9 除外）。

### 实际发布含缺陷工件数（每项分母 12）

| 策略 | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | 正常接受 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 同质性优先 |12|12|12|12|12|12|12|12|12|12|36/36|
| 无审计发布 |12|12|12|12|12|12|12|12|12|12|36/36|
| 新增审计发布 |0|0|0|0|0|0|0|0|0|0|36/36|

### 未审计故障工件减干净工件：平均百分点差（各 n=12；F7 不运行）

| 编号 | 故障 | NC accuracy Δ | LP AUC Δ |
|---|---|---:|---:|
|F1|目标派生 same-label 边|+23.949|-2.039|
|F2|显式 one-hot 目标特征|+32.904|+4.520|
|F3|单个跨切分重复实体|+0.084|-0.034|
|F4|缺失时间当0|+3.394|+1.588|
|F5|一个群组成员跨切分|+0.156|+2.006|
|F6|违反 simple-graph 契约的自环|+0.146|-0.002|
|F7|端点悬空|N/A|N/A|
|F8|声明分层但执行按标签排序连续切分|-66.238|-0.003|
|F9|测试集选checkpoint|+0.793|+2.900|
|F10|特征行/实体版本错配|-36.755|-7.165|

完整 seed/domain 数字、范围及曲线见 `benchmark/results/artifact_release_20260918/summary.json` 与各域 `_results.json`。汇总 `REPORT.md`；复核入口 `benchmark/validate_artifact_release_study.py`。

### 论文采用与限制

正文 §3 Q1 替换旧耗时 Q1，并由 Intro 引用；旧耗时记录保留附录。新增 Appendix F 记录矩阵及全部 signed effects。结论：可运行或高分不能证明完整性，发布门把明确约束转为可重复检查。**不写“实验证明比人工构造好”**：未招募人工参与者、未衡量自然错误率，也不是 multi-agent vs single-agent 消融。

F2 不是原始文本标签留存后重新 CLIP 编码；F8 不是宣称任何非分层划分都错误；F6 不代表所有 GNN 都禁止自环。F4/F5/F8 改变 cohort，F1 可改变 LP 任务边，相关差值不能统一称为因果“虚高”。arXiv 子图稀疏，LP 波动尤其大。门只拒绝，未展示修复后成功发布；依赖可信 contract，对隐藏代理标签/伪造来源/未知故障不作保证。


> 本文件是实验数字与状态的唯一权威来源；所有核心结果均使用 sanitized graph 与明确的 protocol view。

## 2026-09-19 新完成：审稿补强实验（#4 划分鲁棒性 / #5 验证选模 / #6 契约 n=3 / #7 开销 / #8 LP 披露）

> 运行器：`benchmark/run_split_seed2_nc.py`（隔离输出到 `results/newdomain_nc_split2/`，**不写**生产 protocol_views/manifests；超参镜像生产 run_protocol_matrix 默认值：epochs 300 / hidden 256 / layers 2 / dropout 0.5 / proj_dim 768 / init seeds 0–4）、`benchmark/run_heldout_contract.py`（Amazon/Reddit 契约构建+审计）。**2026-09-20 已写入论文**：§5.1（7/16 一致率 + 开销句）、§5.3（划分鲁棒段）、§5.6（Amazon/Reddit 扩写 + tab:heldout）、§4.4（LP 暴露约束句）、附录 B（Partition robustness 段 + tab:splitseed2 + Held-out contract builds 段）；摘要/引言/Q2/附录 A 同步。注意复跑范围是**融合模态网格**（单模态 cells 未重跑），论文措辞已按此限定。

### #5 划分鲁棒性：split_seed=2 的固定 GraphSAGE trans−split 降幅

| Domain | seed=1 drop（生产） | seed=2 drop | seed=2 五种子范围 | trans acc（seed2 vs 生产） |
|---|---:|---:|---|---|
| arXiv | +1.26pp | **+1.04pp** | 0.66~1.37 | 90.55 vs 90.77 |
| Steam | +2.29pp | **+0.75pp** | 0.00~1.61 | 68.71 vs 68.43 |
| TMDB | +0.89pp | **−0.17pp** | −0.61~0.08 | 58.25 vs 59.15 |
| WikiArt | +9.75pp | **+10.02pp** | 9.83~10.42 | 89.06 vs 88.95 |

结论（诚实版）：**大效应稳定（WikiArt ~10pp 在两个划分下都在），小效应是划分敏感的**——TMDB 的 0.89pp（摘要区间的下界）在第二划分下消失并轻微翻负；Steam 从 2.29 衰减到 0.75；arXiv 1.26→1.04 与 WikiArt 保持稳定。四域 trans 绝对值与生产差：全精度最大 0.9077pp（TMDB；表显 0.90），论文按"within one point"表述。注意逐模型绝对精度并非跨划分不变——Steam GIN 移动 2.43pp、WikiArt GCNII 2.06pp——故论文的稳定性声明限定为 fixed-GraphSAGE。

### #4 验证-测试选模一致率：7/16 = 44%

split_seed=2 套件记录了逐模型 best-val accuracy（生产留存工件只有测试指标，此统计只能在新跑中测）。16 个 域×协议 cell 里，**验证选出的模型与测试 argmax 相同的仅 7 个**（✓：arxiv trans/split、steam split/group、tmdb temporal、wikiart trans/split；✗ 含 wikiart temporal val→graphsage(0.163) vs test→gin(0.014) 等）。该数字把论文"descriptive maxima 不等于部署估计"的声明量化了——验证选择与测试最大值的一致率不到一半，模型排序不应过度解读。注意口径：测于 seed=2 套件（val/test 同一划分内比较）。

### #6 契约复用 n=1→n=3：Amazon、Reddit 全 PASS

`run_heldout_contract.py` 从 e4 存档 schema 转写 YAML（amazon: asin/main_cat/shared_brand；reddit: post_id/subreddit/same_author+same_flair），经**生产 config_driven 路径**确定性建图（0.6s/1.1s；5,364/7,257 节点、12,249/30,585 边），再跑与 Spotify 完全相同的 `audit_domain`：**两域 9/9 检查全 PASS**（raw/schema/五阶段计划/节点唯一/端点/关系声明/血缘/协议溯源）。产物：`results/heldout_contract_20260919.json`；旧异构图已备份为 `*_legacy_heterogeneous.json`。诚实边界：这是合同级复用（无训练、无多模态编码）；reddit 的 created_utc 未验证为可信时间字段，故 temporal 不适用；reddit 关系语义粗糙（61% 空 flair、[deleted] 作者 18%），cap=30 限制其影响。

### #7 开销汇总（Efficiency 维度）

- 训练：split_seed=2 套件实测（4090，300 epochs 上限），6 模型平均单次训练 arxiv 3.8–4.2s / tmdb 1.6–2.3s / wikiart 2.4–5.0s / steam 0.5–0.9s；**每域全部四协议×6模型×5种子 ≈ 1–8 分钟**。
- 构造：held-out 域图构建 0.6–1.1s；schema 阶段 1.4–4.3s、1–2 次调用（E2）。
- 另有 2026-07-12 的 per-epoch/显存/参数量表（GCN/GAT/GraphSAGE，剪枝图，`results/newdomain_efficiency/`，RTX 4090）可作 OpenMAG Q9/Q10 式对照，引用须带日期与范围限定。

### #8 LP 归纳视图消息图饥饿（代码事实，供论文披露句）

`run_four_protocol_graph_tasks.prepare_splits`：消息图 = `bidirectional(trp)`（仅训练正样本）。归纳视图（split/temporal/group）中 val/test 节点在该图上**零邻居**——GNN 编码器对这些端点退化为纯节点特征编码（与 MLP 同型）。因此 LP AUC 从 trans→temporal 的下降中有一部分是 harness 的结构暴露约束，而非纯任务难度；聚类共享同一编码器。建议论文在 Task exposure rules 附录加一句后果说明（措辞待作者定），并可用 LP 表中 MLP 行作"不依赖消息图"的对照参考。

## 1. 实验环境与统一规则

- 代码：`/home/dev_qw/DataCrawlerAgent`
- 环境：`/home/dev_qw/miniconda3/envs/agent_env`
- GPU：NVIDIA RTX 4090
- NC split seed：1
- 初始化 seeds：0–4，共 5 次
- NC Epoch 上限：300；训练和 validation early stopping 沿用 `train_and_eval`
- LP/聚类共享 graph-autoencoder：Epoch 上限 200，hidden=128，validation AP 选 checkpoint；MLP/GCN/GAT/GraphSAGE/GIN/GCNII × 5 seeds
- Modality matching：Epoch 上限 50，hidden=256，validation AP 选 checkpoint；5 seeds
- 模型：MLP、GCN、GAT、GraphSAGE、GIN、GCNII
- 指标：accuracy 与 macro-F1，报告 mean±std
- 表征：冻结 encoder；逐模态 L2 normalize；element-wise mean fusion；再次归一化

普通 transductive 跨 split 边是合法协议条件，不自动叫 leakage。Topology-only probe 只作诊断。

## 2. 四个成熟数据集

| Dataset | Nodes | Labels | Raw relation type | Time provenance | Group provenance | Main modalities |
|---|---:|---:|---|---|---|---|
| arxiv | 45,672 | 4 | citation | arXiv ID year-month | extracted first author | text/image/formula |
| steam | 2,760 | 8 | co-attribute | release_date | primary developer | text/image |
| tmdb | 13,037 | 18 | co-attribute | release_date | primary director | text/image |
| wikiart | 21,968 | 64 | co-attribute | year | artist | text/image |

### 2.1 Provenance-confirmed exclusions

| Dataset | Excluded before all four views | Reason |
|---|---|---|
| arxiv | none | citation 不是由 target category 字段直接生成 |
| steam | `shared_genre` | relation source 与 target genre 同源 |
| tmdb | `shared_genre` | relation source 与 target genre 同源 |
| wikiart | `shared_style` | relation source 与 target style 同源 |

Raw MAG 文件不修改；所有 protocol views 从 sanitized base graph 派生。

## 3. 四种协议

| Protocol | Node masks | Edge view | Interpretation |
|---|---|---|---|
| transductive | stratified 60/20/20 | full sanitized topology | 标准转导条件 |
| split-disjoint | same masks as transductive | within-split edges | 结构暴露敏感性 |
| temporal | chronological 60/20/20; time bucket intact | within-split edges | 面向未来数据 |
| group-inductive | primary group assigned as a whole | within-split edges | 面向未见实体组 |

所有 manifests 满足 group overlap=0，split-disjoint/temporal/group-inductive cross-split edge=0。Steam temporal 排除 4 个缺时间节点，TMDB 排除 1 个，WikiArt 排除 3,078 个；arxiv 时间覆盖 100%。

## 4. 四协议融合模态 NC 主表

| Dataset | Protocol | Best model | Accuracy | Macro-F1 | MLP accuracy | Best GNN−MLP | Directed edges | Train/Val/Test |
|---|---|---|---:|---:|---:|---:|---:|---|
| arxiv | transductive | GIN | **0.9165** | 0.9152 | 0.8914 | +0.0251 | 114,090 | 27,402/9,133/9,137 |
| arxiv | split-disjoint | GCNII | 0.8999 | 0.8990 | 0.8914 | +0.0085 | 50,312 | 27,402/9,133/9,137 |
| arxiv | temporal | GCNII | 0.8664 | 0.8599 | 0.8556 | +0.0108 | 70,706 | 27,402/8,752/9,518 |
| arxiv | group-inductive | GIN | 0.9028 | 0.9021 | 0.8884 | +0.0144 | 52,010 | 27,403/9,135/9,134 |
| steam | transductive | GCNII | **0.6846** | 0.4925 | 0.6621 | +0.0225 | 96,492 | 1,652/548/560 |
| steam | split-disjoint | MLP | 0.6621 | 0.4737 | 0.6621 | −0.0007 | 42,774 | 1,652/548/560 |
| steam | temporal | GraphSAGE | 0.6179 | 0.4329 | 0.6122 | +0.0057 | 51,860 | 1,637/561/558 |
| steam | group-inductive | GraphSAGE | 0.6569 | 0.4449 | 0.6232 | +0.0337 | 47,482 | 1,655/553/552 |
| tmdb | transductive | GraphSAGE | **0.5915** | 0.3939 | 0.5892 | +0.0023 | 325,818 | 7,815/2,600/2,622 |
| tmdb | split-disjoint | MLP | 0.5892 | 0.3619 | 0.5892 | −0.0066 | 144,706 | 7,815/2,600/2,622 |
| tmdb | temporal | GraphSAGE | 0.5470 | 0.3147 | 0.5461 | +0.0009 | 194,700 | 7,799/2,610/2,627 |
| tmdb | group-inductive | GraphSAGE | 0.5775 | 0.3768 | 0.5728 | +0.0048 | 173,914 | 7,820/2,608/2,609 |
| wikiart | transductive | GraphSAGE | **0.8895** | 0.8733 | 0.6477 | +0.2418 | 887,108 | 13,159/4,374/4,435 |
| wikiart | split-disjoint | GraphSAGE | 0.7920 | 0.7419 | 0.6477 | +0.1443 | 389,172 | 13,159/4,374/4,435 |
| wikiart | temporal | GCNII | 0.0101 | 0.0090 | 0.0066 | +0.0035 | 657,620 | 11,237/3,805/3,848 |
| wikiart | group-inductive | GCN | 0.5665 | 0.4354 | 0.4395 | +0.1270 | 887,108 | 13,122/4,421/4,425 |

## 5. Protocol sensitivity

固定每域 transductive winner，按相同初始化 seed 配对计算：

| Dataset | Compared view | Accuracy drop | Paired bootstrap 95% CI | Model-rank correlation |
|---|---|---:|---|---:|
| arxiv | split-disjoint | +0.0169 | [0.0146, 0.0192] | 0.829 |
| arxiv | temporal | +0.0535 | [0.0490, 0.0581] | 0.657 |
| arxiv | group-inductive | +0.0137 | [0.0119, 0.0159] | 0.829 |
| steam | split-disjoint | +0.0329 | [0.0232, 0.0436] | 0.771 |
| steam | temporal | +0.1119 | [0.1008, 0.1247] | 0.812 |
| steam | group-inductive | +0.0412 | [0.0189, 0.0645] | 0.943 |
| tmdb | split-disjoint | +0.0089 | [0.0044, 0.0134] | 0.829 |
| tmdb | temporal | +0.0445 | [0.0409, 0.0489] | 0.943 |
| tmdb | group-inductive | +0.0140 | [0.0109, 0.0168] | 0.943 |
| wikiart | split-disjoint | +0.0975 | [0.0942, 0.1007] | 0.429 |
| wikiart | temporal | +0.8829 | [0.8810, 0.8848] | +0.058 |
| wikiart | group-inductive | +0.3235 | [0.3155, 0.3332] | 0.429 |

> **2026-09-19 勘误（rank correlation 口径）**：上表 Model-rank correlation 原用 `np.argsort` 秩 + Pearson（并列取字典序），wikiart temporal 因 mlp=graphsage 精确并列（0.0066008）得到 −0.086。改用 tie-averaged `scipy.stats.spearmanr` 重算全部 12 格：**仅两格变化**（steam temporal .829→.812、wikiart temporal −.086→**+0.058**, p=0.91），其余 10 格无并列、两法等价。n=6 模型下双侧 5% 临界值约 ±0.83，故 0.429/0.771/0.812 均不显著，引用时必须带此限定。论文三处引用已同步修正（A.1 表 + 附录两段）。

### 5.1 Temporal label support 与 prior shift

| Dataset | Train-visible / global classes | Unseen classes present in test | Coverable test mass | Largest train-visible test class | Seen-class prior TVD | JS (nats) |
|---|---:|---:|---:|---:|---:|---:|
| arxiv | 4/4 | 0 | 1.0000 | 0.3823 | 0.2096 | 0.0280 |
| steam | 8/8 | 0 | 1.0000 | 0.4695 | 0.0873 | 0.0058 |
| tmdb | 18/18 | 0 | 1.0000 | 0.1804 | 0.2334 | 0.0326 |
| wikiart | 45/64 | 18 | **0.3885** | 0.0990 | **0.9650** | **0.6455** |

WikiArt temporal 的训练集缺少全局 64 类中的 19 类；test 中实际出现 18 个训练未见类。训练可见标签只覆盖 1,495/3,848=38.85% 的 test，因此 train-support oracle ceiling 是 0.3885，而实际 accuracy 0.0101 仅为该上限约 2.6%。所以 label emergence 只解释天花板；极高的可见类 prior TVD/JS 与 severe prior shift 一致，剩余差距还可能包含 covariate shift 和模型失配。该行属于 open-set temporal stress test，不能与前三域 closed-label drop 直接比较，也不构成 leakage 证据。

## 6. 四协议模态消融

每个 modality 内按 accuracy 选择最好模型；负 gap 表示完整融合不如最好单模态。

| Dataset | Protocol | Best single (model) | Single acc | Full fusion (model) | Full acc | Full−single |
|---|---|---|---:|---|---:|---:|
| arxiv | transductive | text (GIN) | 0.9214±0.0010 | text+image+formula (GIN) | 0.9165±0.0024 | −0.0049 |
| arxiv | split-disjoint | text (GIN) | 0.9057±0.0014 | text+image+formula (GCNII) | 0.8999±0.0012 | −0.0058 |
| arxiv | temporal | text (GCN) | 0.8929±0.0018 | text+image+formula (GCNII) | 0.8664±0.0021 | −0.0264 |
| arxiv | group-inductive | text (GCNII) | 0.9013±0.0019 | text+image+formula (GIN) | 0.9028±0.0011 | +0.0014 |
| steam | transductive | text (GraphSAGE) | 0.6636±0.0075 | text+image (GCNII) | 0.6846±0.0147 | +0.0211 |
| steam | split-disjoint | image (GraphSAGE) | 0.6432±0.0059 | text+image (MLP) | 0.6621±0.0103 | +0.0189 |
| steam | temporal | text (GraphSAGE) | 0.6072±0.0090 | text+image (GraphSAGE) | 0.6179±0.0052 | +0.0108 |
| steam | group-inductive | text (GraphSAGE) | 0.6312±0.0039 | text+image (GraphSAGE) | 0.6569±0.0171 | +0.0257 |
| tmdb | transductive | image (GCNII) | 0.5725±0.0023 | text+image (GraphSAGE) | 0.5915±0.0021 | +0.0191 |
| tmdb | split-disjoint | image (MLP) | 0.5617±0.0016 | text+image (MLP) | 0.5892±0.0030 | +0.0275 |
| tmdb | temporal | image (GraphSAGE) | 0.5242±0.0044 | text+image (GraphSAGE) | 0.5470±0.0040 | +0.0228 |
| tmdb | group-inductive | image (GCNII) | 0.5552±0.0023 | text+image (GraphSAGE) | 0.5775±0.0035 | +0.0224 |
| wikiart | transductive | image (GraphSAGE) | 0.8883±0.0029 | text+image (GraphSAGE) | 0.8895±0.0018 | +0.0013 |
| wikiart | split-disjoint | image (GraphSAGE) | 0.7876±0.0023 | text+image (GraphSAGE) | 0.7920±0.0027 | +0.0044 |
| wikiart | temporal | image (GAT) | 0.0084±0.0023 | text+image (GCNII) | 0.0101±0.0041 | +0.0018 |
| wikiart | group-inductive | image (GCN) | 0.5999±0.0080 | text+image (GCN) | 0.5665±0.0109 | −0.0334 |

结论：Steam 与 TMDB 的融合收益在四个 protocol 下都为正，平均分别为 +1.91pp、+2.30pp；Arxiv 单独 text 在三种 view 更好；WikiArt 由 image 主导，group-inductive 下融合反而下降 3.34pp。融合价值是 domain- and protocol-dependent，不能写成“多模态总会更好”。

### 6.1 六任务四协议的统一评估合同

图中六项任务现均使用同一批 sanitized protocol views，但按任务语义区分信息暴露：

- **Node classification**：train labels 监督训练，validation 选 checkpoint，test 只评估。
- **Link prediction**：transductive 将无向边固定拆为 60/20/20；其余协议分别以 train/val/test 节点内部边作为正例。消息图仅含 train positives；validation/test positives 从不进入消息传递。固定均匀负例与正例 1:1，并排除该 view 的全部真边。
- **Node clustering**：与 LP 共享不使用标签训练的 graph-autoencoder；推理时可见该协议允许的 split 内拓扑。KMeans 只在 train-node embeddings 上拟合，在 test nodes 上计算 NMI/ARI；标签仅决定 `k` 和评分，不训练表示、不选择模型。表中模型统一由 validation AP 选择，避免按 test NMI 挑模型。
- **Modality retrieval**：冻结模态编码，在 test cohort 内做同实体精确检索，报告两个方向平均 R@1/MRR。
- **Modality alignment**：在 test cohort 描述性计算 linear CKA 与 paired-vs-shifted cosine gap，不拟合 test labels。
- **Modality matching**：train/val/test 实体完全隔离；同标签 hard negatives 仅用于抑制类别捷径，标签不进入模型输入；validation AP 选 checkpoint，test 报 AUC/AP。

Transductive 与 split-disjoint 的节点 masks 相同，所以不消费图边的 retrieval/alignment 结果必须相同；审计已验证这一不变量。它们的相同不是重复实验错误，而是明确说明 node-local frozen representation 不受 edge-only intervention 影响。

### 6.2 Link prediction 与 node clustering 完整主表

每行模型按 mean validation AP 预先选择；AUC/AP/NMI/ARI 均为 5 seeds mean±std。所有 6 模型逐 seed 结果保存在 raw JSON，未用 test 指标挑选下表模型。

| Dataset | Protocol | Selected model | LP AUC | LP AP | Clustering NMI | Clustering ARI |
|---|---|---|---:|---:|---:|---:|
| arxiv | transductive | GIN | 0.9286±0.0012 | 0.9370±0.0013 | 0.1371±0.0517 | 0.0261±0.0360 |
| arxiv | split-disjoint | MLP | 0.8904±0.0006 | 0.8677±0.0007 | 0.4973±0.0238 | 0.4176±0.0480 |
| arxiv | temporal | MLP | 0.8594±0.0013 | 0.8407±0.0021 | 0.4333±0.0177 | 0.3426±0.0230 |
| arxiv | group-inductive | MLP | 0.8885±0.0004 | 0.8674±0.0006 | 0.4876±0.0043 | 0.3918±0.0051 |
| steam | transductive | GraphSAGE | 0.9081±0.0044 | 0.9073±0.0039 | 0.1405±0.0028 | 0.0566±0.0026 |
| steam | split-disjoint | MLP | 0.7735±0.0031 | 0.7685±0.0015 | 0.1435±0.0041 | 0.0657±0.0049 |
| steam | temporal | GAT | 0.6963±0.0062 | 0.7094±0.0086 | 0.1344±0.0062 | 0.0336±0.0179 |
| steam | group-inductive | MLP | 0.8208±0.0010 | 0.8277±0.0009 | 0.1563±0.0041 | 0.0800±0.0072 |
| tmdb | transductive | GraphSAGE | 0.9174±0.0025 | 0.9071±0.0026 | 0.1191±0.0025 | 0.0414±0.0020 |
| tmdb | split-disjoint | MLP | 0.8510±0.0010 | 0.8283±0.0013 | 0.0937±0.0054 | 0.0244±0.0031 |
| tmdb | temporal | GAT | 0.7503±0.0061 | 0.7232±0.0047 | 0.1042±0.0136 | 0.0275±0.0107 |
| tmdb | group-inductive | GCN | 0.8034±0.0059 | 0.7910±0.0066 | 0.1166±0.0167 | 0.0363±0.0091 |
| wikiart | transductive | GAT | 0.9921±0.0004 | 0.9905±0.0004 | 0.6239±0.0033 | 0.2912±0.0043 |
| wikiart | split-disjoint | GAT | 0.9091±0.0011 | 0.9068±0.0009 | 0.5942±0.0033 | 0.2629±0.0092 |
| wikiart | temporal | GCN | 0.6750±0.0196 | 0.6763±0.0170 | 0.3369±0.0219 | 0.1245±0.0278 |
| wikiart | group-inductive | GAT | 0.8571±0.0014 | 0.8553±0.0020 | 0.7204±0.0122 | 0.4129±0.0182 |

LP 在所有域中从 transductive 到更严格的 inductive cohort 均有明显变化；这说明任务难度受部署协议控制，不代表 transductive 本身“泄漏”。聚类的变化不与 NC 单调一致，尤其 arxiv transductive 的 validation-AP-selected GIN 聚类较弱，说明“适合重建边”不等于“最适合类别聚类”，不能用一个任务替代另一个任务的效用结论。

### 6.3 Modality retrieval、alignment 与 matching 完整主表

Retrieval 是双向平均；matching 是 5 seeds mean±std。Arxiv 分别评估 text–image、text–formula、image–formula；其余域评估 text–image。模态缺失/零向量节点先排除，实际 train/val/test paired counts 已逐 cell 保存在 raw JSON。

| Dataset | Protocol | Pair | Retrieval R@1 | MRR | CKA | Cos gap | Match AUC | Match AP |
|---|---|---|---:|---:|---:|---:|---:|---:|
| arxiv | transductive | text–image | 0.1074 | 0.1498 | 0.5540 | 0.0318 | 0.8943±0.0052 | 0.8809±0.0067 |
| arxiv | transductive | text–formula | 0.0106 | 0.0151 | 0.2933 | 0.0076 | 0.8368±0.0102 | 0.8100±0.0110 |
| arxiv | transductive | image–formula | 0.0006 | 0.0027 | 0.2841 | 0.0029 | 0.7558±0.0025 | 0.7266±0.0033 |
| arxiv | split-disjoint | text–image | 0.1074 | 0.1498 | 0.5540 | 0.0318 | 0.8943±0.0052 | 0.8809±0.0067 |
| arxiv | split-disjoint | text–formula | 0.0106 | 0.0151 | 0.2933 | 0.0076 | 0.8368±0.0102 | 0.8100±0.0110 |
| arxiv | split-disjoint | image–formula | 0.0006 | 0.0027 | 0.2841 | 0.0029 | 0.7558±0.0025 | 0.7266±0.0033 |
| arxiv | temporal | text–image | 0.1279 | 0.1740 | 0.5935 | 0.0372 | 0.8809±0.0055 | 0.8638±0.0066 |
| arxiv | temporal | text–formula | 0.0263 | 0.0347 | 0.3348 | 0.0087 | 0.8329±0.0115 | 0.7987±0.0134 |
| arxiv | temporal | image–formula | 0.0009 | 0.0029 | 0.3028 | 0.0033 | 0.7586±0.0050 | 0.7216±0.0066 |
| arxiv | group-inductive | text–image | 0.1057 | 0.1470 | 0.5519 | 0.0320 | 0.8928±0.0048 | 0.8794±0.0056 |
| arxiv | group-inductive | text–formula | 0.0093 | 0.0138 | 0.3015 | 0.0074 | 0.8360±0.0052 | 0.8134±0.0051 |
| arxiv | group-inductive | image–formula | 0.0006 | 0.0028 | 0.2912 | 0.0028 | 0.7674±0.0058 | 0.7366±0.0054 |
| steam | transductive | text–image | 0.6223 | 0.7084 | 0.4071 | 0.1507 | 0.9348±0.0030 | 0.9171±0.0059 |
| steam | split-disjoint | text–image | 0.6223 | 0.7084 | 0.4071 | 0.1507 | 0.9348±0.0030 | 0.9171±0.0059 |
| steam | temporal | text–image | 0.4543 | 0.5720 | 0.4934 | 0.1268 | 0.8736±0.0353 | 0.8504±0.0421 |
| steam | group-inductive | text–image | 0.5815 | 0.6869 | 0.4410 | 0.1483 | 0.9305±0.0042 | 0.9137±0.0064 |
| tmdb | transductive | text–image | 0.8496 | 0.8949 | 0.4491 | 0.1810 | 0.9633±0.0021 | 0.9563±0.0030 |
| tmdb | split-disjoint | text–image | 0.8496 | 0.8949 | 0.4491 | 0.1810 | 0.9633±0.0021 | 0.9563±0.0030 |
| tmdb | temporal | text–image | 0.7850 | 0.8426 | 0.3697 | 0.1627 | 0.8962±0.0042 | 0.8807±0.0039 |
| tmdb | group-inductive | text–image | 0.8466 | 0.8968 | 0.4695 | 0.1830 | 0.9627±0.0013 | 0.9550±0.0019 |
| wikiart | transductive | text–image | 0.1329 | 0.1967 | 0.2106 | 0.0732 | 0.8150±0.0016 | 0.8033±0.0027 |
| wikiart | split-disjoint | text–image | 0.1329 | 0.1967 | 0.2106 | 0.0732 | 0.8150±0.0016 | 0.8033±0.0027 |
| wikiart | temporal | text–image | 0.0849 | 0.1240 | 0.1281 | 0.0445 | 0.6679±0.0037 | 0.6629±0.0048 |
| wikiart | group-inductive | text–image | 0.1137 | 0.1761 | 0.2543 | 0.0709 | 0.7367±0.0014 | 0.7037±0.0038 |

结果显示：Steam/TMDB 的 text–image 对齐和可匹配性较强；WikiArt 的 temporal cohort 在 retrieval、CKA、cosine gap 和 matching 上同时下降；Arxiv text–image 明显强于包含 formula 的组合，尤其 image–formula 检索接近零。这里应写成 encoder/pair compatibility 与 cohort shift，而不是把 matching 高分直接解释成通用语义对齐。

### 6.4 六任务结果规模与 correctness audit

- NC：64 dataset×protocol×modality cells、384 model cells、1,920 seeded runs。
- LP + clustering：16 dataset×protocol cells、每 cell 6 models×5 seeds；共 480 次 label-free graph-autoencoder 训练，同时产生 480 组 LP 与 480 组 clustering 评估。
- Retrieval + alignment：24 protocol×modality-pair cells、48 个检索方向；全部为冻结表示上的确定性评估。
- Matching：24 protocol×modality-pair cells、120 seeded runs。
- 新六任务审计：**2,665/2,665 PASS**。检查包含输入哈希、协议 cell 完整性、seed 身份、checkpoint 范围、正负例平衡、正边 split 互斥、held-out positives 不进入消息图、实体 overlap=0、retrieval 单调性/范围、CKA 范围、所有 mean/std 从逐 seed 重算，以及最终文档 40 行结果与 JSON 精确一致。

性能数字不“证明”框架正确；框架正确性仍由 provenance、raw/derived 分离和 protocol invariants 支持。六任务只证明这些经过审计的产物能在多类任务中被一致消费，并揭示不同任务和协议的适用性边界。

## 7. Topology-only null diagnostics

实验包括 100 次训练标签置换和 10 次 degree-preserving double-edge-swap；所有 40 个重连图均通过度数序列逐点校验。输入图先移除 target-derived relations。

| Dataset | Observed | Label-permutation null | Degree-preserving null | Obs−perm | Obs−rewire | Test coverage |
|---|---:|---:|---:|---:|---:|---:|
| arxiv | 0.6208 | 0.2967 | 0.5349 | +0.3241 | +0.0859 | 0.4605 |
| steam | 0.5839 | 0.4813 | 0.5482 | +0.1026 | +0.0357 | 0.9071 |
| tmdb | 0.3696 | 0.1637 | 0.3174 | +0.2058 | +0.0521 | 0.9336 |
| wikiart | 0.8618 | 0.0234 | 0.8221 | +0.8384 | +0.0397 | 0.9605 |

Observed 明显高于 label-permutation null，说明真实拓扑携带标签相关信号；但相对 degree-preserving null 的增量仅为 3.57–8.59pp，说明其中很大部分可由度数结构与邻域覆盖解释。该实验不作 leakage 裁决，也不能替代 provenance 审计。

## 8. Framework 与 schema 证据

### 8.1 自动化成本 E2

六域 × 三 seeds 的已存结果显示 schema discovery/audit 使用约 1–2 次 LLM calls、约 1.1k–3.0k tokens、约 1.4–4.3 秒。25–90 分钟人工时间是专家估计，不是受试者实验。

当前生产代码已将 `SchemaDiscoveryAgent` 放入 Collector 与 Builder 之间；LLM plan 即使遗漏该阶段或最终 Verifier，编排器也会补齐强制质量门。成熟域走 YAML-confirm 模式，真正新域可走 LLM-discover 模式，但二者都必须通过同一确定性 validator。

### 8.2 Held-out schema discovery E4/Exp6

Amazon、Reddit 与四个 arxiv 子域均通过基本 build validity 与 target-relation decontamination，但质量审计发现 arxiv_cs/math 选择 `license`，arxiv_condmat/astroph 产生 843/275 类。结论是 validity 不等于 benchmark quality，必须保留报告卡门；不能写成“6/6 高质量泛化”。

### 8.3 Loop robustness E6

旧 32-run 实验可支持“确定性 target-relation 排除稳定、LLM label discovery 仍脆弱”。不得继续写“LP 高分修复率 100%”或“self-verifying”；统一称 audit-capable pipeline。

### 8.4 MAG landscape

| Work | Published scope | 与本文的边界 |
|---|---|---|
| MAGB (KDD 2025) | 多领域 text/image MAG，GNN-as-Predictor 与 VLM-as-Predictor | 本文不竞争单纯模型数量；重点是从真实源自动构建、provenance audit 和四协议发布 |
| MM-GRAPH (CVPR 2025) | 7 个 text/image graph 数据集和多种图学习框架 | 本文补的是 schema-to-benchmark 自动化与 protocol-conditioned integrity，而非再造一个静态数据集包 |
| OpenMAG (arXiv 2026) | 19 数据集、6 域、16 encoders、24 models、8 tasks | 本文的差异是数据生成与审计流水线、raw/derived view 分离和 integrity card |

一手来源：[MAGB](https://arxiv.org/abs/2410.09132)、[MM-GRAPH](https://arxiv.org/abs/2406.16321)、[OpenMAG](https://arxiv.org/abs/2602.05576)。此表用于定位贡献，不据此声称现有 benchmark 全部存在泄漏。

### 8.5 生产复用与最终证据审计

- `run_reusability_audit.py` 在 Steam、TMDB、WikiArt 与 held-out Spotify 上统一检查 YAML task、deterministic schema audit、五 agent 强制计划、节点 ID、关系声明和边端点，四域均 PASS。**2026-09-19 扩展**：Amazon、Reddit 经生产路径建图后以同一审计全 PASS（见顶部 #6 节），held-out 契约复用从 n=1 扩为 n=3。
- Spotify 源没有可信 timestamp，因此只声明三种适用协议；没有用随机顺序冒充 temporal。
- `validate_final_evidence.py` 对所有逐 seed 数值、temporal label support 和聚合结果做 1,739 项重算检查，1,739/1,739 PASS；规模为 64 个 dataset×protocol×modality cells、384 个 model cells、1,920 个 seeded runs。
- 自动测试当前为 8/8 PASS，覆盖时间桶、group 隔离、cross-edge pruning、通用 YAML metadata、schema lineage、强制质量门和 raw/derived graph 分离。

## 9. Integrity cards

当前 report card 已去除旧 NC、旧 modality-missing 和旧 LP 的混合字段，只从当前四协议矩阵、protocol manifests、data-quality audit 与 topology nulls 生成。四域 provenance、duplicate/entity、self-loop/exact relation duplicate、edge endpoint 与 protocol invariants 全部通过。Applicability 结论：

- arxiv：高特征可预测性，模型区分度有限；图增益小。
- steam：图价值随 protocol 变化，split-disjoint 下 MLP 最好。
- tmdb：异质结构，GNN 相对 MLP 增益很小。
- wikiart：图价值显著，但对结构暴露、未见艺术家和时间 label emergence 非常敏感。

旧 report card 中的 LP 数值保留为 topology diagnostic，不再参与 correctness pass/fail。

## 10. 其他已有实验的使用规则

- TMDB genre-word attribution：剥离 genre 词后 retrieval R@1 仅下降约 0.004，可用于排除“显式 genre 词导致检索高分”的解释。
- 原 LP、聚类和跨模态脚本/表仍是旧口径，不得引用；当前可引用版本只来自 `benchmark/results/four_protocol_tasks/`。Few-shot、跨域、noise、neighbor、steering、IGC 仍未按 sanitized four-protocol 口径重跑，不进入论文核心数字。
- OpenMAG/PyG 的旧 LP 高分结果只能重命名为 external topology predictability/exposure，不得写成已发表数据集“存在结构泄露”的证明。
- E7 合成同质性 sweep 只能说明受控条件下 GNN−MLP gap 会随 topology 改变；不支持真实域上的普适因果律或固定 crossover。
- CLIP-vs-ImageNet 对照只说明 encoder sensitivity；当前没有 sample-level overlap 证据，不报告“污染量”。

## 11. 尚未完成且不能伪造的实验

| Experiment | Status | Blocker/next action |
|---|---|---|
| 四协议模态消融 | complete | 64 cells × 6 models × 5 seeds |
| Topology permutation + degree-preserving null | complete | 100 permutation + 10 rewires/domain；度数序列校验通过 |
| 最终证据一致性审计 | complete | 1,739/1,739 checks；所有逐 seed 均值、temporal support 和聚合 headline 重算一致 |
| 生产 YAML 复用合同 | complete | Steam/TMDB/WikiArt/held-out Spotify 均 PASS；Spotify temporal 因无可信时间而 unavailable。2026-09-19 增：Amazon/Reddit 同审计全 PASS（n=3 held-out） |
| 第二划分种子划分鲁棒性 | complete | split_seed=2 全套 480 runs（2026-09-19）；小降幅划分敏感、WikiArt 大降幅稳定 |
| 验证-测试选模一致率 | complete | 7/16=44%（2026-09-19，seed=2 套件实测；生产留存工件无 val 指标，不可回溯计算） |
| 六任务四协议 benchmark | complete | NC 1,920 seeds；LP/聚类 480 joint seeds；matching 120 seeds；retrieval/alignment 24 cells |
| 六任务独立 correctness audit | complete | 2,665/2,665 checks PASS；输入哈希、逐 seed 重算和最终文档一致 |
| Sample-level pretraining overlap | blocked | 需要可查询的预训练样本索引/指纹库 |
| Exp3 真人效率 baseline | blocked | 需要真实受试者与计时协议 |
| Exp4b 外部用户复用 | blocked | 需要真实外部用户 |
| MAG landscape | complete | 已按 MAGB、MM-GRAPH、OpenMAG 一手论文定位差异 |

## 12. 可复现产物

- `benchmark/data/protocols.py`
- `benchmark/generate_protocol_views.py`
- `benchmark/run_newdomain_nc.py`
- `benchmark/run_four_protocol_graph_tasks.py`
- `benchmark/run_four_protocol_modality_tasks.py`
- `benchmark/aggregate_validate_four_protocol_tasks.py`
- `benchmark/aggregate_protocol_results.py`
- `benchmark/run_topology_nulls.py`
- `benchmark/aggregate_protocol_modalities.py`
- `benchmark/results/protocol_views/`
- `benchmark/results/newdomain_nc/protocol_matrix/`
- `benchmark/results/four_protocol_tasks/{graph,modality}/`
- `benchmark/results/four_protocol_tasks/six_task_summary.{json,md}`
- `benchmark/results/topology_nulls.{json,md}`
- `benchmark/results/report_cards.{json,md}`
- `benchmark/validate_final_evidence.py`
- `benchmark/sync_protocol_manifests.py`
- `benchmark/results/final_evidence_audit.{json,md}`
- `benchmark/run_reusability_audit.py`
- `benchmark/results/reusability_audit.{json,md}`
- `tests/test_protocols.py`
- `tests/test_framework_integrity.py`
- `tests/test_four_protocol_tasks.py`

## 13. 框架机制实验（2026-09-15，已完成）

### 13.1 Paired schema-role comparison

四域 Steam/TMDB/WikiArt/Spotify × generation seeds 0–2，共 12 paired cases。每域用 Random(20260915) 抽取最多 300 个条目生成字段画像；确定性反馈使用完整数据。模型 endpoint 返回 qwen-plus，每请求 temperature=0.2、max_tokens=1200、timeout=35s、无自动重试。请求 seed 不保证服务端完全确定性。

每个 case 先生成一个 proposal，两臂共享它；single-agent arm 沿用原对话修订，independent reviewer arm 用新对话接收同样的 profile、proposal 和 validator feedback。按 seed 交替修订请求顺序。参考 label/ID 仅用于评分，不进入修订反馈。两臂逻辑调用预算相同，文本封装与角色提示不同，实际 tokens 不完全相同。

| Arm | Cases | Parsed / valid / publishable | ID agreement | Base target agreement | Revision tokens | API errors |
|---|---:|---:|---:|---:|---:|---:|
| Single-agent revision | 12 | 12 / 12 / 12 | 12/12 | 9/12 | 18,434 | 0 |
| Independent schema reviewer | 12 | 12 / 12 / 12 | 12/12 | 9/12 | 18,476 | 0 |

Publishability 的实验定义是 production validator.valid，且去除 target-derived relations 后至少一个关系 shared_fraction≥0.05。这是 schema 阶段的可发布候选条件，不代表真正完成构图或任务适用性已验证。

配对结果：both pass=12、reviewer-only=0、single-only=0、neither=0。没有观察到角色拆分的 publishability 增益；小样本且检查出现 ceiling，不能宣称两臂统计等价，更不能声称端到端多智能体优于单智能体。

Spotify 三组两臂都选择 playlist_subgenre，参考为 playlist_genre。TMDB seed 0/1 的 single arm 选择 genres、reviewer 选择 genres[0]；base-field agreement 忽略索引，因此不能替代精确标签语义一致性。总实际 API calls=36（12 initial + 24 revisions），total_tokens=52,891。没有新增 NC/LP/matching 训练。

### 13.2 Production-validator fault injection

对四域真实数据和清理了 target-derived relations 的 configured schema 使用内存副本，分别注入 duplicate ID、missing ID、nonexistent target field、nonexistent relation field、target-derived relation，共 20 个已知缺陷；原 schema 与 harmless relation renaming 共 8 个阴性对照。

相同 production validator 的输出在 harness 内采用三种 decision policy；不修改生产代码或原始数据。成功拦截定义为 reject candidate，或者排除注入的 target-derived relation。

| Policy | Faults prevented | False rejections | Total case evaluations |
|---|---:|---:|---:|
| Full checks | 20/20 | 0/8 | 28 |
| Without schema rejection | 4/20 | 0/8 | 28 |
| Without lineage exclusion | 16/20 | 0/8 | 28 |

去掉 schema rejection 后仅保留 4 个 target-derived relation 排除；去掉 lineage exclusion 后仍拒绝 16 个 ID/不存在字段缺陷。这说明两个判定在测试缺陷集上职责互补。未覆盖 semantic alias、近重复或全部 protocol violations，不称为端到端 pipeline ablation。

### 13.3 核验与产物

- 离线核验 114/114 PASS：输入数据 SHA-256、12 个唯一 case、84 个 fault/control variant rows、逐 case 与聚合一致、共享 proposal 交付、反馈无参考一致率字段、响应完整性、tokens 与计数重算。
- `tests/test_framework_integrity.py` 与 `tests/test_protocols.py`：8/8 PASS。
- Runner：`benchmark/run_framework_mechanisms.py`；离线核验：`benchmark/validate_framework_mechanisms.py`。
- 输出目录：`benchmark/results/framework_mechanisms_20260915/`，包含 12 个 domain_seed JSON、results.json、summary.json、validation.json；完整保存 prompts/responses/usage/model/seed。
- 本实验不并入 2,520 个 downstream seeded training runs，也不追加到此前 1,739/2,665 项审计计数。

### 13.4 论文表述校正

NC/fusion historical headline 是按 test mean 选择架构的 descriptive maxima；LP/clustering 仍按 validation AP 选择架构。Intro 新采用固定 GraphSAGE 的回顾性分析：四域 transductive→split-disjoint drop 为 1.26/2.29/0.89/9.75pp。相关代码与输入哈希记录在论文目录的 empirical_reanalysis.py / empirical_investigation.json。

当前机制实验闭合的是 schema-role 对照与 production-validator 指定缺陷消融。端到端 matched-budget coordination baseline、真人效率与外部用户复用仍未完成。
