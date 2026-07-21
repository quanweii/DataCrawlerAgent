# 实验路线图（NeurIPS D&B）— 2026-07-11

> 配套 `docs/paper_logic_DnB_FINAL_2026-07-11.md`（逻辑线 FINAL）。
> 产出自 6-agent Workflow（4 审计 + 合成 + 完备性批判），亲读 `benchmark/results/`、`docs/OpenMAG(1).pdf`、数据集产物核实。
> **目标**：补全论文所需的全部实验（agent + benchmark），对标 OpenMAG 5 维度，不写初稿。
>
> 🔴 **2026-07-15 两次修复（本文数字全部作废，仅留执行史）**：
> 1. Mean 融合修复 (GIN 训练崩溃, grad-clip)
> 2. **Text 去泄露修复** (label 明文字段 "Style:"/"Genres:" 在 text 特征中)
> 
> **最终权威**见 [`docs/benchmark_text_decontamination_2026-07-15.md`](benchmark_text_decontamination_2026-07-15.md)
> 与 `benchmark/results/newdomain_nc/modality_ablation_decontaminated_2026-07-15.json`。
> 下方文字保留作执行记录，**所有含 text 的绝对数字一律勿引用**。 全≤多数类 |
| P0-4 spotify 零代码新域 | ✅ | **已建出**：10000 节点/17440 边/homo 0.741/text-only，NC+LP+leak-profile 齐全；无 spotify agent 类 → C3 证据坐实 |
| P0-5 LP Hits@3/10 | ✅ | 改 **on-the-fly 向量化过滤 Hits@k**（1000 random filt neg，标准协议）；5 域全非退化（file-based ratio=1 退化 + 100× 硬负采样太慢，已弃） |
| P0-6 英文 card/licensing | ✅ | `DATASET_CARD_en.md` + `DATASET_LICENSING_en.md`（含 spotify 第 5 域） |
| P0-7 IGC canonical 冻结 | ✅ | 4 个 `.bak` → `archive_pre100eval/` + README |
| P0-8 可复现 manifest | ✅ | `REPRODUCIBILITY.md` + `requirements-pinned.txt` + `checksums.sha256`（85 行） |
| P0-9 Ethics | ✅ | `ETHICS_STATEMENT.md`（ToS/robots/PII/consent/dual-use） |

**执行中修复的 2 个真问题**：① spotify 文本含 genre 标签→泄露（MLP 0.951）→ 移除 genre 重编码→0.635；② LP Hits@3/10 ratio=1:1 退化（全 1.0）→ on-the-fly 1000 过滤负→非退化。

**5 域同质性谱现完整**：tmdb 0.24 → steam 0.52 → spotify 0.74 → wikiart 0.89 → arxiv 0.93。

---

## §A1. P1 完成状态（2026-07-12 更新，15/15）

**P1 全部完成**。关键诚实纠正（写论文必守）：

| P1 | 状态 | 关键产出 / 纠正 |
|---|---|---|
| P1-1 arxiv NC 重聚合 | ✅ | 42 cell（6模型×7模态）；formula 消融 no-op +0.0006~0.0041 |
| P1-2 聚类 NMI/ARI | ✅ | 新驱动；高同质 GNN 助聚类，低同质无效（与 steering 一致） |
| P1-3 few-shot k=3/10 | ✅ | 新驱动；高同质 few-shot GNN 胜（wikiart k10 gcn 0.823>mlp 0.773） |
| P1-4 arxiv 编码器族 | ✅ | **clip_large 赢 arxiv**(gcnii 0.9061>base 0.8984>siglip 0.8501)；**域依赖发现**：SigLIP 赢 steam(mlp 0.727>0.694) 但输 arxiv(−5.6pt)——SigLIP 64-token 截断长论文，CLIP 分块+mean-pool 全文→证 per-domain encoder profiling 必要 |
| P1-5 steering p/CI | ✅ | **r=0.846[0.600,0.960] p=0.0005 n=12**（旧 r=0.463/n=16 废止） |
| P1-6 跨域双协议 | ✅ | linear-probe 12/12 负 Δ-0.258 / FT 中性 |
| P1-7 跨域 LP | ✅scope | scope→NC-only（LP 边域特有，跨域无意义） |
| P1-8 E2 功效 | ✅ | **McNemar p=0.0625 边际**（旧"0.714≫rule"夸大→"边际非决定"） |
| P1-9 E2 文件对齐 | ✅ | 坏件归档，reconstructed 权威 |
| P1-10 verifier 回路 | ✅ | **0.918→0.880(删关系不足)→0.024(跨切分剪枝 C2)** |
| P1-11 复用 manifest | ✅ | steam/tmdb/wikiart 94.6-97.2%；**arxiv 例外 68-78%**；spotify 零代码确认 |
| P1-12 DGF/LGMRec 迁域 | ✅ | wikiart+steam NC + wikiart LP；与标准 GNN 持平，LGMRec 硬负略优 |
| P1-13 主表显著性 | ✅ | Wilcoxon **n=5 floor p=0.0625**（无法到 α=0.05）；leak-free 表 MLP 全域竞争力强 |
| P1-14 模态缺失鲁棒 | ✅ | **text 主导**（drop_text wikiart -0.120）；image 边际 |
| P1-15 数据质量审计 | ✅ | 5 域 **0% 重复节点/边/自环/边泄露** |

新驱动脚本：`benchmark/run_newdomain_{clustering,fewshot,modality_missing}.py` + `scripts/{run_verifier_reverify_wikiart,data_quality_audit}.py`。

---

## §A2. P2 补充完成（2026-07-12，OpenMAG 广度缺口闭合）

补完红色(必补)+黄色(建议补)实验，闭合 OpenMAG 广度短板：

| P2 | 维度 | 状态 | 关键结果 |
|---|---|---|---|
| P2-11 模态检索/对齐 | Data Quality | ✅ | steam t→i R@1=0.469/MRR0.576；wikiart R@1=0.111（弱对齐）；**tmdb image 512-d≠text 768-d 编码器不一致→检索 N/A**（数据质量备注，需重编码）|
| P2-1 效率分析 | Efficiency Q9/Q10 | ✅ | 4 域 GCN/GAT/GraphSAGE time+peak_mem：随节点数 scale，GAT 最贵(3-4×)，arxiv gcn 18ms/706MB；全域秒-分钟级训练 |
| P2-2 标签噪声 | Robustness Q7 | ✅ | tmdb+wikiart GCN/GAT/GraphSAGE × rho{0,0.2,0.4,0.6}：GCN/GAT 鲁棒(≤-0.032)，**graphsage 最敏感(tmdb -0.089)**=精度/鲁棒 trade-off |
| P2-10 多 split 方差 | 统计严谨 | ✅ | tmdb 3 split_seed×5 init：**split-std≤0.017≈init-std**（除 gin），单 split 头条稳定→堵"5-seed 单 split"质疑 |

**OpenMAG 对照更新**：模态任务(检索/对齐)扩到新域✓、效率新 4 域✓、鲁棒全 4 类噪声(feature/edge/label/modality)✓、统计严谨(显著性+多 split)✓。**仍 scope out**：G2Text/G2Image（质量差/空，FINAL §4 硬约束）、trainable encoder（frozen 定位）、24 模型全测（10 模型够）。新驱动：`run_newdomain_{modality,efficiency,labelnoise}.py`。

**★ 2026-07-12 数据一致性修复：tmdb image 重编码 512-d→768-d★**。审计发现 tmdb 海报被旧脚本用 CLIP-base(512-d) 编码（其余全域 headline 均 768-d），已用 CLIP-large 重编码为 768-d（旧备份入 `_archive_512d_bug/`），**全域 10 个 headline embedding 现统一 768-d**。tmdb text+image 特征 1280→1536-d，全量重跑级联（NC/LP/leak-profile/perseed/multisplit/cluster/fewshot/modmiss/advanced/labelnoise/检索）+ 重聚合 leak_profile/significance/multisplit。**关键确认**：768-d 几乎没改 NC/LP（text 主导，P1-14），但解锁了 tmdb 模态检索（N/A→R@1=0.783，全域最强）。详见 `docs/tmdb_768d_unification_2026-07-12.md`。

**★ 2026-07-12 公平性修复：全域 newdomain 实验开 768-d 投影★**（OpenMAG 公平模态对比协议）。先前新域路径 `build_features` 只拼接不投影（text 768 / t+i 1536 / t+i+f 2304 → 多模态参数更多，模态消融被参数量混淆）。现 6 个 newdomain 驱动默认 `--proj-dim 768`（learnable Linear→768 再进 GNN），与旧 arxiv openmag 路径 + OpenMAG 对齐。**全量重跑**：NC(全模态×5域)/LP(5域)/cluster/fewshot/modmiss/labelnoise/advanced/perseed/multisplit + 重聚合 leak_profile/significance/multisplit。**核心公平结论**：投影后 text 单独最强(3/4域)，仅 steam t+i 微胜(+0.021)；formula +0.001 no-op（更干净）。印证 text 主导/丰富度非性能（P1-14/P1-1 强化）。lp_ceiling 结构指标不变。详见 `docs/projection_768d_fairness_2026-07-12.md`。
新增 docs：clustering/fewshot/modality_missing/advanced_models/main_table_significance/e2_power_boost/cross_domain_{dual_protocol,lp_scope}/code_reuse_manifest/data_quality_audit（均 _2026-07-12.md）。

---

## §A. 图例

- **状态**：✅done（产物齐全）/ 🟡partial（有但要补）/ ❌missing（无产物）/ 🔵aspirational（有代码无结果，慎用）
- **优先级**：**P0**=拒稿级必补 / **P1**=大修强烈建议 / **P2**=nice-to-have
- **贡献映射**：C1 数据集 / C2 去泄露协议 / C3 框架+skill / C4 benchmark 价值 / cross=跨贡献基础设施
- **OpenMAG 5 维**：Necessity / Data Quality / Effectiveness / Robustness / Efficiency

---

## §B. OpenMAG 镜像清单（5 维度）

| 维度 | OpenMAG 做了 | 我们已有 | 待补 | 状态 |
|---|---|---|---|---|
| **Necessity** | Q1 单/多模态消融；Q2 邻居数 sweep 证结构有用 | steering(E1 皇冠)、融合消融、E2 citation、IGC greedy=oracle | 统一 uni/multi NC+LP 模态偏好图；steering 当 Q2 答案（生成式邻居 sweep 出 scope） | 🟡partial |
| **Data Quality** | Q3 finetune-vs-frozen；Q4 编码器策略；Appendix A 构建文档；CLIP-L/14 标准；3-run | **leak-free 协议（OpenMAG 没有的护城河）**、LP 三档负采样、逐关系 homophily、5-seed（>OpenMAG 3）、dataset card、CLIP/SigLIP embedding | arxiv metadata 回填；统计源对齐；card 英文化；leak-profile 表；编码器版本/dim 明示；可选 trainable arm | 🟡partial |
| **Effectiveness** | Q5 ~24 模型 × NC/LP/聚类（Acc/F1/MRR/Hits@3/NMI/ARI）；Q5b 检索/G2Text/G2Image | NC(6 模型×5 seed×4 域)+LP(6×3 regime×4 域)、融合、边关系预测 | 聚类 NMI/ARI；LP Hits@3/10；few-shot 新三域；高级多模态 GNN(DGF/LGMRec) 迁到新域 | 🟡partial |
| **Robustness** | Q7 特征/视觉/标签噪声；Q8 边扰动 | 特征+边噪声(4 域,GCN)、邻居消融、LP 去饱和 | **模态缺失鲁棒(Q7b，多模态 benchmark 必备)**；标签噪声；多模型；significance test | 🟡partial |
| **Efficiency** | Q9 复杂度；Q10 训练时/显存 vs 精度 | 仅 arxiv：理论复杂度+实测(9 模型)；NC 记录 training_time | steam/tmdb/wikiart 实测时+显存；time-vs-acc 图（D&B 构建向优先级低） | 🟡partial |

---

## §C. P0 拒稿级缺口（提交前必须闭合）

| # | 实验 | 贡献 | 状态 | 计划（脚本/数据/产出） | 工作量 |
|---|---|---|---|---|---|
| P0-1 | **arxiv 旗舰 agent.json metadata 回填** | C1 | ❌ | 现状：metadata.statistics 是 stub（total_images=0、无 homophily/num_labels/avg_degree），与 text+image+formula 卖点矛盾。从 `processed/statistics.json`(37210 有图/74122 字符/45531 有公式)+`manifest.json`(homo 0.933)+节点标签(4 类)重写，对齐 steam/tmdb/wikiart schema | ~1h |
| P0-2 | **跨产物统计对齐（单一权威表）** | C1 | 🟡 | 4 源不一致：arxiv 边 57045(agent/card/manifest) vs 57590(statistics)；num_classes 7(bug) vs 4(真)；steam 54642 vs 54424。修 statistics num_classes 7→4，定 canonical 边数（raw vs unique 双标），四源对齐一张表 | ~2h |
| P0-3 | **leak-profile 表（C1/C2 头条差异化产物）** | C2 | ❌ | thesis 承诺"每数据集附 leak-profile 一等元数据"，但散在 leakfree_*.json/igc_*.json/lp_*_summary.json。聚合为 `benchmark/results/leak_profile_table.json`+md 表：每域 LP-ceiling 原/剪、多数类基线、删边数、逐关系 homophily、定义性边标记(wikiart shared_style=1.0)、LP random/hard/hard_degree AUC、NC 保持 acc | ~3h |
| P0-4 | **Spotify 零代码新域——必须真建（C3 关键证据）** | C3 | ❌ | steam/tmdb/wikiart 都带代码迭代（crawler 子类~100 LOC），**只有 spotify 纯 YAML 零代码首触构建**才是"1 YAML=1 域"的诚实证据。重生成 spotify_tracks_5k.json（Spotify API 或公开 Kaggle 快照）→ 持久化 `data/spotify_raw/` → skill Phase 5(build_graph_from_config) → Phase 6 全管线 NC/LP/leak-profile → **零 Python 编辑**，捕获 skill 日志；若需改 Python，如实报为 skill bug | ~1-2 天 |
| P0-5 | **LP Hits@3/Hits@10 恢复** | C1 | 🟡 | `tasks/link_prediction.py:155` 算了 hits@3/10，但 `run_newdomain_lp.py:200` 硬编码只留 {auc,ap,mrr,hits@1}，:270 聚合同。locked scope 要求 MRR+Hits@1/3/10。改一行+重跑（per-seed 未持久化，可能要全重跑 4 域×6 模型×3 regime×5 seed） | ~4-8h GPU |
| P0-6 | **Dataset card + licensing 英文化** | C1 | 🟡 | `DATASET_CARD.md`/`DATASET_LICENSING.md` 仅中文，NeurIPS 要英文。9 节 datasheet 译英 + Datasheets-for-Datasets 合规声明 + OpenMAG Appendix-A 式逐域构建段（源/边定义/预处理/模态） | ~1 天 |
| P0-7 | **IGC canonical run 冻结 + 文档对齐** | C2 | 🟡 | 数字漂移：.bak(Jul3 tmdb oracle 0.255/closed_loop 0.315) vs current(Jul9 0.300/0.401)；MEMORY 引 Jul3、FINAL 引 Jul9；closed_loop std 0.060→0 未释。指定 Jul9 为 canonical，.bak 移 archive/+README，对齐 MEMORY/FINAL 数字，加 reproducibility 注（deterministic, split_seed=1） | ~2h |
| P0-8 | **可复现 manifest：pin 版本 + SHA256** | cross | ❌ | `requirements.txt` 用 `>=` 下界无 CUDA/PyG-build pin；无任何产物 SHA256。冻 pinned requirements + 记 torch.cuda/PyG build/encoder hash(CLIP-ViT-L/14,SigLIP) + 算每个 `_agent.json`/split/embedding 的 SHA256 → `REPRODUCIBILITY.md`+checksums.sha256 | ~0.5 天 |
| P0-9 | **Ethics & 采集合规段（超出版权）** | cross | ❌ | DATASET_LICENSING 只讲版权层级，没讲 Steam/TMDb/WikiArt **采集伦理**：API ToS 合规、robots.txt、限流策略、是否采集 UGC/PII(Steam 评论/TMDb 评分)、同意、dual-use。写 `ETHICS_STATEMENT.md` | ~1 天 |

---

## §D. P1 大修级（强烈建议）

| # | 实验 | 贡献 | 状态 | 计划 | 工作量 |
|---|---|---|---|---|---|
| P1-1 | NC 主表 arxiv summary 重聚合（含 formula 6 cell） | C1 | ✅done | 重生成 **42 cell**（6 模型×7 模态，全 leak_free_pruned 单协议）。formula 消融确认 no-op：text+image→+formula **+0.0006~+0.0041**；formula 单独最弱(0.788<text0.902<image0.818)。印证"formula=模态丰富度非性能" | done |
| P1-2 | 节点聚类 NMI/ARI（4 域） | C1 | ✅done | 新驱动 `run_newdomain_clustering.py`(leak-free+KMeans on GNN logits,5 seed)。高同质 GNN 助聚类(wikiart gcnii 0.848≫raw 0.637;arxiv gcnii 0.785≫raw 0.440)，低同质无效(steam/tmdb)。与 steering 一致。doc `clustering_results_2026-07-12.md` | done |
| P1-3 | Few-shot NC k=3/10（steam/tmdb/wikiart） | C1 | ✅done | 新驱动 `run_newdomain_fewshot.py`(leak-free 剪枝图+k/类标签,10 采样)。高同质 few-shot GNN 明显胜(wikiart k10 gcn 0.823>mlp 0.773)，低同质 MLP 胜。few-shot 下结构价值放大。doc `fewshot_results_2026-07-12.md` | done |
| P1-4 | arxiv 编码器族（旗舰目前 1 族，其余 3 族） | C1 | 🟡 | steam/tmdb/wikiart 有 clip_base/clip_large/siglip；arxiv 仅 1。`encode_domain.py` 给 arxiv 跑 clip_base+siglip(text+image) 再 NC | ~6-8h |
| P1-5 | steering GCN-only vs best-GNN 对齐 + r=0.463 的 p/CI | C4 | ✅done | **已修**：GCN-only 一致模型回归远强于旧值。GCN-MLP gap vs 同质性 **r=0.846 [boot 0.600,0.960] p=0.0005 n=12**；GCN_F1 vs homo r=0.913 [0.774,0.975] p=3.5e-5；fit gap=0.364·homo−0.332 crossover@0.91；leave-one-domain-out r=0.70–0.94。旧 r=0.463/n=16 废止（混 best-GNN 模型+误计 n）。产物 `steering/steering_causal_analysis_2026-07-12.json` | done |
| P1-6 | 跨域迁移诚实双协议报告 | C4 | ✅done | 双协议已报：linear_probe **12/12 负**(mean Δ-0.258)，full_finetune **中性**(mean Δ≈0)。doc `cross_domain_dual_protocol_2026-07-12.md` | done |
| P1-7 | 跨域 LP 迁移（4 benchmark 域） | C4 | ✅done(scope) | **scope 缩为 NC-only**（roadmap 合法选项）：LP 边是域特有 co-attribute 关系(无共享语义)，跨域 LP 迁移语义无意义；FINAL 已只称"跨域 linear-probe 12/12 负"。doc `cross_domain_lp_scope_2026-07-12.md` | done |
| P1-8 | E2 citation fuzzy-bucket 统计功效增强（唯一 agent 必要性证据） | cross | ✅done | **McNemar 配对检验（n=60）+ bootstrap CI**（无新 LLM 调用，更诚实）。结论软化了旧头条：5/5 不一致对偏向 LLM vs shortlist **p=0.0625（边际）**，vs production **p=0.625（ns）**；combined non-clean(n=17) LLM 0.824 vs prod 0.706 vs shortlist 0.529。FINAL §3/§4 已改为"边际非决定"。重跑扩 fuzzy 桶为可选 future work（endpoint 换模型不可比）。产物 `e2_power_boost_2026-07-12.json`+md | done |
| P1-9 | E2 coverage-curve 文件对齐 | cross | ✅ | `e2_coverage_curve.json`(Jun23 坏的 RateLimitError 全 0) 与 `e2_coverage_curve_reconstructed.json`(Jul3 真 4/146) 共存。删/标旧件，只留 reconstructed，呈现"端到端瓶颈=语料覆盖而非 matcher" | ~30min |
| P1-10 | verifier↔builder 回路：一次干净 re-verified 跑 | C3 | ✅done | **确定性测量驱动 3-round wikiart trace 已归档**（`igc/wikiart_verifier_reverify_trace.json/.log`）。诚实路径：R0 verify 0.918(标 shared_style homo1.0 定义性) → R1 删 shared_style 0.880 → **R2 re-verify FAIL**(same_artist homo0.877 仍泄露) → **R3 跨切分剪枝(C2) 0.024 PASS**。关键发现：**删关系不足，跨切分边剪枝才是去泄露机制**——精确印证 FINAL §3。LLM verifier 仅编排包装(非 load-bearing) | done |
| P1-11 | 代码复用 manifest（94% 可审计） | C3 | ✅done | 可审计 manifest：steam/tmdb/wikiart 复用 **94.6–97.2%**；**arxiv 例外 68–78%**(2037 行 citation/OCR 引擎)；spotify 零代码 CONFIRMED(0 Python/41 YAML)；builder 155→31 半可审计。doc `code_reuse_manifest_2026-07-12.md`+csv | done |
| P1-12 | 高级多模态 GNN（DGF/LGMRec）迁到≥1 新域 | C4 | ✅done | DGF/LGMRec/MMGCN/MGAT 迁到 wikiart+steam(NC)+wikiart(LP)，0 新模型代码(get_model 通用)。诚实：高级模型与标准 GNN 持平(wikiart MMGCN 0.910≈gcnii 0.910)，LP 硬负上 LGMRec 略优(hard_deg 0.910 vs gcn 0.869)。doc `advanced_models_results_2026-07-12.md` | done |
| P1-13 | **4 域主表配对显著性检验（Wilcoxon）** | cross | ✅done | 加 per-seed 持久化+Wilcoxon(4 域)。诚实：**n=5 Wilcoxon floor p=0.0625**，无法到 α=0.05；报效应方向。leak-free 表上 MLP 全域竞争力强(GNN 仅 arxiv 最高同质边际正)。causal 主张靠 steering(r=0.846)。doc `main_table_significance_2026-07-12.md` | done |
| P1-14 | **模态缺失鲁棒（Q7b）** | C1 | ✅done | 新驱动 `run_newdomain_modality_missing.py`(zero/50%-dropout 模态块)。**text 主导**：drop_text wikiart -0.120/tmdb -0.034；**image 边际**(drop_image ≈0，tmdb 反 +0.007)。与 formula no-op 一致=模态丰富度非性能。doc `modality_missing_results_2026-07-12.md` | done |
| P1-15 | **数据质量证明：去重/重叠/多边审计** | C2 | ✅done | 5 域全审：**0% 重复节点/重复边/自环/边泄露**；LP 全 transductive(87-99.7% 节点重叠，符合预期)；图全 simple graph。verdict: 支持 leak-free+dedup 声称。脚本 `scripts/data_quality_audit.py`，doc `data_quality_audit_2026-07-12.md` | done |

---

## §E. P2 nice-to-have

| # | 实验 | 贡献 | 计划 | 工作量 |
|---|---|---|---|---|
| P2-1 | steam/tmdb/wikiart 效率剖析 | cross | `run_newdomain_nc.py` 加 time_per_epoch+peak_memory_mb，GCN/GAT/GraphSAGE×4 域→time-vs-acc 图 | ~4h |
| P2-2 | 鲁棒：标签噪声+多模型 | cross | `run_newdomain_robustness.py` 加 label-noise(翻标签)≥1 域；wikiart 扩到 6 模型 | ~6h |
| P2-3 | LP steering 跨 3 域聚合 | C4 | `steering_lp_all.json` 只聚 wikiart，并 steam+tmdb | ~2h |
| P2-4 | 静态 leaderboard | C1 | 从 lp/nc summary 生成 NC acc/F1 + LP MRR/Hits@k 排行（md 或 GitHub Pages） | ~0.5 天 |
| P2-5 | Add-new-domain 贡献者指南（英文） | C3 | `ADD_NEW_DOMAIN.md`：YAML→skill Phase5/6→NC/LP→leak-profile，以 spotify 为例 | ~0.5 天 |
| P2-6 | 永久 release DOI（Zenodo/HF） | C1 | P0 修完后打包 4 `_agent.json`+splits+embeddings+leak-profile+card，Zenodo(arxiv CC0+3 域 structure-only)+HF | ~1 天 |
| P2-7 | Trainable-encoder arm（Q3 镜像） | cross | ≥1 域端到端 BERT/ViT trainable，比 T/V frozen/train 四设置；或 arxiv q3 null 结果支撑"frozen 足够" | ~2-3 天 |
| P2-8 | 逐域模态偏好图（uni vs multi，Q1 镜像） | cross | 从现有 newdomain_nc 抽 uni(text)/uni(image)/multi，画分组柱+逐域偏好标注，arxiv 加 uni(formula) | ~3h |
| P2-9 | 自监督/对比 baseline | C1 | `SelfSupervisedGNN`(advanced_models 已有)仅 arxiv；迁 wikiart+steam linear-eval | ~1 天 GPU |
| P2-10 | 多 split 方差（3 split_seed） | C1 | 现 5 seed 单 split_seed=1。tmdb(低同质,split 敏感)跑 3 split_seed 报 split-variance 带 | ~0.5 天 GPU |
| P2-11 | 模态检索扩到新域 | C1 | modality_matching 仅 arxiv；用现有 embedding 跑 steam/tmdb/wikiart text→image Recall@1/5/10 | ~0.5 天 |

---

## §F. 已稳（DONE-secure，只需呈现，勿重复造）

NC 主表(74 文件,6 模型×5 seed×4 域,leak-free,LP-ceiling 双报) · LP 三档(4 域×6×3 regime×5 seed,缺 hits@3/10) · leak-free NC+LP-ceiling(4 域全≤多数类) · 跨域 NC 迁移(12/12 linear-probe 负) · steering E1(MLP-constant 自检+GCN 穿越) · 融合消融(逐域学习权重) · 边关系预测(steam .86/wikiart .98) · 模态匹配(arxiv text-image .85) · E2 citation fuzzy(LLM .714 vs rule 0,n=7) · IGC greedy=oracle(tmdb/wikiart) · encoder 比较(arxiv CLIP-large .895 最优)。

---

## §G. 执行顺序（依赖感知，~3-4 周）

- **Phase 0 — 数据完整性冻结（~2-3 天）**：P0-1 arxiv metadata 回填 → P0-2 统计对齐 → P0-7 IGC canonical 冻结 → P0-9 E2 文件清理 → P0-6 card/licensing 英文化。（多为 bookkeeping）
- **Phase 0b — 严谨性与证明（~2 天，批判新增）**：P0-8 可复现 manifest(pin+SHA256) → P1-13 主表显著性检验 → P1-15 去重/重叠审计 → P0-9b(P0-9 已列) ethics 段。
- **Phase 1 — 头条资产组装（~2-3 天）**：P0-3 leak-profile 表 → P0-5 LP hits@3/10 恢复 → **P0-4 spotify 必须真建** → P1-1 NC arxiv 重聚合。
- **Phase 2 — benchmark 核心缺口（~2-3 天 GPU）**：P1-2 聚类 → P1-3 few-shot 新三域 → P1-4 arxiv 编码器族。
- **Phase 3 — C4 价值严谨（~2 天）**：P1-5 steering 对齐+CI → P1-6 跨域双协议 → P1-7 跨域 LP → P1-8 E2 功效增强。
- **Phase 4 — 广度/鲁棒（~2-3 天）**：P1-12 DGF/LGMRec 迁域 → P1-14 模态缺失鲁棒 → P2-1 效率 → P2-2 标签噪声 → P1-10 verifier 干净跑 → P1-11 复用 manifest。
- **Phase 5 — 发布打包（~1-2 天，数据修完后）**：P2-4 leaderboard → P2-5 贡献者指南 → P2-6 Zenodo/HF DOI → licensing 复核。

---

## §H. Agent 诚实红线（每处都要守）

agent 的 load-bearing 支柱**只有三个**，每个都有诚实约束：

1. **语义 wrangling（LLM 唯一胜规则处）**= E2 citation fuzzy：LLM recall 0.714 vs rule 0.000/0.286。
   - 约束：n=7 太小→增强(P1-8)；scale-up 4/146 recovery **不是** scale 上的 agent 必要性，必须如实报"端到端瓶颈=语料覆盖非 matcher"；注模型溯源；删坏的全 0 文件。
2. **可复用零代码接口**（框架+skill）。
   - 约束：**仅在 spotify 真零代码首触建出，或 claim 缩到 steam/tmdb/wikiart 共享 builder + YAML 声明式**。现"spotify 建出(5k/79k)"是假声称，不得出现。出复用 manifest 让 94% 可审计。
3. **端到端编排**（verifier-builder 回路）。
   - 约束：必须呈现脆弱性——只 1 次干净 verify→rebuild（无再 verify），2/3 崩溃/跳过，日志 LLM 计数幻觉。要么产 1 次干净 re-verified 跑，要么谨慎 scope。

**绝不能声称**（已证伪）：agent 去泄露/建图决策强于规则。greedy(无 LLM)=oracle on S1b blind（tmdb 0.3002=0.3002，wikiart 0.8796=0.8796）；**closed_loop 反而比 greedy 差**（tmdb 0.4005>0.3002）。把 greedy=oracle 当**优点**写（协议 principled+robust，不依赖昂贵/不稳定 LLM）。**披露 steam 例外**：greedy 在 steam 保留全部关系（早停阈值被 baseline 满足），oracle 能找 shared_genre。

**去泄露归因**：LP-ceiling 坍塌（wikiart 0.921→0.024，4 域全≤多数类）是 NC 跨切分**边剪枝**（C2 测量/协议操作）的功劳，**不是 LLM、也不是单纯删 shared_style**。归到 C2 协议。逐域机制精确（steam 上 greedy 保留全部关系，机制不同）。

**DO-NOT-USE（硬约束，不作 benchmark 任务引用）**：G2Image(空目录) / G2Text(仅 2 低质初步文件) / Reddit-Amazon 异构 GNN RGCN/HAN/HGT(仅代码零结果)。

---

## §I. Reviewer 攻击面 + 诚实重定位（批判产出，写论文时逐条防）

| 攻击面 | 诚实重定位 |
|---|---|
| Formula"独有模态"但消融 +0.0007（no-op） | 当**模态丰富度/完整性**卖，不当性能；或找数学密集子集证 formula 有用 |
| "agent 建了数据集"但 greedy=oracle | agent 只在语义 wrangling(citation)挣到位置；"agent-built"是编排非决策必要，每句保持边界清晰 |
| "自校验框架"但回路只成 1 次、closed_loop<greedy | 改称"audit-capable pipeline"；披露脆弱；scope 到单次干净 wikiart 跑 |
| "零代码/94% 复用"但 94% 不可审计、spotify 未建 | 出 manifest；spotify 首触建出前 extensibility 未证 |
| "leak-free 靠 C2" 但机制逐域不同、0.045 应为 0.024 | 逐域精确归因；改 0.921→**0.024** |
| "6/6 跨域负"但仅 linear_probe | 报双协议：linear-probe 负(特征不迁移)、FT 中性(模型重学) |
| "主动设计同质性谱"但 4 域是自然源选择 | 数据集用"spanning natural sources"；"controlled/causal"**只**留给 steering |
| 5 seed 单 split 无 split 方差 | 补 tmdb 多 split_seed(P2-10) |

---

*生成于 2026-07-11，Workflow `wf_ba3391c0-966`（4 审计+合成+批判）。逻辑线见 `paper_logic_DnB_FINAL_2026-07-11.md`。*

---

## §A6. Path-B agent 自治实验（2026-07-21，C1 主线证据）

> 配套 [`docs/path_b_phase0_results_2026-07-21.md`](path_b_phase0_results_2026-07-21.md)（详细）+ 脚本 `benchmark/path_b/probe_{p1..p5}_*.py`。
> 目的：证伪/证实"agent 作为自治真实数据 MAG 构造框架"能否当 C1 主线。
> 模型 qwen-plus + DashScope text-embedding-v3(1024d)。分支 `feature/path-b-agent-enhancement`。

这组实验分两段：**P1–P3 是证伪门**（决定 agent 价值在哪、不在哪），**P4–P5 是 C1 主线正面证据**。

### P1–P3：证伪门（塑造 agent 的诚实 scope）

| 实验 | 脚本 | 核心结果 | 对论文 |
|---|---|---|---|
| **P1 schema 发现（弱正）** | `probe_p1_schema.py` | 藏 YAML，LLM 单次推断 4 域×3 seed。spotify 精度 0.50 vs 启发式 0.29，避开定义性 shared_genre；但 wikiart/tmdb 过度去泄露、**steam 2/3 seed 把 price 选成 label** | single-shot 发现**脆**→驱动 P4 加闭环 |
| **P2 引用对齐（❌死）** | `probe_p2_citation.py` | n=60 同候选短名单：**fuzzy 桶 emb 0.778 = LLM 0.778**（rule 0.111）；全量 emb 0.967>LLM 0.950 | **embedding 全复制语义匹配→agent 不必要**。撤"引用匹配 LLM 独有"主张 |
| **P3 语义审计（✅正但单次太噪）** | `probe_p3_audit.py` | 注 20 条语义脏数据进 200 wikiart：LLM 抓 mislabel/field_swap **1.00**（embedding 结构做不到），但 **clean FP 0.26**、跨批 dup 漏 0.20 | 唯一"无廉价代理"语义区；单次太噪，作 limitation/未来工作，不抬 C1 |

**门结论**：3 柱里 2 柱（P2 确定、P1 大部分）有廉价代理 → agent 价值不在"决策/匹配"。但这**驱动了 P4 的正确设计**：把 agent 价值放到"端到端自治 schema 发现 + verifier 去泄露"。

### P4–P5：C1 主线正面证据

| 实验 | 脚本 | 核心结果 | 支撑 |
|---|---|---|---|
| **P4 自治 skill 决定性测试** | `probe_p4_autonomous_skill.py` | LLM 发现 schema + 确定性 verifier 闭环（全程不问人），4 金标准域×3 seed：**硬门槛（build_valid AND decontam_correct）12/12**；label 复现 12/12；去泄露正确 12/12（一律按 C2 剔定义性关系，比 3/4 含泄露的人写金标准更一致）。归因：label 是 single-shot 选对（loop 不碰 label），loop 价值=去泄露+剪退化（确定性，稳健） | **C1 自治 schema 产出**（头条证据）|
| **P5-A label 抗 prompt 稳健性** | `probe_p5_robustness.py` (Part A) | 4 prompt 变体×4 域×2 seed single-shot：3/4 域全变体稳；steam 脆（V3/V4 选退化 `type`）。**verifier 闭环救回**：V4 steam round0 `type`→loop 测 n_classes=1→revise `genres` 2/2 | **闭环把脆弱 single-shot 变稳健自治** |
| **P5-B 新域端到端泛化** | `probe_p5_robustness.py` (Part B) | amazon（本地、脏、非 4 金标准）3 seed：**3/3 硬门槛过**——发现 label=`main_cat`(7 类≥20)+关系 `same_brand`(28612 边)+**避定义性 `same_main_cat`**+建真图(5438 节点/7 类/homophily 0.979) | **C1 新真实域泛化**（端到端）|

### 实验线↔逻辑线对应（读两线时的索引）

- **C1（agent 框架主线）** ← P4（12/12 自治产出）+ P5-A（抗 prompt）+ P5-B（新域泛化）
- **C2（测量去泄露）** ← verifier 的去泄露引擎 = P4/P5 的确定性 verifier + 原有 structure-leakage ceiling/CLIP 污染/text strip
- **agent 诚实红线** ← P2（引用不独有）+ IGC greedy=oracle + P1 single-shot 脆
- **C1 软肋 limitation** ← P1/P4 关系召回偏稀疏 + P5-A 替代 label 不被纠

### 3 个投稿前 TODO 状态
① label 稳健性 ✅过（P5-A）；② 关系召回 🟡仍稀疏（待加"多提议候选"步骤）；③ 端到端建图 ✅过（P5-B amazon）。

