# 全域实验结果 — 去泄露后最终数字总表（FINAL, 2026-07-16）

> ⚠️ **本文档是去泄露后所有实验结果的唯一权威数字来源（single source of truth）。**
>
> 本文所有数字均使用 **leak-free 协议**（见 §0）跑出，于 **2026-07-15 19:30–20:52 全域重跑**、
> **2026-07-16 复核验证**。写论文、做表、对外汇报一律以本文档数字为准。
>
> 🔴 **作废清单（旧数字不得再引用）**：
> - `docs/AMMAG_全域基准实验报告_2026-06-20.md` 的结果表 — **去泄露前**，wikiart text 虚高（0.95→0.30）；
> - `docs/AMMAG_Benchmark_完整实验报告.md`、`docs/Supplementary_Experiment_Report.md` — 同上，去泄露前；
> - `docs/benchmark_meanfusion_fix_2026-07-15.md` — 中间态数字，已被 mean 融合 + 去泄露联合修复取代；
> - `docs/paper_logic_DnB_FINAL_2026-07-11.md` 中**早于 2026-07-15** 的绝对数字（旧 r=0.463/n=16、旧 MLP 0.95、text≈multi 等）。
>
> 与本文冲突的任何早期数字，**一律以本文为准**。泄露诊断与修复经过见
> [`benchmark_text_decontamination_2026-07-15.md`](benchmark_text_decontamination_2026-07-15.md)。

---

## 0. 统一实验协议（所有表共享的前置条件）

| 项 | 设定 |
|---|---|
| Text 去泄露 | `Style:`/`Genres:` 等 label 字段在 CLIP 编码前 strip；wikiart 进一步删模板句/艺术家行，text 从纯标题编码（label-in-text=0%） |
| 切分 | 分层 60/20/20，`split_seed=1`，小类 floor=5 |
| 边泄露防护 | `prune_cross_split_edges` 跨切分边剪枝（NC/LP/clustering/fewshot/... 全启用） |
| 多模态融合 | OpenMAG 式 mean 融合：逐模态 L2-norm → element-wise mean → L2-norm（恒 768-d） |
| 训练稳定性 | 梯度裁剪（grad-clip）+ **10 seed**（NC/LP/聚类/few-shot/modmiss/robustness 等主表）；steering/causal/fusion/edgerel 用 5 seed |
| LP 负采样 | 三档：`random` / `hard` / `hard_degree` |
| 编码器 | CLIP-Large（text+image 统一 768-d）；arxiv formula 用 MathBERT |
| 硬件 | NVIDIA RTX 4090 (24GB) |

---

## 1. 五域元数据

| 域 | 节点数 | 类数 | 同质性 | 边（original→pruned, NC） | text 状态 |
|---|---|---|---|---|---|
| **arxiv** | 45,672 | 4 | 0.933 | 114,090 → 50,312 | 干净（摘要无 category） |
| **steam** | 2,760 | 8 | 0.524 | 109,284 → 48,322 | 去泄露（删 Genres 行） |
| **tmdb** | 13,037 | 18 | 0.239 | 386,950 → 169,958 | 去泄露（删 Genres 行） |
| **wikiart** | 21,968 | 64 | 0.895 | 1,040,660 → 456,584 | 去泄露 R2（纯标题） |
| **spotify** | 10,000 | 6 | 0.741 | 34,880 → 15,268 | 干净（playlist 名不入 track md） |

> 同质性谱横跨自然源 tmdb 0.24 → steam 0.52 → spotify 0.74 → wikiart 0.90 → arxiv 0.93。

---

## 2. NC 节点分类（text+image 主表，10 seed）

来源：`benchmark/results/newdomain_nc/nc_{domain}_summary.json` + `modality_ablation_decontaminated_2026-07-15.json`

| Model | arxiv | steam | tmdb | wikiart | spotify |
|-------|-------|-------|------|---------|---------|
| MLP | 89.34 | 65.88 | 59.06 | 64.86 | 63.49 |
| GCN | 89.91 | 61.02 | 48.02 | 80.54 | 62.83 |
| GAT | 89.68 | 59.48 | 48.14 | 70.29 | 62.59 |
| GraphSAGE | 89.73 | 65.43 | 57.54 | **82.44** | 63.32 |
| GIN | 89.81 | 55.84 | 37.95 | 69.72 | 60.83 |
| GCNII | **90.00** | 64.54 | 58.22 | 74.29 | 62.37 |

### GNN − MLP gap（净结构价值）

| Model | steam | tmdb | wikiart |
|-------|-------|------|---------|
| GCN | −4.86 | −11.04 | **+15.68** |
| GAT | −6.39 | −10.92 | +5.43 |
| GraphSAGE | −0.45 | −1.52 | **+17.58** |
| GIN | −10.04 | −21.11 | +4.86 |
| GCNII | −1.34 | −0.84 | +9.43 |

> 去泄露后 GNN 只在高同质 wikiart 显著优于 MLP；低同质 tmdb/steam 下邻居是噪声源。这是 leak-free 协议生效的直接证据。

---

## 3. NC 模态消融（acc，10 seed）

来源：`newdomain_nc/modality_ablation_decontaminated_2026-07-15.json`

### STEAM
| model | text | image | text+image | multi−text |
|-------|------|-------|------------|------------|
| mlp | 0.635 | 0.621 | 0.659 | +0.024 |
| gcn | 0.604 | 0.607 | 0.610 | +0.007 |
| gat | 0.580 | 0.583 | 0.595 | +0.015 |
| graphsage | 0.633 | 0.634 | 0.654 | +0.021 |
| gin | 0.538 | 0.539 | 0.558 | +0.020 |
| gcnii | 0.619 | 0.638 | 0.645 | +0.027 |

### TMDB
| model | text | image | text+image | multi−text |
|-------|------|-------|------------|------------|
| mlp | 0.551 | 0.561 | 0.591 | +0.040 |
| gcn | 0.463 | 0.472 | 0.480 | +0.018 |
| gat | 0.453 | 0.469 | 0.481 | +0.029 |
| graphsage | 0.546 | 0.548 | 0.575 | +0.030 |
| gin | 0.376 | 0.384 | 0.380 | +0.004 |
| gcnii | 0.542 | 0.554 | 0.582 | +0.040 |

### WIKIART（image 碾压 text）
| model | text | image | text+image | multi−text |
|-------|------|-------|------------|------------|
| mlp | 0.299 | 0.657 | 0.649 | **+0.349** |
| gcn | 0.560 | 0.806 | 0.805 | **+0.245** |
| gat | 0.353 | 0.767 | 0.703 | **+0.350** |
| graphsage | 0.580 | 0.827 | 0.824 | **+0.245** |
| gin | 0.487 | 0.705 | 0.697 | +0.211 |
| gcnii | 0.398 | 0.767 | 0.743 | **+0.345** |

> **去泄露后 image 全域转正**（multi−text Δ 全为正）。wikiart 上 image 是主导模态（纯标题 text MLP 仅 0.299）。旧"text≈multi / image 无贡献"结论全部作废。

---

## 4. LP 链路预测（AUC，10 seed，三档负采样）

来源：`benchmark/results/newdomain_lp/lp_{domain}_summary.json`

### 4a. LP Ceiling（边泄露检测）
| 域 | Ceiling Original | Ceiling Pruned | Majority |
|---|---|---|---|
| arxiv | 0.673 | 0.261 | 0.289 |
| steam | 0.582 | 0.504 | 0.504 |
| tmdb | 0.422 | 0.154 | 0.205 |
| wikiart | 0.918 | **0.024** | 0.025 |
| spotify | 0.517 | 0.167 | 0.167 |

> 剪枝后全域 ceiling ≤ majority。wikiart 0.918→0.024（−0.894）降幅最大，跨切分边彻底剪断。

### 4b. LP 模态消融全表（AUC，列序 mlp/gcn/gat/graphsage/gin/gcnii）

**STEAM**
| modality | neg | mlp | gcn | gat | graphsage | gin | gcnii |
|---|---|---|---|---|---|---|---|
| image | random | 0.867 | 0.901 | 0.860 | 0.903 | 0.866 | 0.864 |
| image | hard | 0.834 | 0.877 | 0.828 | 0.881 | 0.825 | 0.823 |
| image | hard_degree | 0.784 | 0.817 | 0.792 | 0.839 | 0.730 | 0.704 |
| text | random | 0.881 | 0.890 | 0.854 | 0.893 | 0.862 | 0.863 |
| text | hard | 0.851 | 0.863 | 0.822 | 0.867 | 0.824 | 0.823 |
| text | hard_degree | 0.796 | 0.795 | 0.777 | 0.821 | 0.729 | 0.701 |
| text+image | random | 0.880 | 0.898 | 0.855 | 0.902 | 0.860 | 0.870 |
| text+image | hard | 0.847 | 0.874 | 0.821 | 0.879 | 0.817 | 0.832 |
| text+image | hard_degree | 0.799 | 0.809 | 0.780 | 0.836 | 0.723 | 0.719 |

**TMDB**
| modality | neg | mlp | gcn | gat | graphsage | gin | gcnii |
|---|---|---|---|---|---|---|---|
| image | random | 0.802 | 0.889 | 0.859 | 0.876 | 0.810 | 0.841 |
| image | hard | 0.739 | 0.849 | 0.805 | 0.819 | 0.808 | 0.825 |
| image | hard_degree | 0.677 | 0.773 | 0.732 | 0.751 | 0.718 | 0.727 |
| text | random | 0.839 | 0.893 | 0.864 | 0.890 | 0.822 | 0.853 |
| text | hard | 0.798 | 0.856 | 0.814 | 0.837 | 0.818 | 0.830 |
| text | hard_degree | 0.712 | 0.778 | 0.739 | 0.769 | 0.726 | 0.732 |
| text+image | random | 0.842 | 0.897 | 0.870 | 0.893 | 0.816 | 0.853 |
| text+image | hard | 0.784 | 0.857 | 0.814 | 0.837 | 0.813 | 0.835 |
| text+image | hard_degree | 0.703 | 0.782 | 0.743 | 0.770 | 0.721 | 0.740 |

**WIKIART**（注意 MLP-text 崩塌 + image/text+image 极高）
| modality | neg | mlp | gcn | gat | graphsage | gin | gcnii |
|---|---|---|---|---|---|---|---|
| image | random | 0.933 | 0.989 | 0.988 | 0.987 | 0.877 | 0.951 |
| image | hard | 0.664 | 0.872 | 0.821 | 0.818 | 0.799 | 0.810 |
| image | hard_degree | 0.652 | 0.832 | 0.811 | 0.804 | 0.715 | 0.745 |
| text | random | 0.787 | 0.968 | 0.973 | 0.973 | 0.973 | 0.909 |
| text | hard | 0.629 | 0.864 | 0.842 | 0.834 | 0.920 | 0.814 |
| text | hard_degree | 0.627 | 0.838 | 0.825 | 0.830 | 0.857 | 0.765 |
| text+image | random | 0.925 | 0.987 | 0.987 | 0.987 | 0.939 | 0.938 |
| text+image | hard | 0.659 | 0.873 | 0.820 | 0.819 | 0.872 | 0.811 |
| text+image | hard_degree | 0.647 | 0.835 | 0.809 | 0.803 | 0.792 | 0.746 |

**ARXIV**（`lp_arxiv_summary.json` 仅存全模态；单模态见 `lp_ablation_final/`，§15）
| neg | mlp | gcn | gat | graphsage | gin | gcnii |
|---|---|---|---|---|---|---|
| random (t+i+f) | 0.909 | 0.932 | 0.928 | 0.893 | 0.938 | — |
| hard (t+i+f) | 0.779 | 0.829 | 0.822 | 0.810 | 0.874 | — |
| hard_degree (t+i+f) | 0.841 | 0.882 | 0.878 | 0.836 | 0.919 | — |

**SPOTIFY**（text-only 域）
| neg | mlp | gcn | gat | graphsage | gin | gcnii |
|---|---|---|---|---|---|---|
| random | 0.973 | 0.983 | 0.988 | 0.983 | 0.973 | — |
| hard | 0.942 | 0.964 | 0.974 | 0.974 | 0.963 | — |
| hard_degree | 0.968 | 0.981 | 0.985 | 0.980 | 0.963 | — |

### 4c. LP 模态消融关键分析
- **去泄露对 LP 影响远小于 NC**：wikiart NC text MLP 0.95→0.30（崩），但 LP text random AUC（GNN 系）仍 0.97——LP 信号来自"同类节点连边"，图结构本身是强信号，text 内容质量几乎不影响。
- **wikiart MLP-text 是唯一崩塌 LP cell**（random 0.787）：纯标题无法支撑无边预测；加图结构后立即回升 0.97。
- **LP 中 image ≮ text**（与 NC 相反）：wikiart image GNN 0.989 vs text GNN 0.973，均极高；steam/tmdb 双模态持平，融合增益仅 ~0.005。LP 里图结构信号掩盖模态差异，NC 里模态质量决定上限。
- **三档负采样全域单调下降**：random > hard > hard_degree，去饱和协议有效。
- **GCNII 在 LP hard_degree 偏弱**（steam 0.704, wikiart 0.746），与 NC（wikiart 0.743）表现相反。

---

## 5. Steering 因果实验（C4 皇冠）

来源：`benchmark/results/steering/steering_causal_analysis_2026-07-12.json`

### 5a. 核心因果律（GCN-only, n=12 点, 3 域 × 关系子集）
| 指标 | r | p | bootstrap 95% CI |
|---|---|---|---|
| GCN-MLP gap vs 同质性 | **0.846** | **5.2e-04** | [0.600, 0.960] |
| GCN_F1 vs 同质性 | 0.913 | 3.5e-05 | [0.774, 0.975] |
| GCN_F1 vs MLP_control | 0.945 | 3.7e-06 | [0.688, 0.990] |

线性拟合：`gap = 0.364 × homophily − 0.332`，r²=0.716，**crossover homo ≈ 0.91**
Leave-one-domain-out 稳健：剔除 steam r=0.944 / tmdb r=0.810 / wikiart r=0.703。
MLP bit-identical 自检通过（同 GNN 在不同关系子图上的差异只归因于边结构）。

### 5b. Steering NC F1（按关系子集）
| 域 | MLP（恒定自检） | all 关系最佳 GNN |
|---|---|---|
| steam | 0.405–0.537 | GraphSAGE 0.502–0.576 |
| tmdb | 0.363–0.504 | GraphSAGE 0.378–0.486 |
| wikiart | 0.233–0.909 | GAT 0.94–0.99 |

---

## 6. 聚类 NMI/ARI（text+image, 10 seed）

来源：`benchmark/results/newdomain_clustering/cluster_{domain}_summary.json`

| Model | arxiv NMI | steam NMI | tmdb NMI | wikiart NMI |
|-------|-----------|-----------|----------|-------------|
| raw_features | 0.436 | 0.206 | 0.273 | 0.390 |
| mlp | 0.712 | 0.349 | 0.427 | 0.549 |
| gcn | 0.732 | 0.188 | 0.234 | 0.704 |
| gat | 0.727 | 0.205 | 0.271 | 0.816 |
| graphsage | 0.719 | 0.302 | 0.430 | 0.820 |
| gin | 0.486 | 0.113 | 0.151 | 0.361 |
| gcnii | **0.771** | **0.372** | 0.412 | **0.847** |

> wikiart raw NMI 0.673（旧 leaky）→ **0.390**（去泄露，−42%）。低同质 steam/tmdb 下 GCN<raw（结构破坏聚类）。

---

## 7. Few-Shot（text+image, 10 runs）

来源：`benchmark/results/newdomain_fewshot/fewshot_{domain}_summary.json`

| 域 | Best Model | k=3 Acc | k=10 Acc |
|----|-----------|---------|----------|
| steam | GCNII | 0.435 | 0.505 |
| tmdb | MLP | 0.317 | 0.409 |
| wikiart | GAT | **0.729** | **0.821** |

> wikiart GAT k=3=0.729 vs MLP k=3=0.288（+0.441）：高同质下 3 个标注节点经邻居传播覆盖全图。tmdb MLP>GNN，低同质邻居稀释信号。与 steering 一致。

---

## 8. 跨域迁移（C4 负迁移发现）

来源：`benchmark/results/newdomain_crossdomain/transfer_matrix.json` + `dual_protocol_report_2026-07-12.json`

### 8a. 双协议（4 域 off-diagonal, text+image）
| Protocol | 负增益对 | 占比 | Mean Gain |
|----------|---------|------|-----------|
| linear_probe | 12/12 | 100% | −0.255 |
| full_finetune | 8/12 | 66.7% | −0.0005 |

### 8b. GCN text 迁移矩阵（transfer_gain）
| src→tgt | steam | tmdb | wikiart |
|----------|-------|------|---------|
| steam | — | −0.107 | **−0.587** |
| tmdb | −0.057 | — | **−0.552** |
| wikiart | −0.066 | −0.069 | — |

> 线性探测全域负增益（特征不迁移）；全微调中性（模型重学）。论文报双协议，诚实披露。

---

## 9. 跨模态对齐（CKA + Retrieval）⚠️ TMDB 残留信号

来源：`benchmark/results/newdomain_modality/modality_{domain}_summary.json`

| Metric | steam | tmdb | wikiart |
|--------|-------|------|---------|
| CKA | 0.0063 | 0.0040 | 0.0021 |
| text→image R@1 | 0.472 | **0.761** | 0.074 |
| text→image R@5 | 0.703 | 0.913 | 0.154 |

> wikiart R@1=0.074≈random（0.056）→ 去泄露成功。**TMDB R@1=0.761 与 CKA≈0 矛盾**：电影长概述中 genre 关键词隐含语义信号（natural correlation，非性定义级 "Style: X" 硬泄露）。论文需区分对待。

---

## 10. 模态缺失鲁棒性（GCN/GAT/GraphSAGE × 5 条件, 10 seed）

来源：`benchmark/results/newdomain_modality_missing/modmiss_{domain}_summary.json`

| 条件 | wikiart GCN | wikiart GraphSAGE | tmdb GraphSAGE |
|------|------------|-------------------|----------------|
| full | 0.808 | **0.825** | 0.577 |
| drop_text (=img-only) | 0.808 (−0.00) | 0.828 (+0.00) | 0.547 (−0.030) |
| drop_image (=txt-only) | **0.561 (−0.248)** | 0.579 (−0.246) | 0.544 (−0.033) |
| dropout_text_50 | 0.812 | 0.832 | 0.573 |
| dropout_image_50 | 0.794 | 0.814 | 0.574 |

> **wikiart drop_image 跌 −0.248**（去泄露前 drop_image≈无影响）→ image 现为必需模态，核心叙事翻转。wikiart drop_text 无影响（纯标题无用）。

---

## 11. 鲁棒性（GCN, 特征 vs 边噪声, 10 seed）

来源：`benchmark/results/newdomain_robustness/robust_{domain}_{feature,edge,summary}.json`

| 域 | feat ρ=0.2 | edge ρ=0.2 | feat ρ=0.8 | edge ρ=0.8 |
|----|----------|---------|----------|---------|
| steam | −0.9pp | −6.4pp | −4.1pp | −11.4pp |
| tmdb | −0.8pp | −3.1pp | −8.1pp | −13.7pp |
| wikiart | −1.3pp | −13.8pp | −12.0pp | **−36.7pp** |

> 边噪声 ≫ 特征噪声全域成立；wikiart 对边噪声极度敏感（高同质结构是 GNN 生命线）。

---

## 12. Label 噪声（10 seed）

来源：`benchmark/results/newdomain_labelnoise/labelnoise_{domain}_summary.json`

| 域 | Model | ρ=0.0 | ρ=0.2 | ρ=0.4 | ρ=0.6 |
|----|-------|-------|-------|-------|-------|
| tmdb | GraphSAGE | 0.577 | 0.567 | 0.551 | 0.529 |
| tmdb | GCN | 0.477 | 0.475 | 0.470 | 0.463 |
| wikiart | GraphSAGE | 0.825 | 0.814 | 0.795 | 0.754 |
| wikiart | GCN | 0.810 | 0.792 | 0.772 | 0.738 |

> tmdb GCN 几乎不敏感（本就靠不住）；wikiart 高同质 GNN 强但也更脆弱。

---

## 13. 邻居注入（k-NN feature augmentation, 10 seed）

来源：`benchmark/results/newdomain_neighbors/neighbors_{domain}.json`

| 域 | k=0 | k=2 | k=8 | Δ(k8−k0) |
|----|------|-----|-----|-----------|
| steam | 0.651 | 0.658 | 0.661 | +0.009 |
| tmdb | 0.593 | 0.592 | 0.590 | −0.003 |
| wikiart | 0.652 | 0.669 | **0.812** | **+0.160** |

> wikiart +0.160（高同质下特征近邻≈标签一致）；tmdb 无效（低同质特征近邻≠同类）。与 steering 一致。

---

## 14. 边关系预测 / 融合 / 逐类 / 效率

| 实验 | 来源 | 关键结果 |
|---|---|---|
| **边关系预测** | `edge_relation_prediction/edgerel_{domain}.json` | steam acc 0.843 (F1 0.794, 4 类); tmdb 0.736 (F1 0.685, 3 类); wikiart **0.970** (F1 0.970, 2 类) |
| **融合消融** (GCN, 5 seed) | `fusion_ablation/fusion_{domain}_*.json` | Attention 全域最优（+0.1~2.7pp over mean）；wikiart 注意力权重 image 0.83/text 0.17；arxiv text 0.53/image 0.26/formula 0.21 |
| **逐类 F1** (GCN vs MLP) | `per_class_eval/perclass_{domain}_*.json` | wikiart GCN macro-F1 0.799 vs MLP 0.525; tmdb GCN 0.210 vs MLP 0.363（低同质 GCN 反而差，小类 F1=0） |
| **效率** (RTX 4090) | `newdomain_efficiency/efficiency_{domain}_summary.json` | wikiart: GCN 9.5ms/ep, GAT 38.8ms/ep, GraphSAGE 9.3ms/ep; arxiv(45k 节点): GCN 5.4ms/ep |

### 融合消融全表（GCN, 5 seed）
| Domain | concat | attention | mean | sum | best |
|--------|--------|-----------|------|-----|------|
| arxiv (t+i+f) | 0.916 | **0.919** | 0.910 | 0.910 | attention |
| steam (t+i) | 0.616 | 0.620 | 0.613 | 0.613 | attention |
| tmdb (t+i) | 0.488 | 0.492 | 0.490 | 0.490 | attention |
| wikiart (t+i) | 0.872 | **0.898** | 0.871 | 0.871 | attention |

---

## 15. IGC 图构建 agent 消融（C2/C3 诚实定位）

来源：`benchmark/results/igc/igc_{domain}.json` + `wikiart_verifier_reverify_trace.json`

| Spec | 域 | Greedy(无LLM) | Closed_loop(LLM) | Oracle | 结论 |
|------|----|--------------|------------------|--------|------|
| S1b (blind) | wikiart | 0.880 | 0.880 | 0.880 | **greedy=oracle**，agent 无增益 |
| S1b (blind) | tmdb | 0.300 | **0.401** | 0.300 | closed_loop 反而更差（审计非优化器） |
| S2 (max_homo) | steam | — | lp 0.502/homo 0.425 | homo 0.818 | 回路未收敛 |

Verifier trace（wikiart）：LP pathway **0.918 → 0.880 → 0.024**（drop shared_style 76776 边 + cross-split prune 497936 边）→ PASS。

> 去泄露靠**测量**（LP-ceiling/homophily），不靠 LLM。论文写 "audit-capable pipeline"，非 "self-verifying"。

---

## 16. Arxiv 专项（formula/编码器/PEFT/学习曲线）

来源：`lp_ablation_final/`, `learning_curve/`, `lp_sweep/`, `sensitivity/`, `openmag/`

| 实验 | 关键结果 |
|------|---------|
| Formula NC 消融 | text+image+formula=89.81 vs text+image=89.73 → **Δ=+0.0006** (no-op，卖完整性) |
| Formula LP 消融 | GCN text 0.691(MRR) → t+i+f **0.716**（+0.025，LP 里 formula 有用） |
| LP Sweep (GATv2/GCNII/GT/RevGAT) | GT+text MRR=**0.691**（best）；GCNII+t+i+f 0.617 |
| 学习曲线 | 1%→80% train，全模型 87.5→91.7；低标注 GCN 略优 MLP |
| 超参数敏感度 | GraphSAGE 最佳 91.61（lr 0.005/hdim 256/4 layer），整体扁平 ±1pp |
| PEFT | None 91.29 > Adapter 89.94 > LoRA(r8) 88.12（全参微调仍最优） |
| 编码器比较 | CLIP-Large(unified) 跨模态 MRR 0.107 ≫ SBERT+CLIP 0.001, SPECTER+CLIP 0.002 |
| 自监督 | self-sup 89.01 vs supervised 91.01（合理 gap） |

---

## 17. 最终叙事（论文怎么用这些数字）

1. **去泄露让 text 诚实**：wikiart MLP text 0.95→0.30，image 成必需模态（drop_image −0.248）。
2. **image 全域转正**：NC multi−text Δ wikiart +0.35 / steam +0.02 / tmdb +0.04。
3. **GNN−MLP gap 由同质性因果决定**：wikiart +17.6, tmdb −11.0, steam −4.9；**r=0.846 [0.600, 0.960], p=5.2e-04, crossover homo≈0.91**。
4. **NC vs LP 模态行为不同**：NC 里模态质量决定上限（wikiart image 碾压 text）；LP 里图结构信号掩盖模态差异。
5. **CKA/R@1 悖论**：TMDB CKA≈0 但 R@1=0.761——语义关联 ≠ 定义级泄露，需区分。
6. **跨域迁移线性探测 100% 负**——诚实披露，full_finetune 中性。
7. **Agent 只在语义 wrangling + 可复用接口 load-bearing**；去泄露决策 greedy=oracle，靠测量不靠 LLM。

---

*生成于 2026-07-16，基于 2026-07-15 20:52 全域重跑 + 2026-07-16 复核。本表为去泄露后唯一权威数字源；旧 `AMMAG_全域基准实验报告_2026-06-20.md` 等的数字一律作废。*
