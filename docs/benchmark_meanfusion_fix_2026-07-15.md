# Benchmark 修复：多模态消融"变差"的真因与修复（2026-07-15）

> 🔴 **2026-07-15 文本去泄露修正（本文数字已过时）**：
> 同日稍后发现 wikiart/steam/tmdb 的 **text 特征含 label 明文字段**（"Style: Neo Impressionism" /
> "Genres: Racing"），导致 text-only 精度虚高、多模态消融的 "text ≈ multi" 结论被污染。
> **本文所有含 text 的 NC/LP 数字已作废，真数见
> [`docs/benchmark_text_decontamination_2026-07-15.md`](benchmark_text_decontamination_2026-07-15.md)。**
> ↓ 只有 arxiv（text 无泄露）、image-only 数字、steering(r=0.846)、GIN 崩溃诊断 仍有效。↓

> 本文取代 `docs/projection_768d_fairness_2026-07-12.md` 的结论（那次"768-d 投影公平性修复"基于对 OpenMAG 的误读，引入了非对称瓶颈——**不是**多模态变差的真因）。本文是 NC 类实验的**旧权威**——已被文本去泄露修正取代。

---

## 0. 一句话结论

"多模态消融后效果反而变差" **不是数据 bug、不是 768-d 投影瓶颈、不是融合方式问题**。真因是 **NC 训练循环没有梯度裁剪**，导致 **GIN（默认超参敏感）在高维多模态输入上频繁崩溃到接近多数类**，把多模态均值拉低、方差炸高，制造"变差"假象。修复（梯度裁剪 + OpenMAG 式 mean 融合 + GIN 单独 lr=0.005 + 10 seed）后：**多模态 ≈ text（Δ 在 −0.001~−0.044，无崩溃），GIN 稳定，显著性突破 n=5 floor（p=0.002）**。

---

## 1. 诊断（受控实验，非猜测）

### 1a. 证伪"投影瓶颈"假说
受控实验 `benchmark/diag_modality_fusion.py`（同一切分/seed/训练循环，只换融合）：wikiart GCNII `(多模态−text)` = mean **−0.014** / sum −0.020 / concat −0.013 / **bottleneck +0.002**。投影瓶颈反而是对多模态**最友好**的；切到 mean 只会更差。→ 投影**不是**真因。

### 1b. 真因 = GIN 训练崩溃（无梯度裁剪）
per-seed 扫描：崩溃几乎全在 GIN（wikiart `gin+text+image` **0.035**、tmdb 0.206、arxiv std 0.127）+ 偶发 GCNII 坏种子。稳定模型（GCN/GAT/GraphSAGE/MLP）多模态 ≈ text（±0.002–0.008）。NC 循环 `train_and_eval` 此前**没有**梯度裁剪（LP 循环有）。

### 1c. 修复已验证
`benchmark/diag_stabilize.py`：加 `clip_grad_norm_(1.0)` → wikiart `gin+text+image` **0.47±0.118（崩溃）→ 0.78±0.014（稳定）**；GIN/text 0.70±0.061 → 0.80±0.002。

### 1d. image 特征没坏
tmdb text→image R@1=0.783；image-only NC 0.55–0.83（合理）。残差"多模态微低于 text"是**真实的 text-dominance**（文本丰富的文化元数据域），与论文 P1-14 框架一致。

---

## 2. OpenMAG 协议澄清（关键）

读 `docs/OpenMAG(1).pdf` 原文（非旧报告）：
- 编码器 = **CLIP-ViT-L/14（frozen）**，text 与 image 都进**同一个 768-d 共享空间**；"unifying the node embedding dimensionality at **768** across all tasks"。
- 基线融合 Φ = **逐元素（mean/sum）→ 768-d**，**不是**拼接，**更不是** learnable Linear(1536,768) 投影。
- 我们先前的 `ProjectedModel`（[gnn.py:328](../benchmark/models/gnn.py#L328) `Linear(in_dim,768)` 非对称瓶颈）是对 OpenMAG 的**误读**，已弃。

---

## 3. 代码改动（2026-07-15）

| 文件 | 改动 |
|---|---|
| `benchmark/data/domain_loader.py::build_features` | 拼接 → **逐模态 L2 归一化后逐元素 mean，再归一化**（恒 768-d，OpenMAG 对齐） |
| `benchmark/tasks/node_classification.py::train_and_eval` | 加 `clip_grad_norm_(model, 1.0)` |
| `benchmark/models/gnn.py` | 加 `MODEL_LR={"gin":0.005}` + `model_lr()`（OpenMAG 逐模型调参；GIN 降 lr 稳定） |
| `run_newdomain_{nc,clustering,fewshot,modality_missing,labelnoise,robustness,neighbors,lp}.py` | 全部：训练循环加梯度裁剪 + `model_lr(model)` + seed 默认 10 |
| `run_newdomain_modality_missing.py` | 重写为 mean 融合语义（drop_text≡image-only, drop_image≡text-only；dropout_50 为真正部分破坏探针） |

诊断脚本：`benchmark/diag_modality_fusion.py`、`benchmark/diag_stabilize.py`；聚合 `scripts/reaggregate_nc_meanfix.py`。

---

## 4. 新头条结果（10 seed，mean 融合，leak-free）

### 4a. NC 模态消融（test acc；多模态 = mean(text,image)）
稳定模型多模态 ≈ text，无崩溃：

| domain | model | text | image | text+image | (multi−text) |
|---|---|---|---|---|---|
| wikiart | gcnii | 0.915 | 0.764 | 0.904 | −0.011 |
| wikiart | gin | 0.806 | 0.705 | **0.799** | −0.007（旧 0.035 崩溃） |
| wikiart | graphsage | 0.933 | 0.827 | 0.918 | −0.015 |
| arxiv | gcnii | 0.906 | 0.819 | 0.900 | −0.006（formula 独有：t+i+f 0.899 = no-op） |
| steam | gcnii | 0.669 | 0.640 | 0.685 | **+0.016** |
| tmdb | graphsage | 0.678 | 0.548 | 0.635 | −0.043 |

**结论**：image/formula 是"模态丰富度"非性能驱动；text 主导。诚实、可辩护。

### 4b. GNN-vs-MLP 配对 Wilcoxon（n=10，**突破旧 n=5 floor p=0.0625**）
| domain | 方向（best 故事） |
|---|---|
| **arxiv**（homo 0.93） | 全部 GNN **显著 > MLP**（p=0.002，Δ+0.003~+0.007） |
| **tmdb**（homo 0.24） | 全部 GNN **显著 < MLP**（p=0.002）—— 异质端，结构有害 |
| steam | gcnii 显著 > MLP（p=0.027）；gcn/gat/gin < MLP |
| wikiart | graphsage/gcnii ≈ MLP；gcn/gat/gin < MLP |
| spotify | graphsage ≈ MLP；余略 < MLP |

→ 同质性×结构价值的因果故事（steering r=0.846）现在有**真 p 值**支撑。wikiart 旧"MLP 0.950 ≫ GNN 0.929"已收敛到 MLP 0.919 ≈ graphsage 0.918（崩溃消除）。

### 4c. LP 去饱和（mean 融合，GIN 10/10 healthy）
| domain | gcn random→hard→hard_degree |
|---|---|
| steam | 0.896→0.871→0.806 |
| tmdb | 0.893→0.858→0.783 |
| wikiart | 0.994→0.905→0.876 |

去饱和单调，GIN 稳定（旧崩溃消除）。

### 4d. 模态缺失（mean 融合，旧"drop_image 反升"伪影已除）
wikiart graphsage：full 0.919 / drop_text 0.828（掉）/ drop_image 0.936（≈full）。
→ 干净的 text-dominance：缺 text 大掉，缺 image 几乎不变。

### 4e. 聚类 NMI（leak-free）
wikiart 0.805 / arxiv 0.761（高同质→聚类好）；steam 0.187 / tmdb 0.234（低同质→差）。与 steering 一致。

---

## 5. 对论文写作的影响（更新 FINAL §4.5 攻击面）

| 旧表述 | 新表述 |
|---|---|
| "多模态消融 +0.0007 / image 边际" | 仍成立，但**机制**是 text-dominance（非投影伪影）；GIN 已稳定不再拖均值 |
| 投影 768-d "公平修复" | **删除**该说法；改为 OpenMAG 式 mean 融合（768-d 共享空间） |
| n=5 Wilcoxon floor p=0.0625 | **改为** n=10、p=0.002，报真实显著性 + 同质性方向 |
| GIN 3 域坍缩 | GIN lr=0.005 + 裁剪后稳定（wikiart 0.80、tmdb 0.39 弱但非崩溃） |

**仍待办（非本轮）**：R5 advanced_models（DGF/MMGCN/MGAT/LGMRec）非 OpenMAG 忠实实现（无逐模态融合），claim 需缩为"标准 GNN 在本数据集"或补忠实实现；R6 多 split_seed（仅 tmdb 有）可补 2 个域。

---

## 5b. 次要实验（同协议重跑，mean 融合 + 裁剪 + 10 seed；权威数字见各自 JSON）

- **效率**（`newdomain_efficiency/efficiency_{domain}_summary.json`）：time/显存随节点数 scale；GCN 全域 ~200k 参数（mean 融合→768-d 输入统一，参数量不再随模态数变化）。例 steam 2.7ms/134MB → wikiart 9.5ms/1083MB → arxiv 5.4ms/433MB。
- **编码器族**（`encoder_comparison_arxiv.json`，arxiv 旗舰）：**clip_large 0.905 > clip_base 0.897 > siglip 0.849**（gcnii，10 seed）。证 per-domain encoder profiling 必要——SigLIP 64-token 截断长论文输给 CLIP。P1-4 发现复现。
- **高级 MAG 模型**（`nc_{wikiart,steam}_summary_advanced.json`）：wikiart MMGCN 0.903 / LGMRec 0.902 / DGF 0.894 ≈ 头条 gcnii 0.904；steam MMGCN 0.611 领先。**⚠️ R5 仍存**：这些是 `Linear+GNN` 简化实现，**非** OpenMAG 忠实逐模态融合（MMGCN 分模态图 / MGAT 模态注意力未实现）——claim 须缩为"标准 GNN 在本数据集"，或补忠实实现。
- **聚类**（`newdomain_clustering/`）：NMI wikiart 0.805 / arxiv 0.761（高同质→好）；steam 0.187 / tmdb 0.234（低同质→差）。
- **few-shot**（`newdomain_fewshot/`）：高同质 GNN 胜（wikiart gcn k10 0.799 > mlp 0.711）；低同质 MLP 胜。与 steering 一致。
- **模态缺失**（`newdomain_modality_missing/`）：drop_text 大掉、drop_image≈full（干净 text-dominance，旧伪影消除）。
- **鲁棒 / 标签噪声 / 邻居 sweep**（`newdomain_{robustness,labelnoise,neighbors}/`）：同协议重跑（10 seed）。邻居 sweep 证结构价值随同质性（高同质 K↑ 有用、tmdb 无效）。

## 6. 产物
- `benchmark/results/newdomain_nc/nc_{domain}_summary.json`（10 seed，mean 融合）
- `benchmark/results/newdomain_nc/main_table_significance_2026-07-15.json`
- `benchmark/results/newdomain_nc/modality_ablation_2026-07-15.json`
- `benchmark/results/newdomain_nc/_archive_premeanfix_2026-07-15/`（旧投影版备份）
- LP / modmiss / cluster / fewshot / robust / labelnoise / neighbors：同协议重跑（mean 融合 + 裁剪 + 10 seed）

*生成于 2026-07-15。诊断详见记忆 `benchmark-multimodal-worse-root-cause`。*
