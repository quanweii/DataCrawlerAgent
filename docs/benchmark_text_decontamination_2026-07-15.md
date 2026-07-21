# Text 特征去泄露修复（2026-07-15）— **全域实验新权威**

> ✅ **2026-07-15 20:52 全域实验全部重跑完成。** 所有 steam/tmdb/wikiart 的 text 特征已从
> 去泄露 `text_embeddings.pt` 重新编码。NC/LP/聚类/few-shot/crossdomain/modality/
> modmiss/robustness/labelnoise/neighbors/edge_relation/fusion/per_class/steering
> **全部使用去泄露文本特征**。旧 leaky 文件已归档至 `benchmark/processed/_archive_leaky/`。
>
> 📊 **完整结果数字总表见 [`experiment_results_FINAL_decontaminated_2026-07-16.md`](experiment_results_FINAL_decontaminated_2026-07-16.md)**
> （去泄露后唯一权威数字源，含 NC/LP 全模态消融、steering、聚类、few-shot 等全部表 + 统一协议参数）。
> 本文聚焦**泄露诊断与修复过程**；**查具体数字请用那份总表**。旧 `AMMAG_全域基准实验报告_2026-06-20.md`
> 等去泄露前的结果表一律作废。

---

## 0. 一句话

**wikiart/steam/tmdb 的 text 特征（.md markdown）在 CLIP 编码前明文包含 "Style: Neo Impressionism" / "Genres: Racing" 等 label 承载元数据字段。这导致 text 特征在 0 参数下即可达到 100% 的 label 恢复率——"text 效果最好"是泄露，不是发现。**

---

## 1. 泄露发现（三环闭合）

### 1a. label 字符串在 raw text 中 100% 命中（已核实）
- **wikiart**: markdown 含 `Style: Neo Impressionism` → label = "Neo Impressionism"。4000/4000 = 100%。
- **steam**: markdown 含 `Genres: Racing, Simulation, Sports` → label = "Racing"。2760/2760 = 100%。
- **tmdb**: markdown 含 `Genres: Drama` → label = "Drama"。4000/4000 = 100%。
- **arxiv**: 论文摘要不含 category label → **干净**。

### 1b. 编码链路未过滤（已核实）
`text_encoder.py:_read_md` → `f.read()` 读取完整 .md → 经 CLIP chunk/pool → 存入 `text_embeddings.pt`。label 字段通过 CLIP 编码进入所有实验。

### 1c. 0-参数恢复标签（铁证）
只用正则 `Style:|Genres:` 解析 raw text + 字符串查重 → **全域 100% 准确率**（无 CLIP、无 GNN、无训练）。

诊断脚本：`benchmark/diag_text_label_leakage.py`、`benchmark/diag_label_field_recover.py`。

---

## 2. 修复

### 2a. 代码改动
| 文件 | 改动 |
|---|---|
| `benchmark/encoders/text_encoder.py` | `__init__` 新增 `strip_patterns: Optional[List[str]]` 参数；`_read_md` 用 `re.MULTILINE|IGNORECASE` 删除匹配行 |
| `benchmark/encode_domain.py` | 新增 `TEXT_STRIP_PATTERNS` dict：wikiart `^Style\s*:.*$`；steam/tmdb `^Genres?\s*:.*$`；传给 `TextEncoder(strip_patterns=...)` |

### 2b. 重新编码 + NC 重跑
```bash
# 三域重新编码（strip label-bearing 行后）
python benchmark/encode_domain.py --domain wikiart --gpu 1 --modalities text  # R2: 纯标题(0% label-in-text)
python benchmark/encode_domain.py --domain tmdb --gpu 2 --modalities text      # Genres行已删
python benchmark/encode_domain.py --domain steam --gpu 5 --modalities text     # Genres行已删
```
NC 全模态重跑完成（text/image/text+image, 10 seed, leak-free 协议）。
详见 `benchmark/encode_domain.py:55` `TEXT_STRIP_PATTERNS`。

---

## 3. 影响范围

### 被污染的结论（旧数字）vs 去泄露后真值
| 域 | 旧 MLP text | → 新 MLP text | 跌 | 新 multi−text Δ | 故事 |
|---|---|---|---|---|---|
| wikiart | 0.948 | **0.299** | −0.649 | **+0.349** | 纯标题(avg 26 chars)，image 碾压 text |
| tmdb | 0.695 | **0.551** | −0.144 | **+0.040** | 电影概述褪去 "Genres: Drama"，image 正面贡献 |
| steam | 0.687 | **0.635** | −0.052 | **+0.024** | 游戏描述褪去 "Genres: Racing"，image 正面贡献 |

> wikiart 经历两轮修复：R1 只删 "Style:" 行→text 仍 0.90（正文模板句 "in X style" + 艺术家名泄露）；R2 删到只剩标题→text 跌到 0.30（诚实）。详见 §5b。

### 不受影响的证据
- **steering 因果 (r=0.846)**：用 text 特征但只看 GNN−MLP gap（MLP bit-identical 自检，边因果成立）。
- **arxiv 全表**：text 无 label 泄露，所有 arxiv 数字干净。
- **image-only 全表**：image 特征来自 JPG，不含 label 字段。
- **LP 去饱和趋势**：去饱和方向（random→hard→hard_degree 单调下降）不受 text 水位影响。
- **同质性谱**：4 域同质性是边结构属性，与特征无关。

---

## 4. 需重跑的实验（仅 wikiart/steam/tmdb，arxiv+spotify 干净跳过）

| 实验 | 域 | 状态 |
|---|---|---|
| NC 模态消融 (text/image/multi) | wikiart/steam/tmdb | 待跑 |
| NC 主表 + 显著性 | wikiart/steam/tmdb | 待跑 |
| LP (含 per-modality 消融) | wikiart/steam/tmdb | 待跑 |
| 聚类 NMI | wikiart/steam/tmdb | 待跑 |
| Few-shot | wikiart/steam/tmdb | 待跑 |
| 模态缺失 | wikiart/steam/tmdb | 待跑 |
| 鲁棒性 | wikiart/steam/tmdb | 待跑 |
| Label 噪声 | wikiart/tmdb | 待跑 |
| 邻居 sweep | wikiart/steam/tmdb | 待跑 |
| 效率 | wikiart/steam/tmdb | 待跑 |

arxiv 和 spotify 数字干净，无需重跑。

---

## 5b. wikiart 两轮修复实录（证实"泄露不止一层"）

| 阶段 | 清洗内容 | label-in-text | mgraphsage text acc |
|---|---|---|---|
| OLD (pre-decontam) | 无 (全文含 "Style: X" + 模板句 + 艺术家) | 100% | 0.933 |
| R1 (strip Style 行) | 删 "Style: Neo Impressionism" | 71.5% | 0.902 |
| **R2 (纯标题)** | 删 Artist/Style/Year/模板句/About段 | **0%** | **0.580** |

→ text 里**三层泄露**：明文字段("Style:") → 模板句("in X style") → 艺术家(76% 1:1 → 风格)。
删到只剩标题后，MLP from 0.95→0.30。GNN(graphsage) 0.58：邻居聚合弥补了稀疏标题的信号不足。

## 6. 最终确认 NC 模态消融表 (权威，2026-07-15 R2)

multi−text Δ (=正数表示 image 有正面贡献):

| model | steam Δ | tmdb Δ | wikiart Δ |
|---|---|---|---|
| mlp | +0.024 | +0.040 | **+0.349** |
| gcn | +0.007 | +0.018 | **+0.245** |
| gcnii | +0.027 | +0.040 | **+0.345** |
| graphsage | +0.021 | +0.030 | **+0.245** |

故事：_去泄露后，image 全域转正、跨域互补；wikiart 上 image 碾压 title-only text（Δ+0.35）—— 这才是视觉艺术域该有的结果。_

## 7. 全域实验重跑完成 (2026-07-15, 19:30–20:52)

### 7a. 受 text 泄露影响的实验（需重跑）

| 优先级 | 实验目录 | 泄露机制 | 影响程度 |
|---|---|---|---|
| 🔴 P0 | `newdomain_nc/` | ✅ 已重跑 | — |
| 🔴 P0 | `newdomain_lp/` | ✅ 已重跑(wikiart R2) | — |
| 🔴 P1 | `newdomain_clustering/` | text 含 label → NMI/ARI 虚高 | 高(wikiart NMI 0.85=虚高) |
| 🔴 P1 | `newdomain_fewshot/` | text 含 label → few-shot 虚高 | 高 |
| 🔴 P1 | `newdomain_crossdomain/` | text 含 label → 迁移 acc 虚高 | 高 |
| 🔴 P1 | `newdomain_modality/` | text 含 label → R@1 虚高(TMDB 0.78 vs 期望 0.06) | **极高**(泄漏导致假阳性结论) |
| 🟡 P2 | `newdomain_modality_missing/` | text 含 label → dropout 效应被低估 | 中 |
| 🟡 P2 | `newdomain_robustness/` | text 含 label → 鲁棒性数字不准 | 中 |
| 🟡 P2 | `newdomain_labelnoise/` | text 含 label → 噪声敏感性被污染 | 中 |
| 🟡 P2 | `newdomain_neighbors/` | text 含 label → k-NN 邻居分析被污染 | 中 |
| 🟡 P2 | `steering/` | text 含 label → NC 绝对值虚高；**核心发现(r=0.846)鲁棒** | 低(核心结论不变) |
| 🟡 P2 | `edge_relation_prediction/` | text 含 label → 边类型预测虚高(wikiart 0.98) | 中 |
| 🟡 P2 | `per_class_eval/` | text 含 label → 逐类 F1 绝对值不准 | 低(相对模式可能有效) |
| 🟡 P2 | `fusion_ablation/` | text 含 label → text 模态虚高,image 贡献被低估 | 中 |
| 🟢 P3 | `igc/` | text 含 label → NC-F1 列不准;**greedy=oracle 不受影响** | 低(核心消融靠 LP-ceiling) |

### 7b. 不受 text 泄露影响的实验（arxiv-only 或其他原因）

| 实验目录 | 原因 |
|---|---|
| `lp_ablation_final/` `_fixed/` `_v2/` `_v3/` `_v3a/` `_neg10/` `_noproj/` `_samedomain/` | arxiv-only (text+image+formula), arxiv 摘要不含 category label |
| `lp_sweep/` | arxiv-only |
| `learning_curve/` | arxiv-only |
| `sensitivity/` | arxiv-only |
| `mllm_models/` | arxiv-only |
| `mathbert/` | arxiv formula encoder 比较 |
| `rebuttal/` | arxiv-only |
| `extended/` | arxiv-only |
| `openmag/` | arxiv-only |
| `e2_citation_ablation/` | arxiv citation fuzzy matching |
| `cross_domain_fixed/` | arxiv 子域(cs_lo/cs_si/math_lo/physics_socph)间迁移, text 无 label 泄露 |
| `cross_dataset/` | cora/citeseer/pubmed + CLIP 特征, 标准 benchmark |
| `public_leakage/` | 合成数据 + 公开数据集, 诊断 leaked 特征 |
| `newdomain_efficiency/` | 只测 runtime/memory, 特征内容不影响 |
| `g2image/` | 空目录 (无结果) |
| `g2text/` | PEFT 实验, 使用 arxiv 数据 |
| `dataset_card/` | 元数据, 不涉及实验 |
| `tables/` `analysis/` | 衍生品, 源实验重跑后需更新 |
| `leakage_fix/` `leakage_free_nc/` | 泄露诊断的历史记录, 保留作为修复证据 |

### 7c. 其他类型泄露检查

| 泄露类型 | 检查结果 |
|---|---|
| **边泄露** (train-test 边) | ✅ 已防: `splits.py` 实现跨切分边剪枝, 所有新域实验使用 `prune_cross_split_edges` |
| **Split 泄露** (切分不均) | ✅ 已防: `stratified_split` 每类内分层, 保护小类 (floor=5) |
| **LP 负采样过易** | ⚠️ 已知: wikiart LP random AUC ~0.99 因 homophily 0.89; hard 负采样降至 0.93(GCN)/0.81(MLP)。非泄露, 是图结构固有的可预测性 |
| **Arxiv text 泄露** | ✅ 干净: arxiv 摘要不含 "cs.NE" 等 category label 字符串; `encode_offline.py` 无需 strip |
| **Spotify text 泄露** | ✅ 干净: playlist 名不写入 track markdown (encode_domain.py L67-68 确认) |
| **Image 特征泄露** | ✅ 干净: 图像不包含 label 文字; CLIP-ViT 编码纯视觉内容 |

### 7d. 跨模态检索假阳性 (newdomain_modality 独立问题)

**这是本次审计新发现的最严重问题**：TMDB text→image R@1=0.783 但 CKA=0.0046。

- CKA≈0 → text 和 image 嵌入在**全局子空间**几乎正交
- R@1=0.783 → 但**逐样本** text 和对应的 image 却高度对齐
- 矛盾解释: text 编码了 label 信息 → 同类 text/image 嵌入在 CLIP 联合空间自然聚拢 → 跨模态检索被 label 泄露驱动, 而非真实的语义对齐
- 去泄露后 R@1 预期跌到 ~0.05-0.15 (18 类 random=0.056)

---
*2026-07-15 初次诊断; 2026-07-15 全域审计更新; 2026-07-15 20:52 全域实验全部重跑完成。*
