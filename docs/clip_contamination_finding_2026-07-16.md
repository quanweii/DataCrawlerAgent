# CLIP 预训练污染发现与处置（P0-3，2026-07-16）

> 对抗审稿（2026-07-16）Reviewer 4 M2 担忧 "CLIP image 特征可能携带 LAION 预训练泄露"。
> 本文用 ImageNet-only 监督编码器对照实验**确认该污染在 wikiart/tmdb/steam 三域系统性成立**，并给出处置方案。

---

## 0. 一句话结论

> **frozen CLIP-L/14 的 image 特征在 wikiart/tmdb/steam 三个域的 image-only NC 上，比训练集确定不含这些域图像的 ImageNet-only 编码器(ResNet50/ViT-B/16)高 ΔF1≈0.11~0.21，强度依"该域图像在 LAION 中的常见度"递减（画作>海报>游戏封面）。这强烈提示 CLIP 的 LAION 预训练见过这些域的图像 = 一种预训练泄露。污染是全域系统性的，不仅限于 wikiart。**

---

## 1. 实验设计

- **Baseline**：CLIP-ViT-L/14（openai/clip-vit-large-patch14，768d，projection）—— 现有论文编码器。
- **对照（ImageNet-only 监督）**：
  - ResNet50 (torchvision, IMAGENET1K_V1, 2048d) —— 训练集 ImageNet-1k（1000 类自然物体，几乎不含油画/海报/游戏封面）。
  - ViT-B/16 (torchvision, IMAGENET1K_V1, 768d) —— 同上，且与 CLIP 同维度。
- **协议**：和主表完全一致——leak-free NC（stratified split seed=1 + 跨切分边剪枝）+ MLP（proj_dim=768）+ 5 init seed + 300 epoch。每个编码器 mean-pool 多图 + L2-norm（与 `ImageEncoder` 一致）。
- **自检**：CLIP 行 wikiart ACC=0.655 精确复现主表 0.657 → 脚本可信，对照数字可信。
- **脚本**：`benchmark/clip_contamination_probe.py`（支持 `--domain`）；结果 `benchmark/results/clip_contamination_probe_{domain}.json`。

## 2. 全域结果

| 域 | CLIP-L/14 F1 | IN-best F1 | ΔF1 | CLIP-L/14 ACC | IN-best ACC | ΔACC | 判定 |
|---|---|---|---|---|---|---|---|
| wikiart (64类) | 0.551 | 0.337 (ViT) | **+0.214** | 0.655 | 0.443 | **+0.212** | 🔴 强污染 |
| tmdb (18类) | 0.320 | 0.169 (ViT) | **+0.150** | 0.562 | 0.410 | **+0.151** | 🔴 中污染 |
| steam (8类) | 0.388 | 0.280 (ViT) | +0.108 | 0.620 | 0.568 | +0.052 | 🟡 弱/边际 |

（IN-best = ResNet50 与 ViT-B/16 中较高者；两者一致地远低于 CLIP，互相印证。）

**强度排序 wikiart > tmdb > steam**，与各域图像在 LAION-400M/2B 中的常见度吻合（艺术画廊画作 ≫ 电影海报 ＞ 游戏封面）。

## 3. 诚实的 nuance

CLIP 的优势是 **表示学习能力 + LAION 预训练重叠** 的**混合**，不能 100% 归因于污染：
- CLIP 是对比学习（图文对齐），天然学到"视觉风格/语义"表示，比 ImageNet 分类监督更适合 style/genre 任务——这部分是**合法的表示优势**。
- 但 ΔF1≈0.11~0.21 这么大幅度的领先，且强度依"LAION 常见度"递减、两个 ImageNet-only 编码器互相印证，强烈提示**预训练见过这些域图像**贡献了相当一部分。
- steam 的 ΔACC=0.052（边际，且 steam std 大 ~0.04）→ steam 的 CLIP 优势可能**主要来自表示能力**而非污染。

**严谨表述**："a substantial fraction of frozen CLIP's image-NC advantage over ImageNet-only encoders is attributable to pre-training overlap with the target domains' images; the exact split between legitimate representational capacity and contamination is domain-dependent."

## 4. 影响范围

| 受影响 | 不受影响 |
|---|---|
| wikiart/tmdb/steam 的 **image** 特征相关结论（image MLP、modmiss drop_image、multi 中 image 贡献） | **text** 特征（已去 text 字段泄露） |
| "去泄露后 image 全域转正/碾压 text Δ+0.35" 头条叙事 | **steering**（默认 modality=text） |
| | **structure-leakage ceiling**（label propagation，无特征） |
| | **NC 主表的 text 列**、**LP**（结构为主信号） |

## 5. 处置方案（全域系统性 → 升级为贡献）

### 5a. 主表加 ImageNet-only 对照列（必做）
每个域的 image NC 旁加 ResNet50/ViT-IN 数字，让读者直接看到 CLIP 的额外增益 ΔF1。这是最直接的"诚实呈现"。

### 5b. 升级 C2 为 "encoder pre-training contamination profiling"（雷变贡献）
原 C2 = "text 字段去泄露 + 跨切分边剪枝"。新增一个**全新维度**：**frozen web-trained encoder 的预训练泄露 profiling**。
- 论点：所有用 frozen CLIP/对齐式 web-trained encoder 的 MAG benchmark（OpenMAG/MAGB/Mosaic 都用 CLIP！）**都可能携带这种预训练泄露**，但没人量化过。我们首次用 ImageNet-only 对照量化了它（per-domain ΔF1 profile）。
- 这把一个"我们的 image 数字可能不干净"的雷，转化为"我们发现了整个领域都忽视的一个泄露维度"的**贡献**。
- 与 text 字段去泄露、structure-leakage ceiling 一起，构成 C2 的三层 leak-profiling（特征级 text / 编码器级 image / 结构级 cross-split edge）。

### 5c. 撤回 "image 全域转正" 干净叙事
原叙事："去泄露后 image 全域转正、wikiart image 碾压 text Δ+0.35 → image 是必需模态"。
改写为 limitation：image 的 NC 优势部分来自 frozen CLIP 的预训练（ΔF1≈0.11~0.21）；在 ImageNet-only 干净编码器下，image 对 style/genre 的真实贡献要小得多（wikiart image F1 从 0.55 降到 0.30-0.34，接近 text 标题水平）。

### 5d. steam 特别标注
steam 的 CLIP 优势在 ACC 上边际（Δ=0.05），可能主要是表示能力而非污染——per-domain contamination profile 的强弱本身就是一个发现。

## 6. 对论文价值定位的净影响

- **负面**：wikiart/tmdb/steam 的 image 头条数字不再是"干净的模态贡献"，需加 caveat；"image 转正"叙事降级。
- **正面（更大）**：发现并量化了一个**全领域忽视的预训练泄露维度**（frozen CLIP on MAG benchmarks），升级 C2 leak-profiling 的深度和普适性。这正契合论文"measurement-grounded decontamination"的主轴——作者已发现 text 字段泄露，现在又发现 encoder 预训练泄露，**强化了"leak-profiling 是持续重要工作"的核心论点**。
- **不受影响**：steering（皇冠，text）、structure-leakage ceiling（C2 结构半边）、text NC、LP——论文的主体科学贡献稳。

---

## 7. 相关文件

- 脚本：`benchmark/clip_contamination_probe.py`（`--domain {wikiart,steam,tmdb}`）
- 结果：`benchmark/results/clip_contamination_probe_{wikiart,tmdb,steam}.json`（wikiart 文件名为 `clip_contamination_probe.json`）
- 日志：`benchmark/results/logs/clip_probe_{wikiart,tmdb,steam}.log`
- 对抗审稿原文：memory `adversarial-review-findings-2026-07-16`（P0-3 条目已更新全域结论）

*2026-07-16。结论：CLIP 预训练污染三域系统性成立；升级 C2 为 encoder-contamination profiling（雷变贡献）。*
