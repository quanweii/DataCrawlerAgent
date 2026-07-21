# PPT 配图详细规格说明（FIGURE SPECS）

> 用途：原 matplotlib 图字小模糊，本文档把 **8 张图** 的精确数据、布局、配色、所有文字标注逐项列清，
> 供你用其他 AI 出图工具（如 Napkin / Gamma / 腾讯文档 AI / Midjourney+文字后修 / 手绘 Figma）重画。
> 每张图含：① 用途 ② 图表类型 ③ 精确数据表 ④ 布局尺寸 ⑤ 配色(精确 hex) ⑥ 全部文字标注 ⑦ 给 AI 的现成 prompt。
>
> 配色统一（品牌中性，可粘贴）：
> - tmdb 红 `#E45756` · steam 橙 `#F2A93B` · spotify 蓝 `#4C78A8` · wikiart 绿 `#54A24B` · arxiv 紫 `#9D6AC9`
> - 通用：强调蓝 `#4C78A8` / 警示红 `#E45756` / 灰 `#9A9A9A` / 正向绿 `#54A24B`
> - 背景：纯白 `#FFFFFF`，网格线浅灰 `#E5E7EB`
> - 字体：无衬线（思源黑体 / Inter / Arial），标题 18–22pt 加粗，轴标签 13–14pt，数据标签 11–12pt 加粗
> - 风格：扁平、去 3D、去阴影；柱顶/点旁直接标数值；坐标轴只留左+下两条线

---

## 图 0 — 贡献全景卡片（C1–C4 Overview）→ Slide 3

**用途**：开篇一页讲清四个贡献，可作整页背景。
**类型**：4 张并排信息卡片（不是数据图），横向排列。
**布局**：宽高比约 11.5 : 5.2（横向 banner）。4 卡等宽，卡间距均匀。每卡分上下两段：上段彩色标题条（占卡高 15%），下段白底正文（占 85%）。

**4 卡内容（从左到右）**：

| 卡 | 标题条文字(白字) | 标题条颜色 | 正文(深灰字，分行) |
|---|---|---|---|
| C1 | `C1  Four+1 MAG Datasets` | 紫 `#9D6AC9` | arxiv · steam · tmdb · wikiart (+spotify) / homophily 0.24→0.93 / unique formula modality / leak-profile first-class metadata |
| C2 | `C2  Leak-free Protocol (moat)` | 红 `#E45756` | cross-split edge pruning + LP-ceiling / greedy = no-LLM = oracle / de-leaking is measurement, not LLM |
| C3 | `C3  Declarative Framework + Skill` | 绿 `#54A24B` | 1 YAML = 1 domain · 94.6% reuse / spotify zero-code (0 Python LOC) / skill: build-mag-dataset, 7 stages |
| C4 | `C4  Benchmark Value (steering crown)` | 蓝 `#4C78A8` | same nodes/features, swap edges / GCN−MLP gap vs homophily r=0.846 / LP / cluster / few-shot / cross-domain |

**顶部大标题**（居中，跨整宽）：`DataCrawlerAgent — NeurIPS D&B Contribution Map`

**给 AI 的 prompt**：
> A clean horizontal banner (11.5:5.2, white background) with 4 equal-width info cards side by side. Each card: a colored header bar on top with white bold text, white body below with short bullet text. Card headers left-to-right: purple "C1 Four+1 MAG Datasets", red "C2 Leak-free Protocol (moat)", green "C3 Declarative Framework + Skill", blue "C4 Benchmark Value (steering crown)". A single centered title above the cards: "DataCrawlerAgent — NeurIPS D&B Contribution Map". Flat design, no 3D, no shadows, sans-serif font, crisp readable text. Card body texts: [C1] arxiv·steam·tmdb·wikiart(+spotify); homophily 0.24→0.93; unique formula modality; leak-profile metadata. [C2] cross-split edge pruning + LP-ceiling; greedy=no-LLM=oracle; de-leaking is measurement not LLM. [C3] 1 YAML=1 domain; 94.6% reuse; spotify zero-code (0 Python); build-mag-dataset skill, 7 stages. [C4] same nodes/features swap edges; GCN−MLP gap vs homophily r=0.846; LP/cluster/few-shot/cross-domain.

---

## 图 1 — 同质性谱（Homophily Spectrum）→ Slide 4

**用途**：展示 5 个域同质性横跨自然源 0.24→0.93。
**类型**：横向条形图（horizontal bar chart），5 条。
**布局**：宽高比约 10 : 4.6。Y 轴 = 域名（大写），X 轴 = 同质性值 0 → 1.08。
**精确数据**（条的长 = homophily，从上到下按域固定顺序 tmdb→steam→spotify→wikiart→arxiv）：

| 域(Y轴标签) | homophily(条长) | 条颜色 | 条内白字标注 |
|---|---|---|---|
| TMDB | 0.24 | 红 `#E45756` | 13.0k nodes · 18 classes · text+img |
| STEAM | 0.53 | 橙 `#F2A93B` | 2.8k nodes · 8 classes · text+img |
| SPOTIFY | 0.74 | 蓝 `#4C78A8` | 10.0k nodes · 6 classes · text |
| WIKIART | 0.89 | 绿 `#54A24B` | 22.0k nodes · 64 classes · text+img |
| ARXIV | 0.93 | 紫 `#9D6AC9` | 45.7k nodes · 4 classes · text+img+formula |

**标注**：
- 每条右端（条外）标 homophily 数值，加粗，如 `0.24`。
- 每条内部左端用白字写节点/类数/模态（见上表）。
- X 轴标题：`Edge homophily (fraction of edges connecting same-label nodes)`
- 图标题：`Four+1 MAG benchmarks span a wide homophily spectrum (natural sources, not tuned)`
- X=0.5 处画一条竖虚线（灰 `#9A9A9A`），线旁标 `homo=0.5 (neutral)`。

**给 AI 的 prompt**：
> Horizontal bar chart, 5 bars, white background. Y-axis labels top-to-bottom: TMDB, STEAM, SPOTIFY, WIKIART, ARXIV. Bar colors and lengths (= homophily value): TMDB red 0.24, STEAM orange 0.53, SPOTIFY blue 0.74, WIKIART green 0.89, ARXIV purple 0.93. At the right end of each bar print the value bold. Inside each bar (left, white bold text) print: TMDB "13.0k nodes · 18 classes · text+img"; STEAM "2.8k nodes · 8 classes · text+img"; SPOTIFY "10.0k nodes · 6 classes · text"; WIKIART "22.0k nodes · 64 classes · text+img"; ARXIV "45.7k nodes · 4 classes · text+img+formula". X-axis from 0 to 1.08 labeled "Edge homophily (fraction of edges connecting same-label nodes)". Dashed vertical grey line at x=0.5 labeled "homo=0.5 (neutral)". Title: "Four+1 MAG benchmarks span a wide homophily spectrum (natural sources, not tuned)". Flat, clean, large readable sans-serif font.

---

## 图 2 — LP-ceiling 坍塌（Leak Eliminated）→ Slide 6 【核心护城河图】

**用途**：证明 C2 防泄露协议把结构泄露消除（wikiart 0.921→0.024 冒烟枪）。
**类型**：分组柱状图（grouped bar），4 组 × 每组 3 柱。
**布局**：宽高比约 10 : 4.8。X 轴 4 组（域），每组 3 柱并排。
**精确数据**（每柱顶标数值）：

| X 组标签 | 柱1 BEFORE pruning(红`#E45756`) | 柱2 AFTER pruning(绿`#54A24B`) | 柱3 Majority baseline(灰`#9A9A9A`) |
|---|---|---|---|
| tmdb (0.24) | 0.413 | 0.154 | 0.205 |
| steam (0.53) | 0.557 | 0.504 | 0.504 |
| arxiv (0.93) | 0.673 | 0.261 | 0.288 |
| wikiart (0.89) | 0.921 | 0.024 | 0.025 |

**标注**：
- Y 轴标题：`Label-propagation accuracy (no features)`，范围 0 → 1.08。
- 图例（左上，无框）：红 = `LP-ceiling BEFORE pruning (leaky)`；绿 = `LP-ceiling AFTER pruning (leak-free)`；灰 = `Majority-class baseline`。
- 图标题：`C2 leak-free protocol: pruned LP-ceiling ≤ majority class on all 4 domains`
- **醒目注解**（重点）：用红色箭头从 wikiart 组的红色 BEFORE 柱(0.921)指出来，旁边写红色加粗 `0.921 → 0.024 (structural leak eliminated)`。

**给 AI 的 prompt**：
> Grouped bar chart, 4 groups on x-axis, 3 bars per group, white background. X-axis groups: "tmdb (0.24)", "steam (0.53)", "arxiv (0.93)", "wikiart (0.89)". In each group 3 side-by-side bars: red BEFORE-pruning, green AFTER-pruning, grey majority-baseline. Values — tmdb: 0.413 / 0.154 / 0.205; steam: 0.557 / 0.504 / 0.504; arxiv: 0.673 / 0.261 / 0.288; wikiart: 0.921 / 0.024 / 0.025. Print each value on top of its bar (bold). Y-axis "Label-propagation accuracy (no features)" 0 to 1.08. Legend top-left (no border): red=LP-ceiling BEFORE pruning (leaky), green=LP-ceiling AFTER pruning (leak-free), grey=Majority-class baseline. Title: "C2 leak-free protocol: pruned LP-ceiling ≤ majority class on all 4 domains". Add a bold red arrow + annotation on the wikiart red bar (0.921) reading "0.921 → 0.024 (structural leak eliminated)". Clean flat sans-serif, large readable text.

---

## 图 3 — Steering 因果实验（皇冠）→ Slide 10 【最重要一图】

**用途**：因果证明"换边→改下游"，GCN−MLP gap vs 同质性 r=0.846。
**类型**：左右双子图（1 行 2 列）。
**布局**：宽高比约 13 : 4.8。左图 = 散点+回归线；右图 = 散点+MLP 平坦参考线。

**左子图 — gap vs homophily**：
- 类型：散点图 + 一条蓝色回归直线 + y=0 水平虚线。
- 12 个点（x=homophily, y=GCN_F1 − MLP_F1），按域着色：
  - steam(橙`#F2A93B`)：(0.529, −0.231), (0.817, +0.031), (0.595, −0.153), (0.440, −0.263), (0.437, −0.064)
  - tmdb(红`#E45756`)：(0.237, −0.284), (0.329, −0.178), (0.294, −0.187)
  - wikiart(绿`#54A24B`)：(0.882, +0.055), (0.909, +0.041)
  - （12 点含同域多关系；颜色按域即可）
- 蓝色 `#4C78A8` 回归直线，斜率 0.364 / 截距 −0.332；图例标 `linear fit (r=0.846)`。
- y=0 处水平灰色虚线。
- 在 x≈0.91 处一条红色 `#E45756` 竖虚线，旁标红字 `crossover h≈0.91`。
- X 轴：`Edge homophily (edges swapped, same nodes/features)`
- Y 轴：`GCN−MLP macro-F1 gap`，范围约 −0.4 → +0.18。
- 左子图标题：`Same nodes/features, only edges change → gap tracks homophily`
- 左上角文字框（浅蓝底 `#F0F4FA`，蓝边）：`r = 0.846  [boot 95% CI 0.60, 0.96]` 换行 `p = 0.0005, n = 12`
- 右下角图例（无框）：橙点=steam、红点=tmdb、绿点=wikiart。

**右子图 — GCN F1 vs homophily + MLP 平坦参考**：
- 类型：散点图（同 12 点的 GCN_F1）+ 每域一条同色水平虚线代表该域 MLP（恒定）。
- 数据点(x=homophily, y=GCN_F1)：steam(0.529,0.307),(0.817,0.568),(0.595,0.385),(0.440,0.274),(0.437,0.473)；tmdb(0.237,0.220),(0.329,0.325),(0.294,0.305)；wikiart(0.882,0.858),(0.909,0.844) 等。MLP 参考值：steam≈0.537、tmdb≈0.504、wikiart≈0.803。
- 每域用其颜色画一条**虚线**横跨该域 homophily 范围 = MLP 值（表示 MLP 不随边变）。
- X 轴：`Edge homophily`；Y 轴：`macro-F1`。
- 右子图标题：`GNN benefit rises with homophily; MLP flat (ignores edges)`
- 左下角灰字：`dashed = MLP (edge-independent)` 换行 `GCN-F1 vs homo r=0.913`

**给 AI 的 prompt（左图）**：
> Scatter plot with regression line, white background. 12 points, x=homophily (0.2–0.95), y=(GCN_F1 minus MLP_F1) (range −0.4 to +0.18). Points colored by domain: orange=steam, red=tmdb, green=wikiart. A blue regression line slope 0.364. Horizontal grey dashed line at y=0. Vertical red dashed line at x≈0.91 labeled "crossover h≈0.91" in red. X-axis "Edge homophily (edges swapped, same nodes/features)", Y-axis "GCN−MLP macro-F1 gap". A light-blue info box top-left: "r = 0.846 [boot 95% CI 0.60, 0.96]" / "p = 0.0005, n = 12". Legend bottom-right: orange=steam, red=tmdb, green=wikiart. Title: "Same nodes/features, only edges change → gap tracks homophily". Large crisp sans-serif.

---

## 图 4 — 代码复用率（Code Reuse）→ Slide 8

**用途**：C3 可复用性，spotify 零代码 100%、三域 ~95%、arxiv 例外。
**类型**：竖向柱状图，5 柱。
**布局**：宽高比约 10 : 4.6。Y 轴 0 → 108。
**精确数据**（柱顶标百分比，加粗）：

| X 轴标签 | 复用率 | 柱颜色 |
|---|---|---|
| spotify (zero-code) | 100.0% | 蓝 `#4C78A8` |
| wikiart | 95.0% | 绿 `#54A24B` |
| tmdb | 94.8% | 红 `#E45756` |
| steam | 94.6% | 橙 `#F2A93B` |
| arxiv (legacy) | 67.7% | 紫 `#9D6AC9` |

**标注**：
- Y 轴标题：`Shared-skeleton reuse = shared / (shared + per-domain incremental)`
- 图标题：`C3 reusability: 1 YAML = 1 domain (spotify 0 Python LOC); arxiv footnoted`
- y=94.0 处一条绿色 `#54A24B` 水平虚线，旁标绿字 `headline 94% (3 co-attribute domains)`

**给 AI 的 prompt**：
> Vertical bar chart, 5 bars, white background. X-axis labels: "spotify (zero-code)", "wikiart", "tmdb", "steam", "arxiv (legacy)". Bar values (print bold on top): 100.0%, 95.0%, 94.8%, 94.6%, 67.7%. Bar colors: spotify blue, wikiart green, tmdb red, steam orange, arxiv purple. Y-axis 0–108 labeled "Shared-skeleton reuse = shared / (shared + per-domain incremental)". Horizontal green dashed line at y=94.0 labeled "headline 94% (3 co-attribute domains)" in green. Title: "C3 reusability: 1 YAML = 1 domain (spotify 0 Python LOC); arxiv footnoted". Clean flat large sans-serif.

---

## 图 5 — 公平模态消融（Fair Modality Ablation, 768-d 投影）→ Slide 12

**用途**：投影后 text 单独最强，image/formula = 丰富度非性能。
**类型**：分组柱状图，4 组（域）× 每组 4 柱（text / image / text+image / text+image+formula，最后者仅 arxiv 有）。
**布局**：宽高比约 11 : 4.8。Y 轴 0 → 1.0。
**精确数据**：

| X 组标签 | text(蓝`#4C78A8`) | image(橙`#F2A93B`) | text+image(绿`#54A24B`) | text+image+formula(紫`#9D6AC9`) |
|---|---|---|---|---|
| steam (homo 0.53) | 0.662 | 0.640 | 0.683 | — |
| tmdb (homo 0.24) | 0.667 | 0.554 | 0.647 | — |
| wikiart (homo 0.89) | 0.916 | 0.769 | 0.906 | — |
| arxiv (homo 0.93) | 0.906 | 0.819 | 0.895 | 0.896 |

**标注**：
- Y 轴标题：`Node-classification accuracy (GCNII, leak-free)`
- 图标题：`Fair modality ablation (all projected to 768-d): text alone is strongest; image/formula = richness, not accuracy`
- 图例（左下，2 列）：text / image / text+image / text+image+formula (arxiv only)
- 在 text 柱与 text+image 柱顶部各标数值（白字嵌柱内或柱顶）。

**给 AI 的 prompt**：
> Grouped bar chart, 4 groups on x-axis, up to 4 bars per group, white background. X-axis: "steam (homo 0.53)", "tmdb (homo 0.24)", "wikiart (homo 0.89)", "arxiv (homo 0.93)". In each group 4 bars: blue text, orange image, green text+image, purple text+image+formula (purple only on arxiv, absent elsewhere). Values — steam: 0.662/0.640/0.683; tmdb: 0.667/0.554/0.647; wikiart: 0.916/0.769/0.906; arxiv: 0.906/0.819/0.895/0.896. Y-axis 0–1.0 "Node-classification accuracy (GCNII, leak-free)". Legend bottom-left: text / image / text+image / text+image+formula (arxiv only). Title: "Fair modality ablation (all projected to 768-d): text alone is strongest; image/formula = richness, not accuracy". Clean flat large sans-serif.

---

## 图 6 — Leak-free 主表（MLP vs GCN vs best-GNN）→ Slide 11

**用途**：去泄露后 MLP 竞争力强，GNN 仅最高同质 arxiv 边际正。
**类型**：分组柱状图，4 组 × 3 柱。
**布局**：宽高比约 10.5 : 4.8。Y 轴 0 → 1.04。
**精确数据 + Δ(best−MLP) 标注**：

| X 组标签 | MLP(蓝`#4C78A8`) | GCN(红`#E45756`) | best-GNN(绿`#54A24B`) | best-GNN 柱顶 Δ 标注 |
|---|---|---|---|---|
| tmdb (homo 0.24) | 0.662 | 0.487 | 0.643 | Δ−0.019(红字) |
| steam (homo 0.53) | 0.694 | 0.611 | 0.681 | Δ−0.013(红字) |
| wikiart (homo 0.89) | 0.933 | 0.900 | 0.922 | Δ−0.011(红字) |
| arxiv (homo 0.93) | 0.899 | 0.904 | 0.908 | Δ+0.009(黑字) |

**标注**：
- 每根 MLP 柱顶标其数值（加粗）。
- Y 轴标题：`Accuracy (leak-free pruned graph)`
- 图标题：`Leak-free NC table: MLP competitive (leak removed); GNN helps only at highest homophily`
- 图例（右下）：MLP (no edges) / GCN / best GNN

**给 AI 的 prompt**：
> Grouped bar chart, 4 groups, 3 bars per group, white background. X-axis: "tmdb (homo 0.24)", "steam (homo 0.53)", "wikiart (homo 0.89)", "arxiv (homo 0.93)". Each group: blue MLP, red GCN, green best-GNN. Values — tmdb: 0.662/0.487/0.643; steam: 0.694/0.611/0.681; wikiart: 0.933/0.900/0.922; arxiv: 0.899/0.904/0.908. Print the MLP value bold above the blue bar. Above each green best-GNN bar print "Δ−0.019", "Δ−0.013", "Δ−0.011" in red for the first three, and "Δ+0.009" in black for arxiv. Y-axis 0–1.04 "Accuracy (leak-free pruned graph)". Legend bottom-right: MLP (no edges) / GCN / best GNN. Title: "Leak-free NC table: MLP competitive (leak removed); GNN helps only at highest homophily". Clean flat large sans-serif.

---

## 图 7 — 鲁棒性（feature vs edge 噪声）→ Slide 14

**用途**：边噪声比特征噪声更伤，低同质 tmdb 最敏感。
**类型**：分组柱状图，3 组 × 2 柱。
**布局**：宽高比约 8.5 : 4.6。Y 轴 0 → 0.38。
**精确数据**（柱顶标数值）：

| X 组标签 | feature noise(橙`#F2A93B`) | edge noise(红`#E45756`) |
|---|---|---|
| steam (homo 0.53) | 0.083 | 0.108 |
| tmdb (homo 0.24) | 0.212 | 0.234 |
| wikiart (homo 0.89) | 0.090 | 0.327 |

**标注**：
- Y 轴标题：`|Δ accuracy| at ρ=0.8 noise (lower = more robust)`
- 图标题：`Robustness: edge noise hurts more than feature noise; low-homophily tmdb most sensitive`
- 图例（左上）：feature noise (rows→Gaussian) / edge noise (add random edges)

**给 AI 的 prompt**：
> Grouped bar chart, 3 groups, 2 bars per group, white background. X-axis: "steam (homo 0.53)", "tmdb (homo 0.24)", "wikiart (homo 0.89)". Each group: orange feature-noise, red edge-noise. Values — steam: 0.083/0.108; tmdb: 0.212/0.234; wikiart: 0.090/0.327. Print each value on top of its bar. Y-axis 0–0.38 "|Δ accuracy| at ρ=0.8 noise (lower = more robust)". Legend top-left: feature noise (rows→Gaussian) / edge noise (add random edges). Title: "Robustness: edge noise hurts more than feature noise; low-homophily tmdb most sensitive". Clean flat large sans-serif.

---

## 通用出图建议（给所有图）

1. **分辨率**：导出 PNG/SVG，宽度 ≥ 2400px（PPT 全宽不糊）。SVG 最佳，可无损放大。
2. **字号**：标题 ≥ 20pt，轴标签 ≥ 14pt，数据标签 ≥ 12pt——宁可图大、字大，不要挤。
3. **留白**：图例不压数据；轴标题与刻度保持距离。
4. **去 chartjunk**：去掉上/右边框、3D、渐变填充、阴影；网格线只用极浅灰水平线。
5. **配色一致性**：8 张图域颜色固定（tmdb红/steam橙/spotify蓝/wikiart绿/arxiv紫），整套 PPT 视觉统一。
6. **顺序**：横轴若含同质性，建议按同质性升序（tmdb→steam→wikiart→arxiv），与 steering 故事线一致。

> 提示：若你用 **Napkin / Gamma / 腾讯文档AI** 这类"文字→图"工具，直接把每张图的「给 AI 的 prompt」+「精确数据表」一起粘进去即可。若用 Excel/PPT 原生图表，按「精确数据 + 标注」手摆最快。
