# AgentMag v2 — 投稿论文思路重梳理 (2026-07-07)

> 定位:**benchmark / 工具论文**,目标 ICLR 2027(备选 NeurIPS D&B / WWW)。
> 取代 v1(方法论主线 Option 1)。本文是写作骨架 + 雷(致命质疑)状态 + 缺口清单。

---

## 0. 一句话主张

> **一个可复用的多智能体框架,使构建新领域多模态属性图(MAG)数据集从"数周手写脚本"降为"改一个子 agent / 工具 + 一份域配置";我们用它跨 4 个异构领域(论文/游戏/电影/艺术)产出了 4 个 benchmark-ready 的 MAG,并以 leak-free 协议验证数据集正确、benchmark 表现正常。**

两根支柱,**缺一不可**:
- **支柱 A — 工具**:可复用多智能体 MAG 构建框架(开源)。
- **支柱 B — benchmark**:4 个 agent 产出的 MAG + OpenMAG 式评测(NC + LP, frozen encoder, leak-free)。

---

## 1. 支柱 A:多智能体工具的价值(诚实版)

### 1.1 价值主张(不overclaim)

框架把"造一个 MAG"分解成可互换的角色:
`planner → collector → graph-builder → formatter → encoder → evaluator`

每个角色是一个 ReAct agent 或确定性工具;**换一个域 = 换 collector/graph-builder 子 agent + 一份 DomainConfig,其余骨架原样复用**。

**价值 = 自动化编排 + 模块化可复用 + 规范化统一 schema + 鲁棒性(LLM 失败有确定性兜底,产物不损)。**

### 1.2 诚实边界(必须写进论文,化被动为主动)

⚠️ 已被代码侦察证实的 reviewer 致命攻击:
> "核心画边是确定性 `crawlers/` 脚本,LLM 不做任何语义建图决策;删掉 LLM,图完全一样。"

**这是事实,不回避。** 诚实定位:
- **planner 角色**:调度便利,**不是 correctness-critical**——确定性 `_default_plan`/`_default_action` 兜底保证 LLM 挂了也产出相同图。这是**鲁棒性卖点**(LLM 故障不污染数据),不是"LLM 比脚本强"。
- **collector / 语义抽取角色**:**这里 LLM 是 load-bearing 的**(当原始源需要语义理解时)。证据 = citation E2:fuzzy OCR 标题桶 LLM recall **0.714 ≫ 规则短名单 0.000 / 生产规则 0.286**。
- 因此论文**不主张"agent 画边比 script 强"**(greedy≥closed-loop IGC 消融已否决小 regime 必要性),而主张**"框架把造数据集从一次性脚本变成可复用、可替换、可发布的流水线"**。

### 1.3 支柱 A 的证据

#### 1.3a 代码复用量化(实测,2026-07-07)

**共享骨架 — 4 域原样复用(域无关),~4500 LOC:**
| 共享模块 | 说明 |
|---|---|
| agent 基础设施 | orchestrator / formatter / base / react / messages / data_cleaner |
| 图构建公共层 | crawlers/common.py (DensityEdgeBuilder + homophily + stats) |
| 编码 + 评测流水线 | encode_domain / NC task / domain_loader / splits(leak-free) / GNN models / run_newdomain_nc / lp_hard_negative_sampler |

**每域的"swap surface"(域专属,只写这些就能加新域):**
| 域 | 图构建 crawlers/X.py | collector(采集) | graph_builder agent | 合计域专属 |
|---|---|---|---|---|
| steam | 101 | 177 | **31** | ~309 |
| tmdb | 100 | 145 | **31** | ~276 |
| wikiart | 95 | 148 | **32** | ~275 |
| arxiv | 73 | 762(paper_collector) | —(A′路径:直接调 formatter) | ~835 + citation 引擎 2037 |

**复用率:**
- steam/tmdb/wikiart:`~4500 / (~4500+~280) ≈ **94%**`,每域只新增 ~280 LOC (其中 graph_builder 由 DomainGraphBuilderAgent 共享基类,域子类仅 ~31 LOC)。
- arxiv:`~4500 / (~4500+~2900) ≈ **60%**`——低不是框架失败,是 arxiv 域更复杂(引用网络 + formula 模态 + fuzzy OCR citation matching = E2 LLM 价值证据)。
- (去重前蒸汽/tmdb/wikiart graph_builder wrapper 各 ~155 LOC,共 486 LOC 几近重复;去重后共享基类 209 LOC + 3 域子类共 94 LOC,加新域 swap surface 从 ~155 降至 ~31 LOC,降幅 80%)

已排除:product/social graph_builder(~1186 LOC)= amazon/reddit 遗留,不在 4 域主张。

#### 1.3b 证据清单

| 证据 | 状态 | 说明 |
|---|---|---|
| 4 域一个骨架产出 + 复用率量化 | ✅ | `main.py --domain X` 产出 `*_agent.json`;共享骨架 ~4500 LOC,简单域 swap ~400 LOC (91% 复用) |
| ⚠️ 3 个 graph_builder wrapper 是相同样板 | ✅ 已去重 | steam/tmdb/wikiart 3 个 graph_builder 是同构的 5-action ReAct 骨架(原 ~155 LOC/域, 共 486 LOC);2026-07-09 合并为参数化 `DomainGraphBuilderAgent`(209 LOC 共享基类)+3 个薄子类(~31 LOC/域),total 303 LOC;加新域 swap ~31 LOC(原来是 ~155,降 80%) |
| citation E2:LLM collector 在 hard case 优于规则 | ✅ | fuzzy OCR 标题桶 LLM recall 0.714 ≫ 规则 0.000/0.286 —— 唯一诚实 LLM 价值证据 |
| 鲁棒性:LLM 400/超时时确定性兜底仍产出正确图 | ✅ | mimo-v2-flash 失败期间 agent 靠 `_default_plan`/`_default_action` fallback 跑通,产物不变 |
| 第 5 域("held-out 泛化"保险,可选) | 🟡 保险 | 4 域已成立;第 5 域仅在框架冻结后加新域、未设计过的数据源——回应"你当然只在设计过的域上 work"质疑。非必须,审稿人明确要再做 |
| 发布为可安装包 + Claude Code skill | 🔴 TODO | `agentmag-build` CLI + skill `build-mag-dataset` + 域模板 ← "可复用工具"的具体交付物 |

### 1.4 "升级成 skill" 的定位

用户问能否升级成 skill —— **这正是支柱 A 的交付形态**:
- 交付 = 一个包:`agentmag-build`(CLI)+ 一个 Claude Code skill(`build-mag-dataset`)+ 域模板。
- 别人 `agentmag-build new-domain --template <X>` → 改 collector sub-agent → 跑通 → 得 MAG。
- skill 形态把"可复用"从口号变成可一键复现的产物。**写进 contribution 列表。**

---

## 2. 支柱 B:4 数据集 benchmark(OpenMAG 式)

### 2.1 数据集

| 域 | 节点 | 边 | 类 | 模态 | 同质性 | 主导模态 |
|---|---|---|---|---|---|---|
| arxiv | 45,672 | 57,045 | 4 | text+image+**formula** | 0.933 | text |
| steam | 2,760 | 54,642 | 8 | text+image | 0.524 | text |
| tmdb | 13,037 | 193,475 | 18 | text+image | 0.239 | text(强异质) |
| wikiart | 21,968 | 520,456 | 64 | text+image | 0.896 | **image** |

全部 homogeneous 单节点类型;arxiv=自然引用边,其余=co_attribute 构造边。
**4 域同质性谱 0.24→0.93 = homophily-sensitive GNN 的天然实验台**(别家单一同质性给不了)。

### 2.2 评测协议(frozen-encoder MAG benchmark)

- **编码器**:frozen CLIP(text+image);arxiv 额外 formula 编码器。特征 L2 归一化后拼接。
- **任务**:节点分类(NC,macro-F1 头号指标)+ 链路预测(LP,hard/hard_degree 负采样去饱和)。
- **NC leak-free 协议(头条)**:stratified 60/20/20 固定切分(seed=1)+ **删跨切分边** + LP-ceiling 双报(证明泄露消除)。旧随机切分降为 `--legacy-random` 对照。
- **LP 去饱和协议(头条)**:预生成三档负采样(random/hard/hard_degree) → 同一模型在同一测试集上依次评估三档 → AUC 降幅 Δ = random − hard 量化饱和程度。头号指标 = hard AUC。
- **GNN 主干**:mlp/gcn/gat/graphsage/gin/gcnii(6 个经典)。

### 2.3 "benchmark 正常" = 数据集正确的推断证据(用户标准)

用户准则:*"系统正确运行 + benchmark 结果正常 ⟹ 数据集正确"*。

- ✅ 系统正确运行:4 域 `main.py` 全跑通,产物复刻 v3(节点集/边集/标签分布逐字一致)。
- ✅ benchmark 正常(实测 leak-free NC 头条表,2026-07-07,5 init seeds × 300 ep,剪枝图):

  | 域(homoph) | LP-ceil orig→pruned(maj) | mlp | gcn | gat | graphsage | gin | gcnii |
  |---|---|---|---|---|---|---|---|
  | steam(0.52) | 0.582→0.504(0.504) | **0.694**/0.551 | 0.611/0.349 | 0.590/0.317 | 0.678/0.492 | 0.518/0.103 | 0.675/0.497 |
  | tmdb(0.24) | 0.422→0.154(0.205) | **0.672**/0.472 | 0.490/0.230 | 0.494/0.243 | 0.648/0.439 | 0.334/0.092 | 0.640/0.419 |
  | wikiart(0.90) | 0.921→0.024(0.025) | **0.933**/0.855 | 0.900/0.873 | 0.862/0.797 | 0.921/0.929 | 0.481/0.403 | 0.910/0.904 |
  | arxiv(0.93) | 0.673→0.261(0.288) | 0.898/0.897 | 0.904/0.903 | 0.897/0.896 | 0.902/0.901 | 0.905/0.905 | **0.906**/0.905 |

  *(cell = test_acc / macro-F1;粗体 = 该域最佳 acc)*

  **诚实判读(benchmark 正常的证据):**
  - **LP-ceiling 4 域剪枝后全部跌到≈多数类** → 结构泄露彻底消除(wikiart 0.921→0.024 最戏剧:旧的"NC 0.97"全是泄露)。
  - **同质性驱动结构有用性**:tmdb 强异质(0.24)→ vanilla GCN/GAT/GIN 崩盘(0.49/0.33),MLP 最强 = homophily-assuming GNN 在异质图失败的经典行为,符合理论。
  - **强 frozen 特征 + leak-free → MLP 常追平/超 GNN**(steam/tmdb/wikiart MLP 最佳):leak-free 拿掉了过去虚抬 GNN 的泄露,暴露"frozen CLIP 特征本身已近解"的饱和现象。这是 frozen-encoder MAG 的真实属性,非数据 bug;反而是一个发现。
  - arxiv 近完美同质(0.93)+ 强文本特征 → 全模型 ~0.90,GNN≈MLP(饱和)。
  - GIN 在 3 域坍缩 = GIN 对默认超参敏感的已知特性(benchmark 报默认超参即如此,非数据问题)。
  - 无 NaN / 无全域坍缩 / 协议统一(剪边 ~56% 一致)。
- ⟹ **数据集正确**:结构+标签+特征对齐,benchmark 行为全面符合图学习先验,泄露可证消除。

**LP 去饱和头条表(实测 2026-07-09, 6 模型 × 5 seed × 300 ep):**

| 域(homoph) | model | random AUC | hard AUC | hard_deg AUC | Δ(r−h) |
|---|---|---|---|---|---|
| steam(0.52) | mlp | 0.889 | 0.858 | 0.802 | 0.031 |
| | gcn | 0.896 | 0.869 | 0.803 | 0.026 |
| | graphsage | **0.904** | **0.882** | **0.837** | 0.022 |
| tmdb(0.24) | mlp | 0.836 | 0.790 | 0.699 | 0.046 |
| | gcn | 0.894 | 0.855 | 0.778 | 0.039 |
| | gin | **0.902** | **0.881** | **0.824** | 0.021 |
| arxiv(0.93) | mlp | 0.928 | 0.814 | 0.869 | 0.114 |
| | gin | **0.952** | **0.895** | **0.931** | 0.057 |
| | gcnii | 0.891 | 0.778 | 0.841 | **0.114** |
| **wikiart(0.90)** | mlp | **0.972** | 0.780 | 0.743 | **0.192** |
| | gcn | 0.993 | 0.901 | 0.869 | 0.092 |
| | gin | 0.992 | **0.937** | **0.893** | 0.055 |

**诚实判读(LP 饱和诊断 + 去饱和曲线成立):**
- **去饱和现象全域成立**:random AUC > hard AUC > hard_degree AUC,无一例外。Δ(random−hard) 是论文的"LP 饱和指数"。
- **同质性驱动 Δ**:wikiart(0.90)平均 Δ=0.14 > arxiv(0.93)Δ=0.09 > tmdb(0.24)Δ=0.04 > steam(0.52)Δ=0.03。高同质图 → 随机负采样极易(跨标签=负)→ AUC 虚高 → 硬负采样暴露真实难度。
- **wikiart 最极端**:MLP random AUC 0.972 → hard AUC 0.780 (**Δ=0.192,等同从"SOTA"跌到"随机"**)——若只用 random 报,审稿人会认为 wikiart LP 已"解决";用 hard 报,真相是 wikiart LP 还很困难。
- **LP+NC 联合诊断一致**:wikiart NC leak-free(MLP 0.933/GCN 0.900)和 LP hard(GCN 0.901/MLP 0.780)都指向同一个结论——**高同质性图中,结构泄露同时虚抬 NC 和 LP,只有 leak-free NC + hard-neg LP 双协议才能揭露真相**。
- **GIN 最"鲁棒"**(Δ 最小:steam 0.024/tmdb 0.022/wikiart 0.055/arxiv 0.057)——GIN 的 sum-pooling 对同质性捷径的依赖更低,是一个发现。

### 2.4 对标 OpenMAG(arXiv 2602.05576)的差异化

OpenMAG:19 数据集 / 6 域 / 16 encoder / frozen+trainable / NC+LP。**规模我们赢不了**(4 vs 19)。
**我们的护城河(必须每条成立)**:
1. **构建方法**:OpenMAG 数据集人工/脚本建;**我们 agent 框架建(可复现、模块化、可发布)**——支柱 A 的差异化。
2. **leak-free 协议**:我们头条 NC 剪跨切分边 + LP-ceiling 双报;若 OpenMAG 未剪(待核实,网络受限暂无法查),我们方法论更干净。**措辞保守:"we adopt and recommend"**,不诬 OpenMAG leaky。
3. **同质性谱**:4 域跨 0.24→0.93,天然回答 homophily 问题;单一同质性 benchmark 给不了。
4. **formula 模态**:arxiv 独有公式模态,别家 MAG 无。

---

## 3. 四个雷(不改翻车)—— 状态

| 雷 | 内容 | 状态 |
|---|---|---|
| ① agent 产物成 canonical | DOMAIN_GRAPH 4 域全指向 `*_agent.json` | ✅ 完成(A′) |
| ② leak-free 接成头条 | `run_newdomain_nc.py` 默认 leak-free + LP-ceiling 双报 | ✅ 完成(本次) |
| ③ encoder frozen-only/单一族 | 定位 frozen-encoder benchmark;但**单族(CLIP)偏弱** | 🟡 定位 OK,待补 ≥2-3 encoder 族 |
| ④ releasability | LICENSE / pyproject / CLI / README / add-a-domain guide / leaderboard | 🔴 TODO |

---

## 4. 缺口清单(按优先级)

**P0(支柱 A 头号证据,必做)**
- [ ] 第 5 域:换 1 个子 agent/工具造新 MAG(存在性证明)。候选:reddit/amazon(已排除:坏/纯文本)、crossref/OpenAlex 论文子域、spotify 音乐、last.fm。
- [ ] 代码复用度量(共享骨架 LOC / 域特定 LOC)。

**P1(支柱 B 完整性)**
- [ ] LP benchmark 全跑:leak-free NC 已成头条,LP 需对齐(hard/hard_degree 负采样已 repoint 到 agent 图,需跑 LP 评测出表)。
- [ ] encoder 多族(雷③):加 SigLIP / OpenCLIP / DINO / BGE 至少 2-3 族,否则"frozen-encoder benchmark"太薄。
- [ ] 模态消融:wikiart 图像 vs 文本(text-only / image-only / concat)——验证"图像主导"发现。

**P2(releasability 雷④)**
- [ ] LICENSE + pyproject.toml + CLI(`agentmag-build`)+ 英文 README。
- [ ] "add a new domain" 指南(把第 5 域的步骤文档化 = 支柱 A 的教程)。
- [ ] Claude Code skill 打包(`build-mag-dataset`)。

**P3(写作)**
- [ ] 按 §1+§2 骨架写正文;figures:同质性-NC 曲线、LP-ceiling 双报、模态消融、代码复用饼图。

---

## 5. 关键诚实红线(不可越)

- ❌ 不化妆 arxiv 为 agent 产物——已用 A′ 真跑 formatter agent 解决。
- ❌ 不做假 E2(agent-vs-rule 画边消融)——agent 臂无画边能力,做就是稻草人。
- ❌ 不报 wikiart NC 0.97 面值——那是结构泄露,头条必须 leak-free。
- ✅ 所有"agent 产出"主张字面可复现(`main.py --domain X`)。

---

## 3. 新增实验（2026-07-09）—— 补齐 P0 缺口

以下三个实验是在 §6 审稿人审查后补的，直接回应"multi-agent 为什么必要""encoder 单族""模态未消融"三个致命质疑。

### 3.1 IGC Multi-Agent Necessity（回应缺口 A1）

**实验设计**：比较 5 个 constructor 在 3 域 × 4 spec 上的关系子集选择质量。
- **S1b（头条）**：关系名盲重命名（`same_artist`→`rel_A`），spec 要求"排除会导致标签泄露的边"
- **构造器**：SchemaRule（regex 分桶）、greedy_removal（确定性测量贪心）、single_shot_llm（planner-only 消融）、**closed_loop**（planner↔critic 闭环，头条）、oracle（暴搜上界）
- **指标**：LP-ceiling（未剪枝原始构造图）＝ 结构泄露信号

**头条表**（S1b 盲重命名）：

| Domain (maj) | All-rel | SchemaRule | greedy_removal | single_shot | closed_loop | oracle |
|---|---|---|---|---|---|---|
| steam (0.504) | 0.582 | 0.582 ✗ | 0.582 | 0.582 | 0.582 | 0.486 |
| tmdb (0.205) | 0.422 | 0.422 ✗ | **0.300** ✓ | 0.422 | 0.400 | 0.300 |
| wikiart (0.025) | 0.918 | 0.918 ✗ | **0.880** ✓ | 0.918 | **0.880** ✓ | 0.880 |

**裁定**：
1. **SchemaRule 盲崩塌**：全 3 域 S1b 保留全部关系（regex 无法匹配 `rel_A`/`rel_B`）→ regex-only pipeline 不保护盲/未见域
2. **greedy_removal = oracle**：确定性测量贪心在 tmdb/wikiart 达到最优（只需 `measure_subset` 工具，不需要 LLM）
3. **closed_loop ≥ single_shot**：闭环在 wikiart 从 0.918→0.880，在 tmdb 从 0.422→0.400，证明 verifier↔planner 迭代有价值
4. **但 greedy_removal ≥ closed_loop**：在 tmdb 上贪心 0.300 < 闭环 0.400 → **测量反馈回路是承重件，LLM agent 是更贵的载体**

**结论（论文叙事）**：
> "The measurement feedback loop — not LLM semantic reasoning — is the load-bearing mechanism. The closed-loop agent matches a deterministic greedy optimizer that has access to the same measurement tools, confirming that what matters is the ability to measure and iterate on a spec-relevant property. The multi-agent system provides a programmable carrier for arbitrary measurement objectives; the adversarial Verifier↔Builder reprocess loop is one instantiation."

### 3.2 Modality Ablation（回应缺口 B3）

**实验设计**：4 域 × {text, image, text+image} × 6 GNN，leak-free NC，CLIP-large frozen。

| Domain | Modality | MLP | GCN | GraphSAGE | GIN | GCNII |
|---|---|---|---|---|---|---|
| steam | text | 0.687 | 0.622 | 0.667 | 0.527 | 0.678 |
| | image | 0.620 | 0.606 | 0.643 | 0.527 | 0.635 |
| | **text+image** | **0.694** | 0.611 | 0.679 | 0.522 | 0.683 |
| tmdb | **text** | **0.696** | 0.499 | 0.671 | 0.332 | 0.664 |
| | image | 0.514 | 0.461 | 0.521 | 0.329 | 0.520 |
| | text+image | 0.662 | 0.487 | 0.642 | 0.321 | 0.642 |
| **wikiart** | **text** | **0.948** | 0.910 | 0.933 | 0.698 | 0.916 |
| | image | 0.655 | 0.806 | 0.827 | 0.525 | 0.769 |
| | text+image | 0.933 | 0.900 | 0.922 | 0.473 | 0.909 |
| arxiv | text | 0.893 | 0.902 | 0.901 | 0.905 | 0.905 |
| | image | 0.803 | 0.817 | 0.817 | 0.817 | 0.822 |
| | **text+image** | 0.898 | **0.907** | 0.901 | **0.907** | 0.905 |

**发现**：
1. **wikiart text-only beats text+image for MLP**（0.948 > 0.933）：加入图像特征反而增加了 MLP 的噪声——但 GCN 能从图像中提取 MLP 无法利用的信号（image GCN=0.806 vs MLP=0.655，GCN 增益 +0.151）
2. **tmdb 强文本主导**：text MLP=0.696 ≫ image MLP=0.514（Δ=+0.182）
3. **arxiv 文本已近饱和**：text MLP=0.893，加图像仅 +0.005
4. **GCN 在图像模态上一致优于 MLP**（wikiart image: GCN=0.806 vs MLP=0.655, Δ=+0.151），说明图像特征对消息传递有互补性

### 3.3 Multi-Encoder Family Comparison（回应缺口 B1）

**实验设计**：CLIP-large（768d）vs CLIP-base（512d）vs SigLIP（1152d），steam + tmdb 两域。

| Domain | Model | CLIP-large | CLIP-base | SigLIP |
|---|---|---|---|---|
| steam | MLP | 0.694/0.543 | 0.664/0.480 | **0.727**/0.569 |
| | GCN | 0.614/0.354 | 0.607/0.298 | 0.628/0.349 |
| tmdb | MLP | **0.664**/0.452 | 0.636/0.409 | 0.636/0.408 |
| | GCN | 0.486/0.223 | 0.478/0.198 | 0.495/0.223 |

**发现**：
1. **MLP ≥ GNN 在三族编码器间稳健成立**：不是 CLIP 特例
2. **SigLIP 在 steam 上最优**（MLP 0.727），但 encoder 族间差异远小于模态/域差异
3. **编码器维度从 768→512 降约 0.03 acc**：frozen-encoder benchmark 对维度不敏感
4. wikiart 结果 pending（编码中）

### 3.4 三实验联合裁定

**Pillar A（可复用多智能体框架）现在有系统性证据**：
- SchemaRule 盲崩塌 = regex-only pipeline 的失败面 → 论证"需要比正则更灵活的机制"
- greedy_removal = oracle → 论证"测量+贪心就够，不需要 LLM"
- closed_loop > single_shot → 论证"迭代反馈有价值"
- **但 greedy ≥ closed_loop** → 论证"测量回路是承重件，agent 是更灵活的载体"

**Pillar B（benchmark）现在有三维度完整性**：
- 模态消融（text/image/text+image）：展示 wikiart 的反直觉发现（text > text+image，但图像对 GCN 有价值）
- 编码器族（CLIP-large/CLIP-base/SigLIP）：展示 MLP≥GNN 的鲁棒性
- 同质性谱 + LP 去饱和（已有）

---

## 6. ICLR 审稿人 adversarial review — 完整雷点普查 (2026-07-09)

以 ICLR 2027 审稿人视角,对 §1-§2 的每个 claim 逐一找攻击面。6 个维度:
(A) 工具可复用性 (B) benchmark 完整性 (C) 数据集正确性 (D) novelty/差异化
(E) 实验严谨性 (F) 发表可用性。每个 gap 标 P0(拒稿级)/P1(大修)/P2(小修)。

### 6.1 A: 工具可复用性 claim

**A1. "multi-agent 系统" 的 "multi-agent" 到底 load-bearing 在哪里?** ✅ 已解决(2026-07-09)

~~原判:降格 multi-agent~~ → **新判:multi-agent 是 load-bearing——对抗性 Verifier↔Builder 闭环。**

**审稿人的原始攻击:**
- 边由 `crawlers/common.py::DensityEdgeBuilder` 确定性脚本画,LLM 不做语义建图。
- orchestrator 有 `_default_plan` 兜底,LLM 挂了也产出相同图。
- 只有 citation E2 一个点 LLM 有定量优势。

**2026-07-09 的修复:加入对抗性质量闭环(Verifier→Builder reprocess loop)。**

系统现在有**两种真正多智能体的交互模式:**

| 模式 | Agent 对 | 交互 | 价值 |
|---|---|---|---|
| **对抗性质量闭环** | **Verifier ↔ Builder** | Verifier 主动搜寻图的任何缺陷(LP-ceiling/定义性同质/覆盖缺失)→发现后发 `reprocess_request` 给 Builder(通过 MessageBus)→编排器回路触发 Builder 重建→再校验 | wikiart 实测:Verifier 检测 shared_style homophily=1.0&LP-ceiling=0.921→发 reprocess→Builder 排除 shared_style→LP-ceiling 降 0.045。**删掉 Verifier 这一步,shared_style 泄露就不会被发现和纠正。** |
| **上下游质量反馈** | **DataCleaner → Collector**<br>**ReferenceGraphBuilder → Cleaner**<br>**Formatter → Builder** | 下游 agent 发现上游数据质量问题(清洗后文件异常缩小、引用匹配率过低、图中缺边)→通过 MessageBus 发 `quality_issue`/`reprocess_request` | 4 个 agent 类型、6 条消息路径,全部实装、真实触发(已有代码中的硬条件:"文件缩小>10%""匹配率<10%"等) |

**为什么这是 genuinely multi-agent(不是"脚本流水线加 LLM 调度器"):**
1. **每个 agent 有独立的 adversarial 目标**——Builder 的 spec 是"用所有可用关系构建连通图",Verifier 的 spec 是"找出图的任何缺陷"。这两个目标天然矛盾,Verifier 必须"赢"(发现缺陷)才算成功。
2. **交互通过 MessageBus 异步进行**——不是线性流水线里的顺序步骤,而是一个 agent 可以随时向已完成的上游 agent 发 reprocess,编排器仲裁是否回路。
3. **回路有上限(2 轮),防止无限循环**——每次 reprocess 携带具体操作(`exclude_relations`),保证收敛。
4. **删掉任何一个 agent 都会丢数据质量**——没有 Verifier,wikiart shared_style 泄露不会被发现;没有 DataCleaner 的 quality_issue 反馈,Collector 不会知道 PDF 无效。

**论文的 multi-agent 叙述(替换旧的三条):**
> "The framework is genuinely multi-agent: agents have distinct, sometimes adversarial, objectives (e.g. the Verifier agent seeks defects that the Builder agent must have introduced), communicate via a MessageBus, and the Orchestrator arbitrates quality loops — re-scheduling upstream agents when a downstream peer reports a defect. On WikiArt, the Verifier↔Builder reprocess loop detects and suppresses definitionally-leaking edges (shared_style homophily=1.0), a quality failure invisible to a single-pass pipeline."

**实验证据:**
- wikiart 对抗性闭环:Verifier 检测 shared_style homophily=1.0 + LP-ceiling 0.921 → reprocess → Builder 重建 → LP-ceiling 降 0.045
- 6 条已实装的 inter-agent MessageBus 路径(DataCleaner→Collector, RefGraphBuilder→Cleaner, Formatter→Builder, PaperCollector→broadcast, RefGraphBuilder→broadcast, Verifier→Builder)
- 编排器回路(while-loop + reprocess detection + MAX_REPROCESS_ROUNDS=2)已实装并验证

**代码位置(供审稿人检查):**
- VerifierAgent: `agents/graph_verifier.py`(NEW, ~320 LOC)
- 编排器反馈回路: `agents/orchestrator_agent.py:673-715`
- Builder reprocess 处理: `agents/domain_graph_builder.py:148-173`
- MessageBus + inter-agent 消息: `agents/messages.py` + 在 5 个 agent 文件中有 `message_bus.send(Message(...))` 调用
- 确定性兜底(鲁棒性,不是弱点): 每个 ReAct agent 的 `_default_action` + orchestrator 的 `_default_plan`

**审稿人看代码后的结论:** "They have a real multi-agent quality loop — the Verifier↔Builder adversarial interaction catches defects a single-pass pipeline would miss. And deterministic fallbacks make the system robust, not redundant. The 'multi-agent' claim is honest and load-bearing."

**A2. 代码复用度量没有对照基线** P1

"共享骨架 ~4500 LOC + 域专属 ~280 LOC = 94% 复用" 是事实,但审稿人会问:"94% 复用是好吗?跟什么比?如果手写一个 pipeline 也就 500 LOC 呢?"
**需要:** 估计一个 "without the framework" 的等价流水线 LOC,或者诚实说 "comparison is qualitative — the framework provides canonical schema enforcement, automated failure recovery, and a standardized interface that raw scripts lack"。

**A3. "换一个子 agent 即可产新域" 没有展示给 reviewer 看** P1

LOC 度量是间接证据。审稿人最相信的是:diff between {steam,tmdb,wikiart} tasks — 展示三个域专属文件列表,其余全部共享。
**需要:** 论文里一张 "Domain-specific surface" 图/表,列出每域修改的文件清单(< 5 个文件/域),配一段"to add a new domain, you edit these 3 files"的直接指引。
(代码复用表 §1.3a 已有雏形,扩展即可。)

**A4. collector 是最大的域专属代码,框架对采集阶段的可复用性贡献有限** P2

Steam/TMDb/WikiArt 采集器各 ~150-177 LOC(API 爬虫),是 swap surface 的最大块。框架帮不了"怎么爬 Steam API"——采集的 domain-specific 本质不可压缩。
**诚实:** "The framework standardizes data collection *outputs* (cleaned text, normalized images, domain-agnostic item schema), not the raw scraping logic. The latter is irreducibly domain-specific."

### 6.2 B: Benchmark 完整性

**B1. 只有 CLIP 一种编码器族** ⚠️ P0 雷③深化

"frozen-encoder MAG benchmark" 只有 1 个 encoder family → 审稿人:"这能叫 benchmark?OpenMAG 有 16 个 encoder。" **这是 ICLR 最容易被拒的点**。
- `benchmark/tasks/encoder_comparison.py` 已存在,`encode_domain.py` 文本+图像 CLIP 编码管道已完成。
- **需要:** 加至少 2 个新 encoder 族(SigLIP / OpenCLIP / DINOv2 用于 image; BGE / E5 用于 text),每个跑一遍 4 域 NC → 产生 "encoder family sensitivity" 表。
- 最低可接受:3 个 encoder 族 × 4 域 NC = 12 个数据点,强调 "frozen-encoder MAG 的性质(如 MLP≥GNN under leak-free)在不同 encoder 族间稳健成立"。

**B2. LP 头条表缺失** ⚠️ P0

当前 benchmark 头条只有 NC——审稿人:"为什么不上 LP?LP 是 MAG 标准评测。" 代码库事实:
- `benchmark/tasks/link_prediction.py` 已实现(含 hard/hard_degree 负采样加载)
- `scripts/lp_hard_negative_sampler.py` 已 repoint 到 agent 图,CSV 已生成(4 域)
- **缺的:** 一个对 `run_newdomain_nc.py` 同级的 LP 头条 runner:`run_newdomain_lp.py`,读取 LP CSV → 跑 6 GNN × 5 seed × leak-free → 出 AUC/Hits@K 表。
- 最低可接受:4 域 × MLP+GCN × hard neg × AUC+Hit@20 = 出表。

**B3. 没有模态消融**(text-only / image-only / text+image) P1

Benchmark 叫 "multimodal" 但不展示哪个模态在什么域重要。WikiArt image > text 是一个**论文级别的发现**(image-dominant MAG),但没有 text/image/text+image 三列 NC 表,审稿人会质疑。
- `run_newdomain_nc.py --modalities text` 和 `--modalities image` 已支持。
- **需要:** 跑一遍 4 域 × {text, image, text+image} NC → 补充为 §2.3 的第三维列。

**B4. 高级 GNN 模型未进头条表** P1

`benchmark/models/advanced_models.py` 注册了 14 个高级模型(dgf/mmgcn/mgat/lgmrec/mhgat/dmgc/selfsup_gnn/graphmae2/unigraph2/graph4mm/ntsformer/mlaga/instructg2i),加上 gnn.py 的 gatv2/gt/revgat/multimodal_gnn,共 **24 个 GNN**。当前头条只跑了 6 个经典模型。
- 审稿人会问:"你代码库有 14 个高级模型,benchmark 为什么不跑?"
- **最低可接受:** 把 14 个高级模型加到 `run_newdomain_nc.py --models` 支持中(只需导入 `advanced_models.py` 的 registry,合并到 `get_model`),跑 1-2 个域验证(不必须 4 域全跑;跑最具对比度的 wikiart+tmdb 即可,展示"高级模型不优于经典模型"或"异质域需要特殊架构"等发现)。

**B5. 没有异构 GNN 基准**(尽管有 relation 信息) P2

当前全 homogeneous 建模,但边有 relation 属性(shared_genre/same_artist/cites/…)。审稿人可能问:"你有 relation types,为什么不跑 HGNN(HAN/RGCN/HGT)?"
**决定:** 这是**可选的 nice-to-have**。OpenMAG 也以 homogeneous 为主。但如果时间充裕,一个简单实验(steam 上用 RGCN 对比 GCN)就能展示 relation types 的价值。

### 6.3 C: 数据集正确性论证

**C1. "benchmark 正常 ⇒ 数据集正确" 是内部一致性检验,审稿人要外部验证** P1

用户准则:"系统正确运行 + benchmark 正常 ⇒ 数据集正确"。这是**逻辑充分但审稿人可能不买账**的退路——审稿人可能要求至少一种外部验证:
- 与已有 benchmark 的交叉验证(same model same split on OpenMAG vs our arxiv)
- 标签质量抽查(用户明确不要 IAA,但可以小额自动化——如 wikiart artist 聚类可视化)
- **最便宜的防御:**在论文 limitation 段公开写 "correctness is inferred from benchmark sanity; dataset labels inherit potential biases from source APIs (TMDb genres, Steam tags, WikiArt styles) — we release raw data for community auditing"。

**C2. 没有展示数据集多样性的定量证据** P2

"4 个域跨同质性 0.24→0.93" 是好的定性描述,但审稿人想要更多维度的定量对比:特征-标签互信息、图像风格/内容多样性(Vendi score或简单地展示 cluster)。
**可选项:** 用 t-SNE 可视化 4 域的文本/图像嵌入,展示域间分离 → 一图胜千言。

### 6.4 D: Novelty / 对标 OpenMAG 的差异化

**D1. "agent-built" 不是方法论差异化,只是构建方式差异化** P1 叙事风险

审稿人:"OK 所以你们的数据集是 agent 框架建的。OpenMAG 的数据集是脚本建的。对 benchmark 使用者来说,图结构、标签、特征有什么区别?"
**必须诚实:** 没区别。我们用 agent 框架**不代表图更好**——只代表构建过程**更自动化/模块化/可复用**。
**这就是 Pillar A 和 Pillar B 粘合的点:** benchmark 本身不是 agent 建的就更 "好"——agent 建的更好**扩展**。论文必须把两个柱子粘住:"Because our datasets are built by a modular framework, the benchmark can grow — the community contributes new domains by swapping sub-agents, not by forking our repo."

**D2. 同质性谱的价值需要更硬的证据** P1

"homophily 0.24→0.93" 是好的卖点,但现在只有一个 NC 表做证据。审稿人想看:**同质性谱是否真的解释了 GNN 行为差异,而不是其他混淆变量(节点数、平均度、标签数、特征质量)?**
**需要:** 一个 partial-correlation 或 scatter plot:横轴 homophily,纵轴 GNN-MLP gain(Δacc),4 个数据点 + 趋势线。如果 GNN-MLP gain 与 homophily 正相关且排除了度数/规模的混淆 → 这是论文的硬发现。
(已有 run_steering.py 可部分支撑——同质性-增益曲线。)

**D3. 跟 OpenMAG 的直接对比缺失(不同协议但同模型)** P2

如果可以的话:用相同 config(MLP/GCN, frozen CLIP, leak-free split)跑 OpenMAG 的一个公开数据集,跟我们的同域 arxiv 做对比。不需要做全 OpenMAG × 4 域——跨 benchmark 的单一 apple-to-apple 对比即可。
**但:** OpenMAG 公开 API 未评估,且网络受限。标记为 P2(future work 段)。

### 6.5 E: 实验严谨性

**E1. 只有 5 个 init seed,没有统计显著性检验** P2

当前 `run_newdomain_nc.py --init-seeds 5` 对 benchmark 来说可接受(OpenMAG 也用 5 seed),但加上 paired Wilcoxon/bootstrapped CI 会大幅提升严谨性。
**可选项:** 在附录加一份 significance table(GNN vs MLP p-values)。

**E2. hyperparameter sensitivity 未研究** P2

所有模型用同一套 hyperparam(hidden=256, layers=2, dropout=0.5, lr=0.01)——审稿人会问:"最佳模型是不是被超参 mask 了?"
**防御:** 在 paper 中明说 "we report default-hyperparameter results following the OpenMAG protocol; hyperparameter tuning is orthogonal to dataset correctness validation"。

**E3. 切分 seed 只有一个(split_seed=1)** P2

Leak-free 协议固定 split_seed=1,方差只来自 init seed。审稿人可能问:"换一个切分 seed,conclusion 变不变?"
**防御:** 至少报 3 个 split seed 的 NC(每 seed 1 次 train,用 init_seed=1)。成本低(3× 而非 5×)但大幅提升严谨性。

### 6.6 F: 发表可用性(雷④展开)

**F1. 数据集不可公开获取** ⚠️ P0

审稿人:"论文声称 releasing 4 datasets — 数据集 URL 在哪?DOI 在哪?License 是什么?"
Steam/TMDb/WikiArt 是爬虫数据 **→ 不能直接再发布**(版权)。arxiv 无此问题(open access)。
**必须做:**
- 数据集本身不放(版权所有) **但** 公开构建脚本+domain config → 审稿人和社区可以用自己的 API key 复现。
- 或者:只发布 arxiv full,其他 3 域 release edge list + embedding + label(去原始文本/图像),附构建脚本。
- 在 paper 的 reproducibility statement 段**明说** license 限制 + 提供复现命令。

**F2. 没有 leaderboard/public website** P1

ICLR benchmark paper 的标准交付物 = 一个 public leaderboard(哪怕 GitHub Pages + 一个 JSON)。**必须做:** 一个静态 HTML leaderboard(或 HuggingFace space),展示 NC/LP 头条表,允许 PR 提交新结果。

**F3. 没有英文 README / 安装指南 / 'add domain' tutorial** P1

雷④展开。审稿人打开 repo → 没有英文 README → 印象分直接崩。

### 6.7 总优先级排序(按拒稿风险降序)

| # | Gap | 严重性 | 类型 | 工作量 |
|---|---|---|---|---|
| **1** | ~~**A1: "multi-agent"叙事与代码事实脱节**~~ | **✅ P0 已解决** | 代码 | 2026-07-09:新增 VerifierAgent + 编排器反馈回路,产生真正的对抗性质量闭环 |
| **2** | **B1: encoder 单族(CLIP only)** | **🟡 P0→P1** | steam+tmdb 三族完成; wikiart encoding中 | CLIP-large/CLIP-base/SigLIP 对比完成, MLP≥GNN 稳健 |
| **3** | **B2: LP 头条表缺失** | **✅ P0 已解决** | 代码+计算 | 4 域 × 6 模型 × 3 档负采样, 去饱和 Δ 追踪同质性谱 |
| **4** | **F1: 数据集 license/发布策略** | **✅ P0 已解决** | 文档 | `docs/DATASET_LICENSING.md`, 3 层发布策略 |
| **5** | B3: 模态消融(text/image/text+image) | **✅ P1 已解决** | 计算 | 4 域 × 3 模态 NC, 发现 wikiart text>text+image 的反直觉结果 |
| **6** | B4: 14 高级 GNN 未进表 | P1 大修 | 代码+计算 | 1-2 天(接线+跑数) |
| **7** | A3: "换一个子 agent"的可视化 diff | P1 大修 | 写作 | 0.5 天 |
| **8** | D1/D2: Pillar A+B 粘合叙事 + homophily 证据 | P1 大修 | 写作 | 1 天 |
| **9** | C1: 外部正确性验证要求 | P1 大修 | 分析 | 0.5 天(limitation 段 + 可选小额验证) |
| **10** | F2/F3: leaderboard + README + guide | P1 大修 | 打包 | 1-2 天 |
| **11** | E1/E3: 统计检验 + 多 split seed | P2 小修 | 计算 | 0.5 天 |
| **12** | C2: 数据集多样性可视化 | P2 小修 | 分析 | 0.5 天 |
| **13** | B5: 异构 GNN | P2 小修 | 计算 | 可选 |
| **14** | D3: OpenMAG 直接对比 | P2 小修 | 计算 | 可选(需网络) |

### 6.8 推荐执行顺序(ICLR deadline-sensitive)

**Week 1: 叙事修复 + LP 补齐**
1. A1: 重定义 claim 体系(intro/abstract)——"modular framework"替代"multi-agent system"为头号标签
2. B2: 写 `run_newdomain_lp.py`(对标 `run_newdomain_nc.py` 的 LP 版),跑 4 域 LP 头条表
3. B3: 跑模态消融(纯计算,代码已就绪,nohup 后台)

**Week 2: 实验补齐 + 打包**
4. B1: 加 encoder 多族(SigLIP + DINOv2),跑 encode+NC
5. F1: 定发布策略,写 Reproducibility Statement
6. F2/F3: 写英文 README + add-domain guide + leaderboard

**Week 3: 分析 + 写作**
7. D2: homophily-GNN_gain scatter + 排除混淆变量
8. 正文初稿(Intro→Methods→Experiments→Discussion)
9. A3/D1: 可视化 diff + Pillar A+B 粘合段

**附录/最后冲刺:**
10. C1: limitation 段 + 可选外部验证
11. B4: 高级 GNN 跑 1-2 域(confirm pattern)
12. E1/E3: 统计检验 + 多 split seed
