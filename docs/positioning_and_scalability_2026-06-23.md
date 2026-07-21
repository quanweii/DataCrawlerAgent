# 定位对标 + 多智能体 Scalability（T2⑩ + T2⑪）

> 2026-06-23 ｜ 配套 `benchmark/results/aggregated_report.md`
> 目的：① 一张表说清「vs OpenMAG 到底新在哪」；② 给 C2「可扩展构建」配上真实度量。

---

## T2⑩ 与 OpenMAG 的对标表

OpenMAG（ICML 2026 poster）= 现有最全 MAG 基准：19 数据集 / 24 模型 / 16 编码器 / 8 任务。
**数量上我们赢不了，赢在「深 + 正交 + 可控」**。

| 维度 | OpenMAG | AMMAG (本文) | 谁强 / 说明 |
|---|---|---|---|
| 数据集数 | 19 | 4 | OpenMAG（量） |
| 域正交性 | 9/19 短视频推荐（高集中）+ 6 电商 | 游戏/电影/画作/论文（真正异构） | **本文** |
| **同质性可控** | ❌ 无（仅 DMGC 一个下游模型） | ✅ E1 旋钮 + tmdb0.24→arxiv0.93 全谱 | **本文（头号差异）** |
| 公式(formula)模态 | ❌ 全部 Text+Visual | ✅ arxiv 独有 formula | **本文（唯一）** |
| 跨域迁移任务 | ❌ 无 | ✅ 4×4 迁移矩阵（负结果） | **本文** |
| 自然边 vs 构造边 | 部分自然(co-purchase) | arxiv 自然引用 + 3 域构造 | 互补 |
| 负采样协议 | 随机为主 | random/hard/hard_degree 三档去饱和 | **本文** |
| 泄露控制 | 未披露 | LP-ceiling 上限 + 跨切分边剪枝(E3/T0②) | **本文** |
| 可复现/可审计构建 | ❌ 依赖不透明平台数据 | ✅ agent 构建 + 固定切分 + manifest | **本文** |
| 统计严谨 | 自陈「不用固定种子、只报 mean」 | 5-seed mean±std + 显著性 | **本文（能反超）** |
| 编码器数 | 16（含 DINOv2/Qwen/ImageBind） | 1（CLIP-L/14 768d） | OpenMAG |
| baseline 模型数 | 24（含 15 MAG 模型） | 6 vanilla GNN + 融合消融 + MultimodalGNN | OpenMAG |
| 任务数 | 8（NC/LP/聚类/检索/匹配/G2Text/G2Image） | NC + LP + 边关系预测(3) | OpenMAG |
| 生成任务(G2T/G2I) | ✅ 有 | ❌ 代码就绪未训（已从主张中剔除，见 T3⑭） | OpenMAG |

### 两个杀手级 positioning 钩子（写进 Related Work）

1. **OpenMAG 自己的洞察 C4**："excessive neighbors introduce semantic noise / over-dense neighborhoods dilute distinct node features" —— 他们**观察到**邻居密度敏感，却**没有机制去控制**。本文 E1 steering + Q2 邻居扫描**直接提供这个旋钮**。
   > 一句话："They observe neighbor-density sensitivity; **we provide the knob.**"

2. **可复现性护城河**：OpenMAG 的图依赖不透明平台数据采集（哪两个视频被共看、哪两个商品被共买）——不可复现、不可审计、不可 steer。本文 agent 构建的图全可复现（固定切分 + manifest + 确定性规则建图）。

### 诚实短板（必须在论文里承认）
- baseline 数量与编码器多样性不及 OpenMAG → 定位为「少而深、可控、正交」的**互补型**基准，不声称取代。
- 无生成任务 → 明确从贡献剔除（见 T3⑭）。

---

## T2⑪ 多智能体 Scalability 度量表（C2 的实测支撑）

C2 主张 = "1 框架 → 4 异构域 = 可扩展构建"。度量（从构建日志 + manifest 实测）：

| 域 | 节点 | 唯一边 | 模态 | 构建墙钟 | agent 调用 | 边/秒 |
|---|---|---|---|---|---|---|
| arxiv | 45,672 | 57,045 | text+image+**formula** | **6.3 min** | 5 agents / 10 ReAct 迭代 | 151 |
| steam | 2,760 | 54,424 | text+image | 94.0 min | 确定性规则建图 | 10 |
| tmdb | 13,037 | 193,346 | text+image | 328.6 min | 确定性规则建图 | 10 |
| wikiart | 21,968 | 520,570 | text+image | 21.4 min | 确定性规则建图 | 405 |

### 关键发现（data-systems 层面，可写进论文）
- **构建耗时被数据采集（API/图像爬取）主导，而非 LLM/agent 开销**：arxiv 纯 agent 编排（数据已就位）仅 6.3 min；steam/tmdb 含图像下载达 94–329 min。→ 框架的「可扩展性」瓶颈在 I/O 与外部 API 限流，**不在多智能体调度本身**。
- **建图规则确定性、零 LLM 参与画边**（已在代码侦察中核实）：边由 `DensityEdgeBuilder`（combinations + cap 采样 + freq 过滤）一次性生成。→ 这正是「可复现/可审计」的来源，也是 E1「relation 是可控自变量」的实现基础（加载时筛选）。
- **同一套 ReAct+blackboard 框架覆盖 4 个异构域**：加新域只需写一个 Collector + GraphBuilder（共享调度/重试/校验/落盘）。这是 scalability 的真实含义——不是单域吞吐，而是**跨域复用**。

### 诚实标注
- steam/tmdb/wikiart 的 rebuild 日志是「原始爬虫」阶段，未记录 ReAct 迭代（因该阶段是确定性脚本，不经 agent）；arxiv 日志含完整 agent 编排。三者不可严格同口径比较 agent 开销，论文应分开报「agent 编排成本」vs「数据采集成本」。

### 数据来源
- 构建：`logs/data_crawler_2026-06-12_*.log`（arxiv）、`logs/rebuild_{steam,tmdb,wikiart_v2}.log`
- 图统计：`benchmark/results/dataset_card/manifest.json`
- 代码侦察：`agents/*graph_builder.py`、`crawlers/common.py:DensityEdgeBuilder`
