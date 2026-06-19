# 多模态属性图数据集扩充：三个新领域

> 汇报人：[姓名] ｜ 日期：2026年6月

---

## 第1页 · 研究背景与动机

### 现有 MAG Benchmark 的领域局限

当前主流多模态属性图（Multimodal Attribute Graph, MAG）benchmark 存在**领域覆盖狭窄**的问题：

| Benchmark | 覆盖领域 |
|-----------|----------|
| MAGB (KDD'25) | Amazon商品、Goodreads图书、Reddit、Yelp |
| MM-GRAPH (CVPR'25) | Goodreads、Amazon运动品 |
| OpenMAG (ICML'26) | Reddit、SemArt艺术、Flickr30k、Bilibili短视频、Goodreads |

**空白领域**：🎮 游戏平台、🎨 经典艺术画作、🎬 电影元数据 — 三者均为**零覆盖**。

### 扩充目标

引入三个全新领域，验证本文**多智能体系统**的领域泛化能力：

| 领域 | 数据源 | 爬取方式 | 多智能体价值 |
|------|--------|----------|--------------|
| 游戏网络 | Steam Store API | REST API 动态发现+爬取 | 展示 API 编排能力 |
| 艺术画作 | WikiArt.org | 网站主动爬取+图片下载 | 展示网页解析能力 |
| 电影网络 | TMDb API | API 爬取+演职关系抽取 | 展示关系推理能力 |

> **核心论点**：三个领域采用三种不同的爬取策略（REST API / 网页爬取 / 关系抽取），证明多智能体系统能适应异构数据源。

---

## 第2页 · 三数据集概览对比

| 维度 | Steam 游戏网络 | WikiArt 艺术网络 | TMDb 电影网络 |
|------|---------------|-----------------|--------------|
| **数据源** | store.steampowered.com | wikiart.org | api.themoviedb.org |
| **爬取方式** | Store搜索分页 + appdetails API | PaintingsByArtist API + 字母索引 | discover API + 详情API |
| **节点类型** | 游戏 (game) | 画作 (artwork) | 电影 (movie) |
| **节点数** | **1,842** | **~25,034** | **13,054** |
| **文本模态** | 标题+开发者+流派+描述 (~1,698字符/节点) | 标题+艺术家+年份 | 标题+导演+演员+剧情简介 |
| **图像模态** | 游戏截图 (3张/游戏) | 画作原图 (1张/画) | 电影海报 (1张/电影) |
| **标签** | 14个流派 (Action/RPG/...) | 25+艺术风格 (印象派/...) | 19个电影类型 (剧情/喜剧/...) |
| **边类型** | shared_genre / same_developer / same_publisher | same_artist / shared_style / shared_genre | shared_genre / same_director / shared_actor |
| **图结构依据** | 共享元数据属性 | 艺术家/风格归属 | 演职人员共享 |
| **现有覆盖** | ❌ 零 | ❌ 零 | ❌ 零 |

### 三领域的差异化设计意图

```
Steam  → 验证「商业产品图」构建（元数据驱动，属性共享连边）
WikiArt → 验证「视觉艺术图」构建（创作者归属连边，图分类任务）
TMDb   → 验证「社会关系图」构建（人物关系连边，多人共享）
```

---

## 第3页 · 多智能体构建流程

### 统一的领域管线架构

每个新领域遵循相同的四阶段管线，由 `OrchestratorAgent` 动态调度：

```
┌─────────────────────────────────────────────────────────┐
│            OrchestratorAgent (LLM 规划调度)              │
│   analyze_task → decide_next_agent → evaluate_result    │
└──────┬──────────┬──────────────┬──────────────┬────────┘
       ▼          ▼              ▼              ▼
 ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
 │Collector │→│ Cleaner  │→│GraphBuild│→│ Formatter│
 │  Agent   │ │  Agent   │ │  Agent   │ │  Agent   │
 └──────────┘ └──────────┘ └──────────┘ └──────────┘
  发现+爬取     清洗+去重    关系推理+连边   格式化输出
  ReAct循环     ReAct循环    ReAct循环      ReAct循环
```

### 三领域的爬取策略差异（体现智能体适应性）

**① Steam — API 编排型**
- **发现阶段**：Agent 调用 Store 搜索分页 API（10种排序×100页），动态发现 3,027 个候选 App ID
- **采集阶段**：对每个 App ID 调用 `appdetails` API，Agent 判断 `type=="game"` 才保留
- **智能体决策点**：遇到限速（连续失败50次）→ 自动暂停60秒 → 恢复

**② WikiArt — 网页解析型**
- **发现阶段**：Agent 遍历 26 个字母索引页，解析 HTML 提取 3,579 个艺术家 URL
- **采集阶段**：对每个艺术家调用 PaintingsByArtist API，批量获取画作列表
- **图片修复**：Agent 处理 `!Large.jpg` 后缀异常，调试 CDN 图片 URL 格式

**③ TMDb — 关系抽取型**
- **发现阶段**：Agent 用 discover API（3种排序×500页）获取 15,574 个电影 ID
- **采集阶段**：每个电影调用**两次** API（详情 + 演职人员 credits）
- **关系抽取**：Agent 从 credits 中抽取导演（job==Director）和前10演员，构建人物关系网

### Blackboard 通信机制

三领域共享的 `Blackboard` 命名空间隔离：
- `collector/` — 原始爬取数据
- `cleaner/` — 清洗后数据
- `graph/` — 图节点+边
- `formatter/` — 最终数据集

---

## 第4页 · 数据统计与质量分析

### 当前构建状态

| 指标 | Steam ✅ | WikiArt 🔄 | TMDb 🔄 |
|------|---------|-----------|---------|
| 节点 | 1,842 | 25,034 | 13,054 |
| 边 | 665,883 | 待构建 | 待构建 |
| 图片 | 5,526 | 10,166(下载中) | 1,200(下载中) |
| 文本总量 | 3.1M字符 | — | — |
| 爬取状态 | **完成** | 原始数据完成 | 爬取完成 |

### 各领域标签分布

**Steam 流派分布**（14类，存在长尾）：
```
Action       ████████████████████ 880 (47.8%)
Adventure    ███████ 316 (17.2%)
Casual       ████ 174 (9.4%)
Indie        ███ 146 (7.9%)
Simulation   ███ 132 (7.2%)
RPG          ██ 92 (5.0%)
Racing/Strategy ▌ 各46
```

**TMDb 流派分布**（19类，分布更均衡）：
```
Drama    ████████████████████ 5,737
Comedy   ████████████████ 4,612
Action   ████████████ 3,362
Thriller ████████████ 3,259
Romance  ████████ 2,379
```

### 质量问题与优化方向

| 问题 | 影响 | 优化方案 |
|------|------|----------|
| ⚠️ Steam 边过密（平均度723） | shared_genre 占99.4%，图近乎全连接 | 流派采样上限 500→50；仅稀有流派全连接；引入 shared_category |
| ⚠️ WikiArt 缺 style/genre | 标签字段在 API 中为空 | 调用 painting 详情页补充风格/题材元数据 |
| ✅ TMDb 关系丰富 | 6,291导演 + 53,500演员 | 人员共享边天然稀疏，质量好 |

### 与现有 Benchmark 对比

| 数据集 | 节点 | 边 | 多模态 | 领域新颖度 |
|--------|------|----|--------|-----------| 
| MAGB-Goodreads | 685K | — | text+image | ❌ 已覆盖 |
| OpenMAG-Reddit | 10K | — | text+image | ❌ 已覆盖 |
| **本文-Steam** | **1.8K** | **665K** | **text+image** | **✅ 全新** |
| **本文-WikiArt** | **25K** | **待构建** | **text+image** | **✅ 全新** |
| **本文-TMDb** | **13K** | **待构建** | **text+image** | **✅ 全新** |

### 下一步工作

1. **图优化**：Steam 重建图降低边密度至平均度30-50；WikiArt/TMDb 用更合理采样
2. **元数据补充**：WikiArt 风格/题材字段补全
3. **Agent 集成**：将三个独立脚本封装为 `ReActAgent` 子类，注册到 `DOMAIN_CONFIGS`
4. **Benchmark 注册**：接入 `dataset.py` / `preprocessor.py` / `default.yaml`
5. **基线实验**：GCN/GAT/GraphSAGE/GIN 在三数据集上的节点分类+链路预测
