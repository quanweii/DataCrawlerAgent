# IGC Agent-Necessity Results (2026-07-03)

> 配套 `benchmark/results/igc/igc_all.json`（3 域 × 4 spec × 5 constructor × 多 seed）。
> 实验：`benchmark/run_igc_agent_necessity.py`。模型 `mimo-v2.5`。设计见 `/home/dev_qw/.claude/plans/foamy-wiggling-owl.md`。
> **目的**：回答"为何必须多智能体"——闭环(planner+critic, executor=测量工具)能否在**盲重命名**下抑制结构泄露,而静态规则/单次 LLM 不能。

## ⚠️ 修订(2026-07-03 decisive 消融)— 原"闭环多智能体独赢"的结论**被推翻**

加了一个确定性贪心 `greedy_removal`(给脚本管线**同一个 `measure_subset` 工具**、但**不给 LLM**):每轮移除使 spec 目标最优的关系,blind-agnostic(根本不看关系名/`bmap`,只看测量值)。这是区分"测量有用"还是"多智能体有用"的唯一判据。

### Decisive 表:S1b(盲)LP-ceiling — greedy vs closed_loop vs oracle(lower=better)

| domain (majority) | all | schema/naive | **greedy(无LLM)** | **closed_loop(LLM)** | oracle |
|---|---|---|---|---|---|
| steam (0.504) | 0.557 | 0.557 | **0.557** | 0.557±0.000 | 0.489 |
| **tmdb (0.205)** | 0.413 | 0.413 | **0.255** | 0.315±0.060 | 0.255 |
| wikiart (0.025) | 0.921 | 0.921 | **0.884** | 0.884±0.000 | 0.884 |

**判决:`greedy_removal ≥ closed_loop` 在全部 3 域成立(greedy 更好或持平),tmdb 上贪心还严格更好(0.255 < 0.315±0.060,差一整个 std,且 = oracle)。多智能体在【任一域】都没有胜过同工具的确定性贪心。**

为什么 tmdb 上 agent 反而更差:closed_loop 的 critic 只读 **solo per-rel LP-ceiling**(见 `_critic_user`),再由 LLM lossy 解释;而贪心直接测量**联合子集** LP(试删每个关系、测真实结果)。LLM 是把测量信息过一个**有损的 solo-LP 代理 + LLM 解读**瓶颈,而非优势。→ **agent 是测量的(有损)载体,不是测量的来源。**

steam 上 greedy=closed_loop=0.557 都没到 oracle 0.489(贪心局部最优陷阱:从 4 关系到单 `shared_category` 需经过 LP 非单调下降的中间子集)。这反而支撑**两层防线**叙事:构造期选择(greedy/closed_loop)去掉能去的,残留的由评估期跨切分剪枝兜底。

### 对论文的影响(path B 在小 regime 死,确认 Option 1)

"多智能体必要"在 ≤4 关系的小题上**不成立**——确定性贪心用同一个测量工具全域追平或胜出。论文主线改为 **Option 1**:核心贡献 = **LP-ceiling 泄露诊断 + leak-free 协议 + feature-saturation 公式**(方法论),agent 降为"我们如何构建数据"的一个(有损)实例化。下文原"头条表 1"把 closed_loop 当赢家的框架**作废**,保留作支持数据(下游 NC-F1、命名 vs 盲控制仍有效,只是不再支撑"agent 必要")。

---

## ★ 头条表 1:S1b (盲重命名) LP-ceiling — lower=better, 括号为多数类基线

| domain (majority) | all-relations | schema_rule | naive_keyword | single-shot LLM | **closed-loop** | oracle |
|---|---|---|---|---|---|---|
| steam (0.504) | 0.557 | 0.557 | 0.557 | 0.557±0.000 | 0.557±0.000 (no-op) | 0.489 |
| **tmdb (0.205)** | 0.413 | 0.413 | 0.413 | 0.413±0.000 | **0.315±0.060** | 0.255 |
| **wikiart (0.025)** | 0.921 | 0.921 | 0.921 | 0.921±0.000 | **0.884±0.000 (=floor)** | 0.884 |

**读法（三域三种情况，干净且诚实）：**
- **可压域 (tmdb)**：盲重命名下，schema_rule / naive / single-shot **全部卡在 0.413**（认不出泄露关系）；闭环把它们都不认识的泄露关系靠**测量**找出来，压到 **0.315**（趋向 oracle 0.255）。
- **结构性域 (wikiart)**：两条关系都泄露（same_artist 0.884 / shared_style 0.917，label≡关系基底），关系选择无法消除。闭环**收敛到可达下限 0.884 = oracle**，而三个基线连"换一条稍好的关系"都没做（卡 0.921）。→ 闭环给出正确诊断"结构性不可消除"，静态规则会误报"已处理"。
- **无事可做域 (steam)**：泄露本就低于阈值(0.557≈多数类 0.504)，闭环**正确地什么都不做**（no-op），与基线一致。

## ★ 头条表 2：去泄露对下游 GCN 的收益（leak-free NC, 5-seed）

构造期去泄露（用闭环选的图 vs 全关系图）对 GCN macro-F1 的影响（MLP 不用边→应不变，作对照）：

| domain | GCN all→closed-loop | MLP all→closed-loop |
|---|---|---|
| steam | 0.314 → 0.314 (+0.000, no-op) | 0.510 → 0.510 |
| **tmdb** | 0.231 → **0.258 (+0.027)** | 0.473 → 0.473 |
| **wikiart** | 0.856 → **0.883 (+0.027)** | 0.864 → 0.864 |

→ **构造期去泄露真实有益**：去掉标签捷径边后 GCN 被迫用特征，泛化更好（+0.027）；MLP 恒定证明收益来自图结构而非特征。这是闭环构造的下游处方依据。

## 支持表：S2 (max-homophily 目标) 闭环同质性 vs 基线

| domain | base(all) | closed-loop | oracle |
|---|---|---|---|
| steam | 0.529 | **0.684** | 0.817 |
| tmdb | 0.237 | 0.260 | 0.329 |
| wikiart | 0.894 | 0.894 (已高) | 1.000 |

→ 闭环在第二个目标(高同质)上也朝 oracle 移动（steam 0.529→0.684），single-shot/schema_rule 都停在基线。证明闭环不是只针对泄露的过拟合,而是通用的"测量→修订"构造能力。

## 命名 vs 盲 (S1a vs S1b) — 控制实验

| domain | S1a(命名) closed | S1b(盲) closed | 解释 |
|---|---|---|---|
| tmdb | 0.255 (=oracle) | 0.315 | 命名时 LLM 可读 same_director→直接命中；盲时需测量,接近但非最优 |
| wikiart | 0.917 | 0.884 | 命名时丢 same_artist 留 shared_style(0.917);盲时测量发现 same_artist(0.884)更低→更准 |
| steam | 0.557 | 0.557 | 无泄露可压 |

→ **盲重命名是关键控制**：命名时连单次 LLM/正则都能靠名字走运；盲时只有**测量**的闭环撑得住。wikiart 盲反而比命名更准(0.884<0.917),因为测量比"name 猜"更可靠。

## 一句话贡献(论文 intro — Option 1 修订)

> **LP-ceiling 作为 MAG 的标签泄露诊断**:在关系名匿名的盲测下,**测量驱动**的关系选择(确定性贪心或闭环多智能体同效)能把可压域的结构泄露从 0.413 压到 oracle 0.255,并**正确诊断**结构性不可消除域(wikiart 下限 0.884)——而纯静态规则/单次 LLM 因拿不到测量信号而全失败(卡 0.413/0.921)。关键诚实结论:**多智能体与同工具的确定性贪心在该任务上等价**(agent 是测量的载体,非来源),因此论文的贡献是**诊断+协议**(LP-ceiling + leak-free),而非 agent 本身;agent 是"如何构建数据"的实例化。残留泄露由评估期跨切分剪枝兜底,且构造期去泄露让下游 GCN +0.027 macro-F1(MLP 恒定)。

## 仍可加强(camera-ready / 决定性后续)

**若要保留任何"agent 必要"主张(Option 2,非 Option 1 默认)**:唯一干净路径是 **scale 实验**——关系数 k 从 2 涨到 ~15-20(合成多关系图或重爬),展示确定性贪心/暴搜 `2^k` 随 k 指数爆炸、而闭环的 LLM 测量引导搜索保持 tractable 且近 oracle。**当前 4 域关系数 ≤4,天然在"agent 最不必要"的 regime**,小题上 agent 必要性已被本消融证伪。

**Option 1 默认下的 camera-ready**:
- 收敛曲线图(LP-ceiling vs critic-round)作为机制图——仍能展示闭环 vs 单次的测量-修订过程。
- 下游 NC-F1 补 n=5 seed + paired Wilcoxon(现 +0.027 无显著性检验)。
- 命名 vs 盲控制表(仍有效,证明"测量比 name 猜更可靠")。
