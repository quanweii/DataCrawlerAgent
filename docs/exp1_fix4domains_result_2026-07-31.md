# Exp1 — 修 4 金标准域（relabel 实验）结果（2026-07-31）

> 目标：把 4 金标准域修到【正确 + 高质量（非饱和 + 图有用）】。
> 结论：**正确性 4/4 全过**（无泄露，报告卡 + lp_ceiling 验证）；**高质量项 relabel 修不动 3/4——arxiv feature-saturated、tmdb 天生异质、steam co-attribute 限，均内禀**。
> → 支撑 thesis 改用 **报告卡诚实标注 + 同质性谱重框**，不硬凑"4 域都图有用+可区分+双模态"。

## 方法
同节点同边，**只换 label**（post-process agent JSON，不重跑 crawler）：arxiv 细化、tmdb/steam 换非-genre label，重跑 leak-free NC（5 seed），对比报告卡指标。

## 1. arxiv 细化（4 子域 → 26 类 `(子域, subcat)`）✅ 跑通
细 label 源 = `graph_data.pt` 自带 `labels(4) + subcat_labels(7)`；`(子域, subcat)` 全局唯一（subcat 单独混子域、配对后不混），**不碰 5GB metadata**。26 类全 ≥200 节点、覆盖 100%、较平衡（max 6705/min 201）。

| | best NC | spread | signal margin | GNN−MLP gap | LP-ceiling(剪枝) | 正确性 |
|---|---|---|---|---|---|---|
| 原 4 类 | **0.900** | 0.007 | 0.611 | +0.007 | 0.261 ≤ 0.288 | ✅ |
| 细 26 类 | **0.463** | 0.008 | 0.317 | +0.001 | 0.051 ≤ 0.147 | ✅ |

- ✅ **去天花板饱和**：0.90→0.46，signal margin 0.32（>>majority 0.147），lp_ceiling 正确性仍 PASS。
- ❌ **模型区分度没恢复**：spread 仍 0.008（全模型挤 ~0.46），macro-F1 ~0.19。
- **诊断**：arxiv 真问题 = **feature-saturation**（摘要文本强决定子类、引用结构冗余）——细化 label 只降天花板、不创区分度。relabel 修不动。

产物：`data/arxiv_datasets/arxiv_multimodal_graph_agent_fine26.json`、`benchmark/results/newdomain_nc/nc_arxiv_summary_fine26.json`、脚本 `benchmark/run_exp1_arxiv_desat.py`。

## 2. tmdb 换非-genre label —— 否决（天生异质）
- 现 label=genres[0]（18 类，homo 0.239）。`shared_genre` 边定义性→去泄露时弱化，剩异质 `shared_actor`(0.216)/`same_director`(0.329)。
- 候选 label 基数（`tmdb_movies_raw.json`）：production_companies[0] **5050** 种、directors[0] **5796** 种——幂律长尾，无平衡 label 可用（top-N + "other" 会把 86%+ 节点塞一个类）。
- **结论**：tmdb 是天生异质图；任何非 genre 的合理 label 都不被 actor/director 边传递信号。relabel 修不动→当**异质谱端点**（heterophilous GNN 的合法测试床）。

## 3. steam 换非-genre label —— 否决（co-attribute 限 + 数据脏）
- 现 label=genres[0]（8 类，homo 0.524）。边 shared_genre/same_developer/same_publisher/shared_category 全是 co-attribute——只对等于自身的 label 同质（定义性）。
- 候选：developers[0] **1658** 种长尾（最大 70=2.5%，top-19 只覆盖 14%）；price 字段是本地化货币串（`"1 473,--€"`）跨币种不可解析。
- **结论**：steam 图价值问题同构于 tmdb——co-attribute 边只对定义性 label 同质。relabel 修不动。

## 4. wikiart —— 不动（已是 4 域里最好）
图有用（GNN−MLP +0.176）、非饱和（spread 0.176）、leak-free（ceiling 0.024≤0.025）。唯一旗 = 单模态（image 主导，text 纯标题无信息）——属 wikiart 性质，非 bug。

## 净结论（写论文怎么用）
1. **"benchmark 正确"4/4 成立**（无泄露，是 thesis 正确性支柱）——这是用户底线，**已达成**。
2. **"4 域都图有用 + 可区分 + 双模态"不可达**：arxiv feature-saturated / tmdb 异质 / steam co-attribute 限，皆**内禀结构特性**，非构造 bug，relabel 实验证实。
3. **正确定位**（覆盖逻辑线 §4.5 中心张力）：用**报告卡（Exp2）诚实标注每域字符** + **同质性谱重框**（feature-saturated arxiv ↔ heterophilous tmdb ↔ graph-useful wikiart）——把"图价值因域而异"本身当 portfolio 卖点，而非"每域都图有用"。
4. arxiv NC 不作"可区分 benchmark"头条；arxiv 价值靠 **规模(45k) + formula 独有模态 + LP(非退化 0.92) + 检索**。26 类 label 可作为 arxiv NC 的更诚实版本（非 trivial 天花板），但须披露 macro-F1 偏低 + feature-saturation。

## 残留（可选 future work，非 P0）
- 若仍想救 tmdb/steam 图价值：需引入**非 co-attribute 的同质边**（如 tmdb 加共享编剧/制片厂、steam 加共享玩家 tag）——即改图结构而非 label，属另一类实验（且需重爬数据）。
- 或换 heterophily-robust 模型（GPR-GNN/Linkx）验证 tmdb 是否对异质方法有区分度——把 tmdb 定位为异质 benchmark 后值得做。
