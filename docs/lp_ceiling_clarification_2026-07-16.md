# LP-ceiling 命名澄清与精确定义（P0-2，2026-07-16）

> 对抗审稿（2026-07-16）Reviewer 4 M1 质疑 "LP-ceiling 度量错了对象 / LP-ceiling=0.024 与 LP AUC=0.97 自相矛盾"。
> 本文经代码核查（`benchmark/data/splits.py:82` + `benchmark/run_newdomain_lp.py:252`）给出精确结论。

---

## 0. 一句话结论

> **"LP-ceiling" 里的 LP = Label Propagation，不是 Link Prediction。** 它度量的是 **transductive 节点分类(NC)的结构泄露上限**，与链路预测(LP, Link Prediction)任务的 AUC 是**两个不同任务的指标，不矛盾**。Reviewer 4 的"自相矛盾"判断源于命名歧义导致的误读；但这个命名歧义是**论文真实风险**（真实审稿人也会同样误读），**必须改名**。

---

## 1. 代码核查：LP-ceiling 到底是什么

`benchmark/data/splits.py:82` 的 `label_propagation_ceiling(y, edge_index, train_mask, test_mask, num_iters=50)`：

> 迭代式 **label propagation**（无特征、仅图结构）→ test 上的 acc/macro-F1。
> 在【原始图】上 = 结构泄露上限（wikiart ≈ 0.915）。
> 在【剪枝图】上应跌到≈多数类 = 泄露被消除的证据。

它做的事：用 train 节点的 one-hot 标签做种子，沿**图边**迭代扩散（标准 label propagation 算法），在 test 节点上取 argmax。

**它度量的是**：在 transductive NC 设定下，如果 GNN 沿全图做消息传递，**train 的标签能经图结构"渗透"到 test 多少**。这是 NC 的结构泄露上限——和 GNN 用的是同一套 message-passing 机制，所以是 NC 泄露的诚实天花板。

**它不度量**：链路预测任务（预测两节点是否连边）的任何东西。

---

## 2. 为什么 "0.024 vs 0.97" 不矛盾

| 指标 | 全名 | 任务 | 值（wikiart 剪枝后） |
|---|---|---|---|
| **LP-ceiling** | **L**abel-**P**ropagation ceiling | 节点分类(NC) | 0.024 ≈ majority |
| **LP AUC** | **L**ink-**P**rediction AUC | 链路预测 | 0.97（random neg） |

- **NC 的 LP-ceiling = 0.024**：剪掉跨切分边后，train 和 test 在图上**不连通**，label propagation 无法把 train 标签传到 test → NC 的结构泄露被消除（跌到多数类基线）。✅ 这是 C2 协议**成功**的证据。
- **LP AUC = 0.97**：链路预测任务（同质性 0.89 的图上，"相连≈同类"使正负边易分）本身可预测性就高。这**不是泄露**，是任务的固有难度低——论文已用 **hard / hard_degree 三档负采样**应对（random 0.97 → hard_degree 0.84，把任务拉回有区分度的区间）。

两者一个是 NC 泄露上限、一个是 LP 任务难度，**本来就不是一回事，不矛盾**。Reviewer 4 把 "LP-ceiling" 读成 "Link-Prediction ceiling" 才会觉得矛盾。

---

## 3. LP（链路预测）任务的泄露防护：已干净

Reviewer 4 还担心 Zhu (WSDM'24) 的 **target-edge inclusion 泄露**（测试正边留在 message-passing 图里）。核查 `benchmark/run_newdomain_lp.py:252`：

```python
train_pos = load_lp_edges(domain, graph, "train_pos")
msg_edges = _make_undirected(train_pos)   # ← message passing 只用 train 正边
val_pos = load_lp_edges(domain, graph, "val_pos")
test_pos = load_lp_edges(domain, graph, "test_pos")   # ← 不进 msg_edges
```

- **message-passing 图只含 train 正边**；val/test 正边**根本不在消息传递图里** → **无 target-edge inclusion 泄露**。
- 负采样禁区 `forbidden_keys = 全部正边(train+val+test)双向 key` → 负样本不会采到任何 held-out 正边。
- ✅ LP 链路预测的泄露防护是干净的，reviewer 4 的 Zhu I3 担忧在本仓库**不成立**。

---

## 4. 必须做的两件事（改名 + 诚实 limitation）

### 4a. 改名（消除歧义，最高优先级）

"LP-ceiling" 这个名字**必然**被读成 Link Prediction ceiling。改名方案：

| 旧名 | 新名（推荐） |
|---|---|
| LP-ceiling | **structure-leakage ceiling**（或 `LP-NC ceiling`，括注 Label-Propagation） |

论文里首次出现时写：
> *structure-leakage ceiling* (computed via label propagation on the graph structure; **not** a link-prediction metric) — the test accuracy achievable by propagating train labels along edges **without any node features**, upper-bounding the structural leakage available to a transductive GNN.

### 4b. 诚实 limitation（高同质图 random-neg LP 接近 ceiling）

`structure-leakage ceiling` 消除的是 **NC 的跨切分边泄露**，不改变 LP 任务在同质图上的**固有可预测性**。因此：

- 论文/README 必须标注：**"on high-homophily graphs, random-negative LP AUC can approach ceiling (e.g. wikiart 0.97); this reflects task predictability, not residual leakage — hence we additionally report hard / hard_degree negative sampling."**
- 每个 released 数据集 README 第一行写明默认 LP 协议下 AUC 可能很高、必须使用附带的 hard_neg CSV。

---

## 5. 对 C2 护城河的影响（修正对抗审稿结论）

Reviewer 4 M1 原文断言 "LP-ceiling 度量错了对象 / 作者自己的数据打穿自己的指标"。**核查后修正**：

- ❌ "自相矛盾 / 度量错对象"：**不成立**——基于命名误读（LP=Label Propagation ≠ Link Prediction）。structure-leakage ceiling 忠实度量了它声称度量的量（NC 结构泄露上限），剪枝图 0.024=majority 是协议成功的正确表现。
- ✅ "命名会误导审稿人"：**成立**——必须改名（§4a）。
- ✅ "LP 任务在高同质图 AUC 高"：**成立但非泄露**——任务固有难度，已用三档负采样应对，需在 limitation 写明（§4b）。
- ✅ "LP target-edge inclusion 泄露"：**不成立**——msg_edges 只用 train_pos，已干净（§3）。

**结论**：C2 护城河（measurement-grounded decontamination）比 reviewer 4 判定的更稳——LP-ceiling 不是"度量错对象"，而是"名字起错了"。改名 + limitation 后，C2 的 NC 半边（structure-leakage ceiling 量化跨切分边泄露）完全成立。仍待补的是 §6 的"度保持扰动"（属 P0-1 steering 范畴，不在本文档）。

---

## 6. 相关代码与文件

- 定义：`benchmark/data/splits.py:82` `label_propagation_ceiling`
- 剪枝：`benchmark/data/splits.py:70` `prune_cross_split_edges`
- LP runner（msg_edges 干净）：`benchmark/run_newdomain_lp.py:252`
- leak-profile 数字：`benchmark/results/leak_profile_table.json`
- 对抗审稿原文：`docs/` memory `adversarial-review-findings-2026-07-16`（本核查修正其 LP-ceiling 条目）

*2026-07-16 核查。结论：改名 + 写 limitation；C2 护城河 NC 半边成立。*
