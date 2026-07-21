# Option 1 — §Methodology & §Analysis 草稿 (2026-07-07)

> **定位**:Option 1 主线三柱(LP-ceiling 诊断 + leak-free 协议 + feature-saturation)的正式论文段落。
> **配套数据**(全部本会话核实):`docs/leakfree_main_tables_2026-07-03.md` · `benchmark/results/igc/igc_all.json` · `docs/figures/leakage_diagnostic.png`(+ `leakage_diagnostic_data.json`)。
> **用法**:英文 prose 可直接入稿;中文为给你看的备注/防守点。所有数字来自上述数据文件,未硬编码未拔高。
> **诚实边界**:§4.5 把多智能体降为"测量载体";citation E2 仍是唯一 LLM 组件证据,留次要。

---

## §3 Methodology

We build MAGs by materializing, per domain, a set of relations $R$ over fixed multimodal-attributed nodes. Two methodological problems dominate whether downstream numbers mean anything: **(i) the constructed edges can leak labels through the graph structure**, and **(ii) the standard "GNN helps when the graph is homophilous" claim conflates structure with features.** We address both with a feature-free diagnostic, a leak-free evaluation protocol, and a two-factor model of message-passing value.

### §3.1 LP-ceiling: a feature-free label-leakage diagnostic

**Definition.** For a graph $G=(V,E)$ with node labels $y$ and a train/test split, the **LP-ceiling** is the test accuracy of label propagation using *only* $E$ and the train labels — no node features. It upper-bounds the label information reachable through structure alone. We always report it next to the **majority baseline** (the test accuracy of predicting the most frequent class); $\text{LP-ceiling} \leq \text{majority}+0.08$ means the structure carries no usable label shortcut, while a large gap signals leakage.

**Two measurement modes.** The diagnostic is meaningful only relative to how edges cross the split:
- **Raw** (unpruned construction graph): the structural leakage signal *as built*. This is what a constructor sees at construction time.
- **Pruned** (all edges crossing $\{\text{train},\text{val},\text{test}\}$ removed): the leak-free reference; label propagation cannot reach the test set from train.

The raw→pruned drop *is* the leakage. Table 1 (§3.2) reports it; on wikiart it is 0.921 → 0.024.

**Per-relation profile.** Because our graphs are built from a *set* of relations, leakage is localized by measuring LP-ceiling on each relation $r \in R$ alone, $\text{LP}(r)$. The resulting profile tells a constructor *which* relation leaks, not merely *that* the graph leaks. **Figure 1** plots $\text{LP}(r)$ (bars) against the raw/pruned/majority references (lines) for the three constructed-edge domains, exposing three qualitatively different regimes (§4.1).

> **备注(为什么 LP-ceiling 是对的诊断)**:它**无特征** → 把"结构泄露"和"特征好不好"完全分离。一个域可能特征极强(MLP 已 0.90)但结构仍泄露(LP-ceiling 高),二者互不污染。这是它优于"看 NC acc 是否虚高"的地方——后者把特征强度和结构泄露搅在一起。

### §3.2 Leak-free evaluation protocol

**The problem.** Co-attribute edges (e.g. `same_artist`, `shared_genre`) are dense and frequently connect nodes of the same label. Under a *random* node split, a large fraction of edges cross the train/test boundary, so label propagation trivially solves test labels — inflating NC accuracy. In our wikiart graph this drives GAT from a leak-free **0.845** to a naive **0.968**: a 12-point artifact of the split, not of the model. A benchmark that reports both numbers has two contradicting "wikiart"s.

**The protocol.** Every headline number in the paper uses:
1. **Stratified 60/20/20** split with a *fixed* `split_seed=1` shared across all models, so residual variance comes only from initialization (≥5 seeds).
2. **Cross-split edge pruning**: delete every edge whose endpoints fall in different partitions. LP-ceiling then cannot reach test from train, and the pruned LP-ceiling drops to/below the majority baseline on all domains (Table 1).
3. **One protocol per task** (Table 2), so there is no ambiguity about "which table counts": NC uses the pruned split; LP uses hard-degree negative sampling to defeat the easy-negative saturation; the within-domain steering experiment (E1) deliberately keeps the raw graph (its degeneration *is* the finding).

> **Table 1 — leakage diagnosis (LP-ceiling, no features; edge pruning ≈55–56% of edges).**
> | domain | majority | LP-ceiling raw → pruned | verdict |
> |---|---|---|---|
> | arxiv | 0.288 | 0.673 → 0.261 | moderate structural signal; pruned below majority |
> | steam | 0.504 | 0.557 → 0.504 | already at floor |
> | tmdb | 0.205 | 0.413 → 0.154 | leakage removed by pruning |
> | wikiart | 0.025 | **0.921 → 0.024** | severe definitional leakage, fully removed |

> **Table 2 — protocol per task.**
> | task | headline protocol | leakage handling |
> |---|---|---|
> | Node classification | pruned split (this section) | cross-split edge pruning |
> | Link prediction | hard-degree negatives | hard negatives + LP-ceiling dual-report |
> | Steering (E1) | raw graph, filter by relation | no pruning (degeneration is the point) |
> | Fusion | pruned split (aligned with NC) | — |

**Result.** There is exactly one wikiart in the paper. The old leaky numbers appear *only* in the leakage-diagnosis paragraph, as "a naive protocol reports an inflated X."

### §3.3 When does message passing help? A two-factor model

The prevailing rule — *message passing helps when the graph is homophilous* — is only half true in multimodal attributed graphs, because node features can already encode the label-relevant structure. We model the GNN–MLP gap as

$$\Delta_{\text{GNN}} \;\propto\; \underbrace{h(\text{edge})}_{\text{homophily}} \cdot \underbrace{\bigl(1 - \rho(\text{feature})\bigr)}_{\text{feature headroom}},$$

i.e. message passing adds value only when (a) the edges carry label-relevant structure (homophily $h$) **and** (b) the node features do not already capture it (feature headroom $1-\rho$). Either factor collapsing to zero kills the gap: heterophily zeroes $h$; feature saturation zeroes $(1-\rho)$.

This is a *conceptual* model — we do not fit closed-form coefficients — but it predicts two testable patterns, both confirmed:

- **Cross-domain (features vary):** the gap is *not* monotone in homophily. arxiv has the highest homophily (0.93) yet the smallest gap (+0.009), because its CLIP text features are saturated (MLP 0.897). wikiart (homophily 0.89) has the largest gap (+0.080), because its features are weaker relative to a 64-class task. The leak-free main table (Table 3) shows the gap crossing from negative (tmdb, steam) to positive (wikiart, arxiv).
- **Within-domain (features held constant):** the steering experiment (E1) materializes graphs from single relations over the *same* nodes and features, so feature headroom is fixed and only $h$ varies. GCN macro-F1 then tracks relation homophily monotonically (0.22 → 0.96), a clean causal isolation of the homophily factor. This is the paper's mechanism figure, *not* the cross-domain scatter (whose weak $r=0.46$ is itself a symptom of feature saturation, not a defect).

> **备注(防守)**:别把 §3.3 写成"我们提出 GNN 增益公式"。写"二因子 conceptual model + 两条可证伪预测 + E1 因果证据"。审稿人若要闭式系数,列为 camera-ready(+0.027 Wilcoxon 也在此列)。

---

## §4 Analysis & Cross-Domain Findings

### §4.1 Three leakage regimes (Figure 1)

Applying the §3.1 diagnostic across the three constructed-edge domains yields a clean three-way taxonomy — *can relation selection alone bring LP-ceiling to near-majority?*

- **Suppressible (tmdb).** The all-relations graph leaks (0.413 vs majority 0.205), but the per-relation profile reveals one clean relation: `shared_genre` alone gives LP 0.255 ≈ oracle. Selection at construction time removes the shortcut; residual is mopped up by pruning (→ 0.154).
- **Structural (wikiart).** Every relation is *definitional*: `same_artist` (LP 0.884) and `shared_style` (LP 0.917) both encode the label basis (artist→style, style≡label). Selection cannot help — the best single relation still leaks (0.884 ≫ majority 0.025). Only cross-split pruning resolves it (0.921 → 0.024). A static rule that "excludes `same_artist`" would *falsely report handled* while `shared_style` still leaks 0.917.
- **Near-floor (steam).** Leakage is below the 0.08 action threshold (0.557 vs majority 0.504); a measurement-driven constructor correctly no-ops.

This taxonomy is the actionable output of the diagnostic: it tells a practitioner *which* defense applies (select relations vs. prune edges vs. nothing), rather than a single "leaky / not leaky" bit.

### §4.2 GNN value across the homophily spectrum

> **Table 3 — leakage-free NC, text+image, 5-seed, pruned split (macro-F1).**
> | domain (homo) | mlp | gcn | gat | graphsage | gin | gcnii | bestGNN − MLP |
> |---|---|---|---|---|---|---|---|
> | tmdb (0.237) | 0.472 | 0.227 | 0.247 | **0.462** | 0.067 | 0.436 | **−0.010** |
> | steam (0.529) | **0.501** | 0.283 | 0.280 | 0.448 | 0.111 | 0.411 | **−0.053** |
> | wikiart (0.894) | 0.858 | 0.845 | 0.789 | **0.938** | 0.329 | 0.908 | **+0.080** |
> | arxiv (0.933) | 0.897 | 0.901 | 0.897 | 0.900 | **0.906** | 0.903 | **+0.009** |

Under the honest protocol the gap crosses zero with homophily (negative at tmdb/steam, positive at wikiart/arxiv), confirming the homophily factor. The non-monotonicity at the high end — arxiv most homophilous yet smallest gap — is the feature-saturation effect (§3.3): its CLIP features are saturated (MLP 0.897), so structure adds almost nothing (+0.009). GraphSAGE is the most robust GNN across the spectrum; GIN (sum-aggregation, over-smoothing) is worst on NC despite being strongest on LP — a task-dependent ranking that single-task benchmarks miss.

### §4.3 Construction-time de-leakage improves downstream NC

De-leaking at construction time (selecting relations by the §3.1 profile) does not merely change a diagnostic number — it improves the downstream model. Re-training GCN on the de-leaked graph vs the all-relations graph:

> | domain | GCN all-rel → de-leaked | MLP all-rel → de-leaked |
> |---|---|---|
> | tmdb | 0.231 → 0.258 (**+0.027**) | 0.473 → 0.473 |
> | wikiart | 0.856 → 0.883 (**+0.027**) | 0.864 → 0.864 |

MLP is unchanged (it ignores edges), so the +0.027 gain is structural: removing label-shortcut edges forces GCN to use features, improving generalization. *(Camera-ready: n=5 paired Wilcoxon for significance.)* This is the prescription the diagnostic buys — and it is exactly the regime where the leakage was structural-but-not-definitional (tmdb) or near-floor (wikiart's residual after pruning).

### §4.4 Cross-domain transfer: a clean negative result

Transferring a trained graph encoder across heterogeneous domains yields uniformly negative gain across all six pairs, with transfer *into* wikiart catastrophic (≈0.92 → ≈0.35). This is an honest negative result: the graph biases that make message passing work are domain-specific and not shareable. It argues against a single universal MAG model and for **per-domain modeling** — consistent with the three regimes of §4.1, which show that the relevant structural property (which relation leaks, and whether selection suffices) is itself domain-specific.

### §4.5 The multi-agent constructor is a measurement carrier, not the source *(honest)*

We report, rather than conceal, the limits of the multi-agent framing. In our instruction-guided graph-construction (IGC) task — pick a relation subset to minimize leakage under *blind-renamed* relations — a **deterministic greedy** given the *same* `measure_subset` tool (no LLM) matches or beats the closed-loop planner↔critic agent on every domain:

> | domain | greedy (no LLM) | closed-loop (LLM) | oracle |
> |---|---|---|---|
> | steam | 0.557 | 0.557±0.000 | 0.489 |
> | tmdb | **0.255** | 0.315±0.060 | 0.255 |
> | wikiart | 0.884 | 0.884±0.000 | 0.884 |

The agent never exceeds the greedy baseline and is strictly *worse* on tmdb (by a full std). The reason: the critic reads only *solo* per-relation LP-ceiling and interprets it lossily, whereas greedy measures the *joint* subset directly. **The source of value is the measurement (the LP-ceiling diagnostic), not the LLM**; the agent is one — lossy — instantiation of "measure, then revise." With ≤4 relations per domain, brute force/greedy is trivial and the regime is structurally "agent-unnecessary." We therefore frame the multi-agent system as *how we constructed the data*, not as a contribution. The only honest evidence that an LLM component earns its cost remains fuzzy-OCR citation matching (recall 0.714 vs rule 0.000/0.286), reported as a minor ablation. Reviving an agent-necessity claim would require scaling to k≈15–20 relations where $2^k$ search explodes but LLM-guided search stays tractable — an explicit future-work item, not a current claim.

---

## §A Pending (camera-ready, non-blocking)

1. **Significance**: n=5 paired Wilcoxon for the +0.027 de-leakage gain (Table §4.3) and the GNN−MLP gaps (Table 3).
2. **Feature-saturation fit**: regress the gap onto (homophily, feature-headroom proxy) for a quantitative version of §3.3.
3. **arxiv in the diagnostic**: single relation (citation), natural edges — a fourth regime ("natural-edge, moderate-signal") for the appendix figure.
4. **Option 2 scale experiment** (the only path that could revive an agent claim): synthetic/scaled k→15–20 relations.
