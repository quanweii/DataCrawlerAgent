# 论文正文段落 — spotify / amazon (数据从 Table 4 移到正文) · 2026-07-22

> 决策: 主数据集表(Table 4 = [fig4_dataset_overview_table.tex](figures/fig4_dataset_overview_table.tex))只保留 4 个 benchmark 域(arxiv/steam/tmdb/wikiart, 列全满含 LP-ceil)。
> spotify / amazon 不是 benchmark 资产, 而是**框架能力验证证据**, 各顶一个贡献点, 用文字写进正文:
>   - spotify → C3(声明式框架可复用 / 零代码可扩展)
>   - amazon → C1(自治 agent 框架泛化到 held-out 新真实域)
> 下方两段为可直接粘贴的英文正文(含数字, 已对权威源核实)。数字来源见文末。

---

## 段落 A — 写进 §3 数据集 / C3 框架可复用 (spotify)

> **位置**: §3 (Datasets) 的"框架可复用性/可扩展性"小节, 或 §5 (Framework+skill)。讲清"新域 = 一个 config 文件"。

**Beyond the four benchmark domains, we stress the declarative framework's reusability on
a fifth domain, \texttt{spotify}, constructed with \emph{zero} domain-specific code: 0
lines of Python and 41 lines of YAML, achieving 100\% reuse of the configuration-driven
graph builder (\texttt{build\_graph\_from\_config}). The resulting graph --- 10{,}000
nodes, 17{,}440 edges, 6 classes, edge homophily 0.741, text-only --- is benchmarked
under the identical protocol (node classification, link prediction, and leak-profile),
demonstrating that onboarding a new domain reduces to authoring a single configuration
file rather than writing custom construction code.**

中文要点(汇报口播): spotify 证明 C3 的"1 YAML = 1 域"不是口号 —— 0 行 Python、100% 复用引擎, 新域只写 41 行 config 就建出图并跑完 NC/LP/leak-profile。

---

## 段落 B — 写进 §3 数据集 / C1 自治泛化 (amazon)

> **位置**: §3 (Datasets) 末尾, 或 §1/§4 讨论 agent 诚实定位处。这是 C1 主线"agent 能泛化到全新真实源"的决定性 held-out 证据。

**To test whether the autonomous agent framework generalizes beyond the four
gold-standard domains, we applied it end-to-end to a held-out, real, noisy source it had
never seen --- the Amazon product catalog. With no human intervention and over three
seeds, the closed loop autonomously (i) discovered the label \texttt{main\_cat} (7
classes, each with $\geq$20 nodes), (ii) proposed the co-attribute relation
\texttt{same\_brand} (28{,}612 edges) while correctly rejecting the definitional
\texttt{same\_main\_cat}, and (iii) emitted a valid, leak-profiled graph of 5{,}438
nodes (7 classes, edge homophily 0.979). The build-validity and decontamination bar is
passed 3/3 times, providing held-out evidence that the agent --- not a hand-written
configuration --- can ingest a fresh source and produce a leak-profiled multimodal
attribute graph.**

中文要点(汇报口播): amazon 是 C1 主线的"泛化考试" —— 框架没见过的真实脏域, 全自治 3/3 seed 端到端通过: 自己发现 label、自己提关系、自己避定义性泄露、自己建真图。证明 agent 不只是会跑 4 个金标准域。

---

## 数字溯源(已核实, 可复用)
- **spotify**: 10000 节点 / 17440 边 / 6 类 / homophily 0.741 / text-only / 0 Python LOC / 41 YAML 行 → 来源 [leak_profile_table.md](leak_profile_table.md) + C3 逻辑线([paper_logic §0/§2](paper_logic_DnB_FINAL_2026-07-11.md))。
- **amazon**: 5438 节点 / 28612 边 / 7 类 / homophily 0.979 / label=main_cat / 关系=same_brand(28612 边)/ 避 same_main_cat(定义性) / 3 seed 端到端 → 来源 P5-B([path_b_phase0_results_2026-07-21.md §P5](path_b_phase0_results_2026-07-21.md))。
- **注意**: amazon 是单次自治跑的产物(非 canonical release benchmark); 这正是它进正文而非主表的原因 —— 它是 C1 泛化证据, 不是 benchmark 资产。
