# Related Work — 论文最终版（FINAL, 2026-07-16）

> 本文是 §2 Related Work 的**单一权威来源**：定位策略 + 可直接贴的英文正文 + 核心对比表 + bibtex。
> 数字全部为 **2026-07-15 去泄露后**最新值；权威数字总表见 [`experiment_results_FINAL_decontaminated_2026-07-16.md`](experiment_results_FINAL_decontaminated_2026-07-16.md)。
> 论文逻辑线见 [`paper_logic_DnB_FINAL_2026-07-11.md`](paper_logic_DnB_FINAL_2026-07-11.md)。
>
> ⚠️ **旧定位已废止**：`PPT_相关工作对比分析.md`（2026-03）的"学术论文垂直 + 公式模态"头条已过时。当前论文是 **4 域 leak-profiled MAG benchmark**（arxiv/steam/tmdb/wikiart），arxiv 仅 1/4，公式只当"模态丰富度"卖（消融仅 +0.0007，不卖性能）。
>
> 📌 **2026-07-16 全篇联网核查修订（16 篇逐篇核实）**：
> - 修正 1 处事实错误（BenchAgents 实为多模态）、8 处 venue/元数据错误、1 处重复引用（MM-Graph = Mosaic of Modalities，合并）。
> - **新增 §2.5 Graph Structural Leakage 子节**：主动引用并区分 Platonov (NeurIPS'23/TPAMI'24)、Zhu (WSDM'24)、Li (NeurIPS'23 D&B)、Brière (bioRxiv'25)——图上结构性泄露早有 prior，必须在文中划清界限。
> - **两处价值主张收窄**（见 §0、§3）：不再说"首个 MAG benchmark"（MAGB/Mosaic 在先）；不再说"结构性防泄露是图 benchmark 独有/没人做过"（Platonov 等在非多模态图上做过）→ 改为"首次在 MAG 上 + 首次量化 LP-ceiling"。

---

## 0. 一句话定位（核查后收窄版）

> **在"多智能体构造 + 多模态属性图 benchmark + 结构性防泄露协议 + 声明式可复用框架"这个四叉路口，之前没有任何一站过——每个对手只占其中一条轴。**
>
> 精确地：
> - 我们**不是**首个 MAG benchmark（MAGB [KDD'25]、Mosaic of Modalities [CVPR'25] 已在先）——我们的独占性来自 **agent 驱动构造 ∩ 测量驱动去泄露协议 ∩ steering 因果实验** 这一**组合**，而非 MAG benchmark 单一角。
> - 图上的结构性泄露检测**已有 prior**（Platonov 在网页图 NC、Zhu/Li 在 LP）——但它们**全部是非多模态图，且没有 LP-ceiling 量化框架**。我们首次把 leak-profiling + LP-ceiling 量化 + 多模态 text 定义级泄露 + NC/LP 统一协议**带进 MAG**。

最接近的两类——agent-built benchmark（BenchAgents/DatasetAgent）和 multimodal graph benchmark（OpenMAG/MAGB/Mosaic）——**前者有 agent 但造的是文本/图像任务实例、无图无防泄露；后者有 MAG 但手工建造、不做结构性泄露量化、无 agent/声明式框架**。

---

## 1. 五群对手（5 clusters；核查后）

| Cluster | 代表工作 | 他们有 | 他们缺（相对我们）|
|---|---|---|---|
| **A. Agent-built dataset/benchmark** | BenchAgents, DatasetAgent, AutoBencher, CAMEL, Benchmark Self-Evolving, AgentInstruct | agent 自动造数据 | 造的是文本/QA/图像**任务实例**，**无图结构**；不查结构性泄露；依赖人工编辑 |
| **B. Agent-built graph / KG** | KARMA, Bian LLM-KG survey, Neo4j Graph Builder | agent 建图 | 建的是**知识图谱**(实体-关系三元组)，非属性图 benchmark；不发布 benchmark、不防泄露 |
| **C. Multimodal graph benchmark** | OpenMAG, MAGB, Mosaic of Modalities (=MM-Graph) | MAG benchmark，广 | **手工建造**；**无系统性、测量驱动的结构泄露量化/去泄露**；无 agent/skill 框架 |
| **D. Benchmark leakage / contamination (LLM)** | AntiLeakBench, BDC Survey, Emperor's New Clothes, DataSAIL | 防泄露研究 | 防的是 **LLM test-in-train 污染**或**通用 i.i.d. 实体对相似度泄露**，**非 message-passing 图的跨切分边结构性泄露、无 LP-ceiling** |
| **E. Graph structural leakage（非多模态图）** 🆕 | Platonov, Zhu (LP pitfalls), Li (LP eval), Brière | **图上**结构性泄露检测（NC 重复节点 / LP target-edge） | **全是非多模态图**（网页图/KG/分子）；**无 LP-ceiling 量化**；不在 MAG；无 agent 框架 |

**关键**：五群对手**没有一个同时覆盖"多智能体造 + MAG + MAG 上的测量驱动去泄露 + 声明式可复用"** → 我们的独占性成立，但措辞必须精确（见 §0、§3）。

---

## 2. 核心对比表（核查后，直接进论文/PPT）

| 能力维度 | BenchAgents | DatasetAgent | AutoBencher | KARMA | OpenMAG | MAGB | Mosaic of Modalities | AntiLeakBench | **Ours** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 多智能体建造 | ✅ 4 agent | ✅ 4 agent | 🟡 单 LLM | ✅ 多 agent | ❌ 手工 | ❌ 手工 | ❌ 手工 | 🟡 自动 | **✅ agent + skill** |
| **多模态属性图(MAG)** | ❌ 任务实例 | 🟡 仅图像 | ❌ QA | ❌ 知识图谱 | ✅ | ✅ | ✅ | ❌ | **✅** |
| **MAG 上的结构性防泄露**(LP-ceiling/跨切分剪枝) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌¹ | ❌² | **✅** |
| 声明式可复用(1 YAML=1 域) | ❌ 需人工编辑 | ❌ | 🟡 QA-spec | ❌ | ❌ | ❌ | ❌ | ❌ | **✅ 94%复用, spotify 零代码** |
| 防泄露量化(leak-profile 元数据) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 🟡 LLM 污染 | **✅ 每域一等元数据** |
| 因果结构实验(steering) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅ r=0.846** |
| benchmark 评测广度(NC/LP/聚类/few-shot/…) | ❌ | ❌ | ❌ | ❌ | ✅ 最广 | 🟡 | 🟡 | ❌ | **✅** |
| 公开 benchmark 资产 | 🟡 NLP | 🟡 图像 | 🟡 QA | ❌ | ✅ 多域 | ✅ | ✅ | 🟡 | **✅ 4+1 域** |

¹ Mosaic of Modalities 做了**基础的随机切分 + LP 测试边移除**（并引用了 WSDM'24 target-edge-inclusion 陷阱），但**无 homophily-aware 的结构泄露分析、无 LP-ceiling 量化、无系统去泄露协议**。
² AntiLeakBench 防的是 LLM test-in-train(knowledge-cutoff)文本污染，**与图的跨切分边结构性泄露是不同机制**。

---

## 3. 我们的优势与价值定位（核查后精确版）

| # | 优势 | 精确论点（核查后） | 证据 |
|---|---|---|---|
| 1 | **唯一"多智能体 + 声明式 skill 构造 MAG benchmark 套件"** | A 群有 agent 无图；C 群有 MAG 无 agent。注意：**不说"首个 MAG benchmark"**（MAGB/Mosaic 在先），独占性来自 agent+leak-control+steering 的组合 | 4 个 `*_multimodal_graph_agent.json`（字面=agent 产物）+ formula 独有模态 + 同质性谱 0.24→0.93 |
| 2 | **MAG 上的测量驱动去泄露是独有护城河**（收窄版） | 图上结构性泄露**已有 prior**（Platonov NC/Zhu LP，见 §2.5），但**全在非多模态图、无 LP-ceiling 量化**。我们**首次把 leak-profiling + LP-ceiling 量化 + 多模态 text 定义级泄露 + NC/LP 统一协议带进 MAG** | LP-ceiling 坍塌 wikiart 0.921→0.024，4 域剪枝后全 ≤ 多数类；text 字段 0 参数 100% 恢复标签 |
| 3 | **去泄露靠测量不靠 LLM**（诚实，反 BenchAgents） | BenchAgents 把 agent 当 benchmark 质量的 load-bearing 环节；我们证明 `greedy(无LLM)=oracle` | 盲测 tmdb 0.300=0.300、wikiart 0.880=0.880；当**优点**写：协议 principled+robust |
| 4 | **可审计的声明式可复用**（94% + spotify 零代码） | BenchAgents 需人工大量编辑；AutoBencher 的"声明式"非建图骨架 | code-reuse manifest（SLOC 计）；spotify 41 行 YAML / 0 Python 造出 1 万节点真图 |
| 5 | **steering 因果实验——没人做过的因果隔离** | 所有对手都没做"同节点同特征只换边"的 MAG 因果对照 | GCN−MLP gap vs 同质性 **r=0.846** [0.600, 0.960], p=0.0005, n=12；MLP bit-identical 自检 |

### 🔥 杀手级 positioning 钩子（写进 Related Work 章末/Intro）

> OpenMAG 的论文**自己观察到**："excessive neighbors introduce semantic noise / over-dense neighborhoods dilute distinct node features"——他们**只是观察，没有机制去控制**。我们用 steering + 邻居扫描**直接提供了这个旋钮**。
>
> 口号：**"They observe neighbor-density sensitivity; we provide the knob."**

---

## 4. 诚实的 gap（别人更强的地方，必须承认）

1. **广度不如 OpenMAG**：他们 19 数据集/6 域/16 encoder/24 模型/8 任务；我们 4+1 域、10 模型。定位 = **"少而精 + 防泄露 + 可控"** 的互补型基准，不抢广度王。
2. **不是首个 MAG benchmark**：MAGB [KDD'25]、Mosaic of Modalities [CVPR'25] 在先。我们补的是 agent 构建 + MAG 上的去泄露 + steering，**不是 MAG benchmark 本身**。
3. **agent 通用性不如 BenchAgents**：他们的 4-agent 分解对任意 NLP benchmark 通用；我们诚实 scope agent 价值到"语义 wrangling(citation) + 可复用接口 + 端到端编排"三点。
4. **KG 构造成熟度不如 KARMA**（NeurIPS'25 Spotlight，~55 引）：我们不声称做 KG，只做 co-attribute 属性图。
5. **encoder 我们做 frozen + per-domain profiling**：OpenMAG 做 frozen+trainable；我们 scope out trainable（诚实），但提供 per-domain encoder profiling（clip_large 赢 arxiv 长文档，siglip 赢 steam 短文本）。
6. **图结构性泄露检测不是我们首创**：Platonov/Zhu/Li 在非多模态图上先做了（见 §2.5）；我们的新颖性在"MAG + LP-ceiling 量化 + 多模态 text 泄露 + 统一协议"。
7. **agent 在去泄露决策上不 load-bearing**：诚实红线，**绝不**夸大成"agent 防泄露/建图决策比规则强"——greedy=oracle 会打穿。

> 承认这些 gap 不是示弱，是让定位可信——审稿人看到你清楚边界 + 主动引用相邻 prior，反而更信"独占交集"主张。

---

## 5. 【可直接贴论文】英文 Related Work 正文（核查后）

> 论文语言为英文，以下为 §2 草稿，引用 key 见 §6 bibtex。每小节先述、再定位差异。

### 2.1 Multimodal Attributed Graph Benchmarks

MAGB [Yan et al. 2025] introduced a systematic multimodal attributed graph (MAG) benchmark and studied the relative efficacy of GNN-as-predictor versus VLM-as-predictor paradigms across text and visual attributes. Mosaic of Modalities [Zhu et al. 2025] (the *MM-Graph* project) assembled seven text+visual graph datasets spanning classification, link prediction, and knowledge-graph completion. OpenMAG [Wan et al. 2026] substantially expanded this line to 19 datasets, 16 encoders, and 24 models, establishing the current breadth baseline. **All three are hand-curated, and none performs homophily-aware structural leakage analysis, reports an LP-ceiling, or attaches leak-profile metadata.** Mosaic of Modalities does perform standard test-edge removal and random splitting (and cites the target-link-inclusion pitfall), but this is basic LP hygiene rather than a measurement-grounded, homophily-aware decontamination protocol. This is precisely the gap our C2 protocol targets, and it motivates the per-domain leak-profile metadata we attach to every released dataset. We emphasize that we do **not** claim the first MAG benchmark — our contribution is the *agent-driven construction, MAG-level leak profiling, and causal steering* layered on top of this benchmark family.

### 2.2 LLM-Agent Pipelines for Dataset and Benchmark Construction

A recent line of work uses LLM agents to automate dataset construction. BenchAgents [Butt et al. 2025] orchestrates four agents (planning, generation, verification, evaluation) to synthesize **multimodal (language + vision) task benchmarks** for constraint satisfaction and causal reasoning, positioning the agent loop as load-bearing for benchmark quality and relying on developer-in-the-loop editing. DatasetAgent [Sun et al. 2025] uses multimodal LLM agents to build image datasets for classification, detection, and segmentation. AutoBencher [Li et al. 2025] declaratively specifies QA topics for an LLM to generate salient, novel, and difficult questions. CAMEL [Li et al. 2023] uses role-playing agents to generate conversational training data; Benchmark Self-Evolving [Wang et al. 2025] uses agents to dynamically extend existing LLM benchmarks; AgentInstruct [Mitra et al. 2024] generates synthetic instruction-tuning data. **These efforts all target text, QA, or image task instances — none involves graph structure, none enforces structural leakage control, and (as BenchAgents reports) several depend on heavy developer-in-the-loop editing.** In contrast, we apply agents to *multimodal attributed graph* construction and use a declarative skill to reduce the per-domain increment to roughly 280 lines of code (≈94% reuse), with Spotify built end-to-end from 41 lines of YAML and zero Python. Our greedy-without-LLM ablation further shows the *decontamination decision itself does not require an LLM* — measurement (LP-ceiling, homophily) suffices — which we frame as a robustness feature rather than a weakness.

### 2.3 LLM-Agent Knowledge Graph Construction

KARMA [Lu et al. 2025] (NeurIPS 2025 Spotlight) uses a multi-agent LLM system to enrich knowledge graphs from scientific literature via entity discovery, relation extraction, and conflict resolution; a recent survey [Bian 2025] reviews LLM-driven KG construction pipelines. Neo4j's LLM Graph Builder provides an industrial text-to-graph tool. **Knowledge graphs (entity–relation triples) belong to a different graph family from the co-attribute attribute graphs we construct**, and these works do not release benchmarks, evaluate GNN performance, or address leakage. Notably, the survey lists *multimodal* KG construction as an open direction, reinforcing — rather than threatening — our positioning. We therefore position our co-attribute MAG family as orthogonal to KG construction.

### 2.4 Benchmark Leakage and Data Contamination

AntiLeakBench [Wu et al. 2025] automatically constructs LLM benchmarks with updated real-world knowledge to guarantee contamination-freeness; the Benchmark Data Contamination survey [Xu et al. 2024] reviews definition, detection, and mitigation of LLM test-in-train contamination; "The Emperor's New Clothes in Benchmarking" [Sun et al. 2025] rigorously audits contamination-mitigation strategies for LLM benchmarks; DataSAIL [Joeres et al. 2025] formalizes leakage-reduced data splitting as a combinatorial optimization over inter-split similarity. **These efforts target LLM test-in-train contamination or i.i.d. entity-pair similarity leakage — none operates on message-passing graph structure, none reports an LP-ceiling, and their scope does not cover MAG structural leakage.** (DataSAIL does address cross-split *interactions* for 2D biomedical entity pairs and its NP-hardness proof reduces from graph minimum k-section, so it is conceptually adjacent; we are precise that it does not target message-passing graph cross-split edges or quantify LP-ceiling.)

### 2.5 Graph Structural Leakage (non-multimodal graphs)  🆕

Closest to our decontamination contribution is prior work on **structural leakage in (non-multimodal) graphs**. Platonov et al. [2023/2024] show that popular heterophily benchmarks (Squirrel, Chameleon) contain many duplicate nodes that cause train–test structural leakage, and that GNN accuracy drops sharply after deduplication. For link prediction, Zhu et al. [2024] define *implicit test leakage* from leaving target edges in the message-passing graph, and Li et al. [2023] survey LP evaluation pitfalls (splitting, negative sampling, test contamination); Brière et al. [2025] benchmark data leakage in biomedical KG link prediction. **These works establish that graph structural leakage is a real and studied problem — we do not claim to discover it. Our distinct contribution is to (i) bring leak-profiling and decontamination into the *multimodal* attribute graph setting for the first time, (ii) introduce the *LP-ceiling* as a quantifiable, constructible degeneracy metric, (iii) expose a previously undocumented *definition-level leakage in multimodal text attributes* (label-bearing metadata fields recoverable at 100% with zero parameters), and (iv) unify NC and LP structural leakage into a single measurement-grounded protocol.** We therefore position our C2 protocol as the *MAG instantiation and unification* of this line, not its origin.

### 2.6 Positioning Statement

**In summary, we are the first work to simultaneously satisfy (i) multi-agent, declaratively reusable construction; (ii) multimodal attributed graph benchmarks spanning a natural homophily spectrum; (iii) a measurement-grounded structural decontamination protocol with LP-ceiling quantification on MAGs; and (iv) a causal, controlled structure–performance experiment.** Where OpenMAG *observes* that excessive neighborhood density dilutes node features, *we provide the knob* to study it.

---

## 6. 参考文献（bibtex，核查后）

```bibtex
% --- Cluster A: agent-built dataset/benchmark ---
@inproceedings{butt2025benchagents,
  title     = {BenchAgents: Automated Benchmark Creation with Agent Interaction},
  author    = {Butt, Natasha and Chandrasekaran, Varun and Joshi, Neel and Nushi, Besmira and Balachandran, Vidhisha},
  booktitle = {ICLR 2025 Workshop on Data Problems for Foundation Models (DATA-FM)},
  year      = {2025}
}

@article{sun2025datasetagent,
  title   = {DatasetAgent: A Novel Multi-Agent System for Auto-Constructing Datasets from Real-World Images},
  author  = {Sun, Haoran and Bian, Haoyu and Zeng, Shaoning and Rao, Yunbo and Xu, Xu and Mei, Lin and Gou, Jianping},
  journal = {arXiv preprint arXiv:2507.08648},
  year    = {2025}
}

@inproceedings{li2025autobencher,
  title     = {AutoBencher: Towards Declarative Benchmark Construction},
  author    = {Li, Xiang Lisa and Kaiyom, Farzaan and Liu, Evan Zheran and Mai, Yifan and Liang, Percy and Hashimoto, Tatsunori},
  booktitle = {ICLR},
  year      = {2025}
}

@inproceedings{li2023camel,
  title     = {CAMEL: Communicative Agents for ``Mind'' Exploration of Large Language Model Society},
  author    = {Li, Guohao and Hammoud, Hasan Abed Al Kader and others},
  booktitle = {NeurIPS},
  year      = {2023}
}

@inproceedings{wang2025selfevolving,
  title     = {Benchmark Self-Evolving: A Multi-Agent Framework for Dynamic LLM Evaluation},
  author    = {Wang, Siyuan and Long, Zhuohan and Fan, Zhihao and Huang, Xuanjing and Wei, Zhongyu},
  booktitle = {COLING},
  year      = {2025}
}

@article{mitra2024agentinstruct,
  title   = {AgentInstruct: Toward Generative Teaching with Agentic Flows},
  author  = {Mitra, Arindam and others},
  journal = {arXiv preprint arXiv:2407.03502},
  year    = {2024}
}

% --- Cluster B: agent-built graph / KG ---
@inproceedings{lu2025karma,
  title     = {KARMA: Leveraging Multi-Agent LLMs for Automated Knowledge Graph Enrichment},
  author    = {Lu, Yuxing and Wu, Wei and Zhao, Xukai and Peng, Rui and Wang, Jinzhuo},
  booktitle = {NeurIPS (Spotlight)},
  year      = {2025}
}

@article{bian2025llmkg,
  title   = {LLM-empowered Knowledge Graph Construction: A Survey},
  author  = {Bian, Haonan},
  journal = {arXiv preprint arXiv:2510.20345},
  year    = {2025}
}

% --- Cluster C: multimodal graph benchmark ---
@article{wan2026openmag,
  title   = {OpenMAG: A Comprehensive Benchmark for Multimodal-Attributed Graph},
  author  = {Wan, Chenxi and Li, Xunkai and Zuo, Yilong and Deng, Haokun and Li, Sihan and Fan, Bowen and Qin, Hongchao and Li, Ronghua and Wang, Guoren},
  journal = {arXiv preprint arXiv:2602.05576},
  year    = {2026}
}

@inproceedings{yan2025magb,
  title     = {When Graph Meets Multimodal: Benchmarking and Meditating on Multimodal Attributed Graph Learning},
  author    = {Yan, Hao and Li, Chaozhuo and Yin, Jun and Yu, Zhigang and Han, Weihao and Li, Mingzheng and Zeng, Zhengxin and Sun, Hao and Wang, Senzhang},
  booktitle = {KDD (Datasets and Benchmarks)},
  year      = {2025},
  doi       = {10.1145/3711896.3737404}
}

@inproceedings{zhu2025mosaic,
  title     = {Mosaic of Modalities: A Comprehensive Benchmark for Multimodal Graph Learning},
  author    = {Zhu, Jing and Zhou, Yuhang and Qian, Shengyi and He, Zhongmou and Zhao, Tong and Shah, Neil and Koutra, Danai},
  booktitle = {CVPR},
  year      = {2025},
  note      = {Project: mm-graph-benchmark.github.io; = MM-Graph}
}

% --- Cluster D: benchmark leakage / contamination (LLM-level) ---
@inproceedings{wu2025antileak,
  title     = {AntiLeakBench: Preventing Data Contamination by Automatically Constructing Benchmarks with Updated Real-World Knowledge},
  author    = {Wu, Xiaobao and Pan, Liangming and Xie, Yuxi and others},
  booktitle = {ACL},
  year      = {2025}
}

@article{xu2024bdc,
  title   = {Benchmark Data Contamination of Large Language Models: A Survey},
  author  = {Xu, Cheng and Guan, Shuhao and Greene, Derek and Kechadi, M-Tahar},
  journal = {arXiv preprint arXiv:2406.04244},
  year    = {2024}
}

@inproceedings{sun2025emperor,
  title     = {The Emperor's New Clothes in Benchmarking? A Rigorous Examination of Mitigation Strategies for LLM Benchmark Data Contamination},
  author    = {Sun, Yifan and Wang, Han and Li, Dongbai and Wang, Gang and Zhang, Huan},
  booktitle = {ICML},
  year      = {2025}
}

@article{joeres2025datasail,
  title   = {Data Splitting to Avoid Information Leakage with DataSAIL},
  author  = {Joeres, Roman and Blumenthal, David B. and Kalinina, Olga V.},
  journal = {Nature Communications},
  volume  = {16},
  pages   = {3337},
  year    = {2025},
  doi     = {10.1038/s41467-025-58606-8}
}

% --- Cluster E (NEW): graph structural leakage (non-multimodal) ---
@article{platonov2023critical,
  title   = {A Critical Look at the Evaluation of {GNNs} under Heterophily},
  author  = {Platonov, Oleg and Kuznedelev, Denis and Diskin, Michael and Babenko, Alexey and Prokhorenkova, Liudmila},
  journal = {NeurIPS 2023; extended in TPAMI 2024},
  note    = {arXiv:2302.11640},
  year    = {2023}
}

@inproceedings{zhu2024pitfalls,
  title     = {Pitfalls in Link Prediction with Graph Neural Networks},
  author    = {Zhu, Yanqiao and others},
  booktitle = {WSDM},
  year      = {2024},
  note      = {arXiv:2306.00899}
}

@inproceedings{li2023lpeval,
  title     = {Evaluating {GNNs} for Link Prediction: Current Pitfalls and New Benchmarking},
  author    = {Li, Juanhui and others},
  booktitle = {NeurIPS Datasets and Benchmarks},
  year      = {2023}
}

@article{briere2025leakage,
  title   = {Benchmarking Data Leakage on Link Prediction in Biomedical Knowledge Graph Embeddings},
  author  = {Bri{\`e}re, others},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.01.23.634511}
}
```

---

## 7. 核查修订记录（2026-07-16 联网逐篇核实）

| 工作 | 原描述 | 核实后修正 | 证据 URL |
|---|---|---|---|
| BenchAgents | "无多模态" | ❌→✅ 实含语言+视觉模态；改为"多模态任务 benchmark，非 MAG 无图" | arxiv.org/abs/2410.22584 |
| Benchmark Self-Evolving | ACL 2025 | → **COLING 2025** | aclanthology.org/2025.coling-main.223 |
| KARMA | NeurIPS 2025 | → **NeurIPS 2025 Spotlight**（~55 引，更强对手） | arxiv.org/abs/2502.06472 |
| Bian LLM-KG survey | "Bian et al." / ICAIS 2025 | → 单作者 **Haonan Bian**；删 ICAIS（未证实） | arxiv.org/abs/2510.20345 |
| **MAGB** | WWW 2025 / ACM MM 2024（内部冲突） | → **KDD 2025 D&B**（两份旧文档都错） | dl.acm.org/doi/10.1145/3711896.3737404 |
| OpenMAG | arXiv:2602.05576 | ✅ 正确（2026-02-05）；19/6/16/24/8 全确认 | arxiv.org/abs/2602.05576 |
| **MM-Graph + Mosaic** | 列为两篇 | → **同一篇**（arXiv 2406.16321, CVPR 2025），数据集 7 | arxiv.org/abs/2406.16321 |
| Mosaic 泄露措辞 | "完全忽略泄露" | → 做了基础随机切分+边移除，无系统量化 | arxiv.org/abs/2406.16321 v2 |
| AntiLeakBench | 泛"2024" | → **ACL 2025**（arXiv 2412.13670） | arxiv.org/abs/2412.13670 |
| Emperor's New Clothes | ICML 2025 | ✅；补 arXiv 2503.16402 | arxiv.org/abs/2503.16402 |
| DataSAIL | "2024" | → **Nat. Commun. 2025**；措辞精确化（碰过 2D 实体对跨 split） | nature.com/articles/s41467-025-58606-8 |
| 🆕 Platonov / Zhu / Li / Brière | （原遗漏） | → **新增 §2.5 + 4 篇引用**（图结构性泄露 prior，必须主动区分） | 见 §6 |

---

## 8. 与其他文档的关系

| 文档 | 关系 |
|---|---|
| `related_work_agent_dataset_construction_2026-07-13.md` | 本文前身（调研底稿）；本文是其**去泄露后 + 联网核查 + 可贴论文版** |
| `paper_logic_DnB_FINAL_2026-07-11.md` | 论文逻辑线权威；本文执行其 §2 Related 定位 |
| `experiment_results_FINAL_decontaminated_2026-07-16.md` | 数字总表权威；本文所有数字以它为准 |
| `PPT_相关工作对比分析.md`（2026-03） | 🔴 定位过时（"学术论文垂直"），仅当历史参考 |
| `positioning_and_scalability_2026-06-23.md` | ⚠️ 论点部分废止，vs OpenMAG 对标表信息有效 |

---

*2026-07-16 联网核查修订。后续如 §2 正文或引用有更新，以本文为准；旧 `related_work_agent_dataset_construction_2026-07-13.md` 的元数据/数字若与之冲突，以本文为准。*
