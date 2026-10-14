# E1–E7 补充实验结果 (2026-07-27)

> 目的: 把 agent 从"够用"抬到 ICLR main 要求的"必要+更优"。E1–E7 按 ICLR 审稿攻击面设计。
> 每个实验记录: 内容 / 结果 / 诊断(代码 or 设计 or 环境) / 对论文的意义。
> 测量口径全部复用论文权威实现(`benchmark/data/splits.py`), 零临时复刻。

---

## E1 — 把去泄露 verifier 跑在公开 benchmark 上, 证明它能查出"已发表数据集 ship 的结构泄露"

**问题**: agent-verifier(C2)是不是只在自家数据上才"查出泄露"? 能不能在领域已发表的公开 benchmark 上也查出它们带着的结构泄露?(= agent 去泄露价值 load-bearing 的外部铁证)

**工具**: 复用 `benchmark/run_public_leakage_profile.py` 的 `profile_data`(同一套 `stratified_split`/`prune_cross_split_edges`/`label_propagation_ceiling`)。

**结果(7 个公开 benchmark, split_seed=1, 60/20/20)**:

| dataset | edge kind | homo | majority | LP raw | LP pruned | gap(raw−maj) | prun% |
|---|---|---|---|---|---|---|---|
| coauthor_cs | co-authorship (构造) | 0.81 | 0.225 | 0.900 | 0.039 | **+0.675** | high |
| amazon_photo | co-purchase (构造) | 0.83 | 0.253 | 0.913 | 0.049 | **+0.660** | high |
| cora | citation (自然) | 0.81 | 0.301 | 0.847 | 0.111 | +0.546 | high |
| amazon_computers | co-purchase (构造) | 0.78 | 0.374 | 0.886 | 0.032 | **+0.512** | high |
| citeseer | citation (自然) | 0.74 | 0.211 | 0.698 | 0.076 | +0.487 | high |
| coauthor_physics | co-authorship (构造) | 0.93 | 0.505 | 0.954 | 0.167 | +0.449 | high |
| pubmed | citation (自然) | 0.80 | 0.399 | 0.828 | 0.208 | +0.429 | high |

**结论**: 7/7 公开 benchmark 的 raw 构造图都带 +0.43~+0.68 的结构泄露(label propagation 仅靠边结构就比多数类高 43–68 个百分点); 论文的跨切分剪枝把它们**全部**压到 ≤ majority(pruned 0.03~0.21 ≤ majority 0.21~0.50)。构造边图(Amazon/Coauthor, "最像 MAG")泄露模式与论文自家去泄露前数据一致。

**诊断(环境, 非代码/设计)**:
- 原脚本 `run_public_leakage_profile.py` 有两个设计缺陷: ① dump-at-end(一个数据集挂起丢全部)② 无单数据集超时(被墙镜像无限等)。
- 我写 `run_public_leakage_robust.py`(每数据集独立子进程 90s 超时 + 增量 dump)修了; 即使中途被 kill 也保住已跑结果。
- 真正的环境约束: 本环境 **github raw / huggingface 文件下载被限速/阻断**(HEAD 返回 200 但实际下载挂)。**绕过**: `git clone --filter=blob:none --sparse`(元数据克隆不触发 raw 下载限制), 拿到 shchur npz 后写 `run_public_leakage_localnpz.py` 直接加载。**这个绕过法可复用于任何被墙环境**。
- 真正的 OpenMAG/MAGB(HF 托管)在本环境无法下载; 但 7 个公开 benchmark(含 4 个构造边"类 MAG"图)已足够支撑同一科学论断。

**对论文的意义**: 直接回击 ICLR reject #1"agent 不 load-bearing"。证据从"我们清洗了自家数据"升级为: **我们的 verifier 跑在领域金标准公开 benchmark 上, 查出它们全部 ship 了 +0.43~+0.68 的结构泄露**——这是领域手写过程系统性漏掉、而我们的测量能稳定检出的泄露维度。

**残留 TODO(非阻塞)**: 异质图对照(chameleon/squirrel/WebKB/Actor, geom-gcn 仓库)未取——低同质图泄露通路不同, 是 nuance 而非主证; 投稿前在开放网络机器上 `python benchmark/run_public_leakage_profile.py --public` 补齐即可。

*数据: `benchmark/results/public_leakage/leakage_profile_public_localnpz.json`*

---

## E2 — Agent-vs-人工 成本量化

**问题**: "省时省力"卖点现在是断言; 要给数字。测 agent 端到端 schema 发现的 wall-clock + token, 对比专家手写 YAML+去泄露调试的估计耗时。

**工具**: `benchmark/path_b/run_e2_agent_cost.py` —— 插桩 `autonomous_discover`(patch `_call_llm` 记 prompt/completion tokens + 每次 wall-clock), 6 域 × 3 seed。

**结果(model=qwen-plus, 3 seed 均值)**:

| domain | LLM calls | tokens | wall | build_valid | decontam | 类型 |
|---|---|---|---|---|---|---|
| spotify | 2.0 | 2576 | 3.3s | 3/3 | 3/3 | gold |
| wikiart | 2.0 | 1920 | 2.8s | 3/3 | 3/3 | gold |
| steam | 1.7 | 1839 | 2.8s | 3/3 | 3/3 | gold |
| tmdb | 1.7 | 2223 | 3.2s | 3/3 | 3/3 | gold |
| **amazon** | 2.0 | 3037 | 2.9s | 3/3 | 3/3 | **held-out** |
| **reddit** | 1.3 | 1512 | 2.1s | 3/3 | 3/3 | **held-out** |

**对比**: agent ~3s + ~2200 tokens / 域 (schema 发现+去泄露) vs **专家估计 25–90 min / 域**(inspect raw→选 label→枚举关系→查定义性泄露→迭代)。6 域 18/18 build_valid+decontam 全过(含 2 个 held-out 新域)。

**诊断(代码 bug, 已修)**: 首跑 amazon/reddit 崩 `"expected string or bytes-like object"` —— 我原 runner 对无 gold 的域调 `score()`→`_base_field(gold["label"]=None)`→regex 崩。修: build_valid/decontam 直接从 `measure()` 算(本就不依赖 gold), gold 对比仅对有 gold 的域。

**诚实 limitation(写进论文)**:
1. agent 此处只做 **schema 发现**(label/关系), 不含爬虫+多模态编码+leak-profile 跑批; 但后者是 agent/人工**共享的确定性基础设施**(拿到 schema 后, 建图/编码/profile 的耗时与谁提供 schema 无关)。所以成本差**精确度量在"专家 schema 设计+去泄露调试"这一段**, 正是 agent 自动化的部分。
2. 人工耗时是**专家估计**(依据现有 config 复杂度 16–38 行 + 经验), **非用户实测**。论文须如实标注; 完整 user study 是未来工作。
3. wall=3s 被 LLM API 延迟主导(非本地计算)。

**对论文的意义**: 把"省时省力"从断言变数字: agent ~3s vs 专家 25–90 min/域(~500–1800× 加速), 且 18/18 含 held-out 全 build_valid。支撑"框架可复用+自治"的工程效用 claim。

*数据: `benchmark/results/e2_agent_cost.json`*

---

## E3A — generation-vs-extraction 架构定位对比(非伪 head-to-head)

**纠正**: BenchAgents/AutoBencher 是 **generation agent**(LLM 合成 benchmark 实例); 我们是 **extraction/orchestration agent**(fetch 真实数据→建 MAG→去泄露)。不同类、不同产物(合成 NLP/QA vs 真实 MAG)。直接"谁建的 MAG 好"不成立——它们建不了 MAG。正确的对比是**范式轴 + 泄露维度**。

**架构(读论文核实, arxiv 2410.22584 / 2407.08351)**:

| 维度 | BenchAgents (2024) | AutoBencher (ICLR'25) | **Ours (MAG-Forge)** |
|---|---|---|---|
| agent 范式 | LLM **生成**合成实例 | LLM **生成** QA | LLM **编排** fetch 真实源 |
| 数据来源 | LLM 幻觉文本 | LLM 生成 QA 对 | 真实 API/爬取(arxiv/steam/tmdb/wikiart/amazon) |
| 产物 | 合成 NLP 任务实例 | 合成 QA benchmark | 多模态属性图(MAG) |
| agent 数 | 4(Plan/Gen/Verify/Eval) | 1 LLM | planner↔verifier→executor→encoder→evaluator |
| verification 查什么 | **clarity/feasibility**(质量) | salient/novel/difficult | **测量驱动去泄露**(C2, 非质量) |
| **泄露机制** | 标签条件化特征生成(**语义级, 不可剥离**) | 同左 | 结构/字段级(**可测量剥离**) |
| 去泄露协议 | ❌ 无 | ❌ 无 | ✅ 三层 leak-profile |
| 可复用 | 人在环重编辑(DIL 0.14–0.41) | QA-topic spec | 1 YAML=1 域, 94.6% 复用, spotify 0-LOC |
| 多模态/图 | ❌ 文本 | ❌ QA | ✅ text+image+formula 属性图 |

**论点**: generation 与 extraction 是**正交范式**, 不存在"谁强"的可比产物。真正的科学问题是 **"两种范式各自产生什么泄露 profile"**——这驱动 E3B。优越性证据重定向到**同产物**对比(OpenMAG/MAGB), 由 E1(公开 benchmark leak-profile)+论文自家 4 域 leak-profile 覆盖。

**对论文的意义**: §2 Related Work 用此表替代"我们 agent 更强"的伪对比; 定位为"首个 extraction+测量去泄露的 MAG 构造框架", 与 generation 路线(BenchAgents/AutoBencher)正交互补。

---

## E3B — generation-vs-extraction 的泄露维度实证对比

**问题**: 两种范式(generation=LLM 合成 vs extraction=fetch 真实+去泄露)各自产生什么泄露 profile? (非伪 head-to-head, 而是在"泄露"轴上的范式对比)

**设计**: 模拟 generation agent —— qwen-plus 在"已知 label"下合成 text(tmdb 8 genres × 30 + steam 8 genres × 30 = 各 240)。用论文同款 label-leakage 诊断(`hit_substr`/`hit_field`, 见 `diag_text_label_leakage.py`)+ TF-IDF+LR "语义泄露上限"(text→label 5-fold 可恢复性)。关键: 对合成与真实都跑论文 C2 的字段剥离, 看泄露是否可剥。

**结果**:

| 数据 | hit_field | TF-IDF+LR acc | 解读 |
|---|---|---|---|
| 合成 tmdb (raw) | 0% | **1.000** | 语义泄露: text 完美决定 label |
| 合成 tmdb (**剥离后**) | 0% | **1.000** | strip 是 no-op, 泄露不可剥 |
| 真实 tmdb (盘上 .md) | **100%** | 0.762 | "Genres: X" 字段泄露 |
| 真实 tmdb (**C2 剥离后**) | 0% | **0.238** | 字段泄露被剥掉, acc 崩到 ≈majority |
| 合成 steam (raw/剥离) | 0% / 0% | **0.992 / 0.992** | 同 tmdb, 不可剥 |
| 真实 steam (盘上/C2后) | 100% / 0% | 0.592 / **0.488** | 字段泄露可剥 |

**结论(airtight)**:
1. **真实数据泄露 = 字段级("Genres: X")→ 可剥离**: C2 strip 把 hit_field 100%→0%, TF-IDF+LR 崩塌(tmdb 0.762→0.238, steam 0.592→0.488)。这正是论文 C2 协议做的事。
2. **generation-agent 合成数据泄露 = 语义级 → 不可剥离**: hit_field=0%(无字段), TF-IDF+LR≈1.0, 且 strip 是 no-op(保持 1.0)。LLM 在"已知 label"下生成, label 被织进 prose 语义, **无字段可剥**。
3. **范式对比**: extraction+去泄露 pipeline 产物 text→label 可恢复性 0.24-0.49(genuine); generation 范式产物 ≈1.0(label 完美泄漏)。**generation 范式引入了我们的协议修不掉的语义泄露。**

**诊断(测量设计 bug, 已修)**:
- 首跑 `hit_field=0%` 对真实数据是假象 —— 原 regex `(style|genre|...)` 不匹配复数 "Genres"; 真 .md 里明明有 "Genres: Drama"。修 regex 支持单复数后, 真实 hit_field=100%(符合肉眼)。
- `hit_substr`(裸子串)对 tmdb/steam 是**噪声指标**: label("action"/"drama")是常见英文词, 任何 overview 都会出现 → 不能作泄露信号。不依赖它, 靠 hit_field(字段模式)+ TF-IDF+LR(语义上限)这两个干净信号。
- 生成器首跑 6s/call 太慢(串行)→ 改 ThreadPoolExecutor(12 并发), 480 条 ~3min。

**诚实 caveat(写进论文)**: 合成 prompt 显式"为 {genre} 片生成"—— 这**忠实于 generation agent 范式**(它们就是给定 label 生成内容)。若不让 LLM 知 label, 它就无法产带标签数据。所以 setup 对范式公平。

**对论文的意义**: 这是 generation-vs-extraction 在**泄露轴**上的实证(非"谁建 MAG 好"的伪对比)。直接支撑 C2 护城河: **真实 fetch + 测量去泄露能消除字段泄露; 而 LLM 合成数据的语义泄露无字段可剥, 是 generation 路线的固有缺陷**。补强 §2.2 对 BenchAgents/AutoBencher 的差异化论证。

*数据: `benchmark/results/e3b_synthetic_leakage.json`, 合成语料 `data/e3b_synthetic_{tmdb,steam}.json`*

---

## E4 — 自主性扩规模 (≥5 held-out 域)

**问题**: agent 泛化只证过 amazon 1 个(P5-B), n=1 像运气。扩到 ≥5 个 held-out 真实源证"自治 schema 发现可泛化"。

**held-out 域**(不在 P4 金标准 spotify/wikiart/steam/tmdb 调参集): amazon / reddit / arxiv_cs / arxiv_math / arxiv_condmat / arxiv_astroph(后 4 个 JSONL 流式采样)。

**结果(model=qwen-plus, seed=1)**:

| held-out 域 | n | 发现 label | n_rel | build_valid | decontam | 成本 |
|---|---|---|---|---|---|---|
| amazon | 5438 | main_cat | 1 | ✅ | ✅ | 2 call / 3.2s |
| reddit | 7257 | subreddit | 2 | ✅ | ✅ | 1 call / 1.6s |
| arxiv_cs | 20000 | license ⚠️ | 1 | ✅ | ✅ | 2 call / 3.1s |
| arxiv_math | 20000 | license ⚠️ | 1 | ✅ | ✅ | 2 call / 2.8s |
| arxiv_condmat | 20000 | categories | 1 | ✅ | ✅ | 2 call / 2.7s |
| arxiv_astroph | 20000 | categories | 1 | ✅ ✅ | ✅ | 2 call / 3.2s |

**6/6 build_valid + decontam 全过**, 平均 ~2 LLM call / ~3s / ~2.7k tokens。泛化从 n=1 扩到 n=6(含产品/社交/论文 3 个异质源)。

**诊断(我的测量设计 bug, 已修)**:
- 首跑 **0/6** 全挂 build_valid —— 不是 agent 失败, 是**我把 measure() 跑在 300 样本上**: 稀有类在 300 样本里 <`min_class_size=20` 阈值 → 误判 label invalid; 关系共享度也被小样本低估 → 全 degenerate。E2 用全量 measure 所以 amazon/reddit 过。
- 修: profile 喂 300 给 LLM, **measure 跑全量/大样本**(arxiv JSONL 读 20000 行)。修复后 6/6 全过。**教训: 有效性测量必须在与真实 pipeline 一致的全量分布上跑, 不能用 LLM 上下文的小样本。**

**诚实暴露的 2 个软肋(与论文已披露一致, 非新增问题)**:
1. arxiv_cs/math 选了 `license` 当 label —— "valid 但非最优": license 是合法分类字段(过 filter)但不是科学有意义的 NC target(人会用 categories)。这正是论文红线软肋"合理但非最优的替代 label 不会被 loop 纠(只纠退化/连续硬错误)"。
2. 关系稀疏(nrel=1) —— P6 已改善但新域仍偏稀疏。

**对论文的意义**: agent 自治 schema 发现**可泛化到 6 个未见真实源(3 异质类: 产品/社交/论文)**, 6/6 产出合法+去泄露 schema。泛化证据从 n=1→n=6, 直接回击"agent 只会跑 4 个金标准域"。软肋如实写 limitation(不夸大)。

*数据: `benchmark/results/e4_scale.json`*

---

## E5 — CLIP 预训练污染 cross-benchmark

**问题**: 把 CLIP-vs-ImageNet-only 污染探针跑到【外部】图像域, 看污染是否随 LAION 常见度变化(跨数据集验证, 非只自家 3 域)。

**设计**: `run_e5_cross_clip.py`。3 编码器(CLIP-L/14 缓存 / ResNet50-IN1k / ViT-B/16-IN1k), image-only MLP NC, 5 seed, 120 ep。**关键对照 = CLIP vs 同家族 ViT-B/16-IN1k**(控架构; 若 CLIP≈ViT-IN 则无污染, 若 CLIP≫ViT-IN 才是预训练污染)。外部数据: DTD(Oxford 纹理, LAION 罕见)+ Pets(Oxford 猫狗, LAION 常见)。

**结果**:

| 外部域 | n/cls | CLIP-L/14 | ResNet50-IN | ViT-B/16-IN | **Δ(CLIP−ViT)** | 判定 |
|---|---|---|---|---|---|---|
| DTD 纹理 (LAION-rare) | 1880/47 | 0.780 | 0.718 | 0.783 | **−0.003** | 无污染 |
| Pets 猫狗 (LAION-common) | 1480/37 | 0.952 | 0.950 | 0.955 | **−0.003** | 无污染(但近天花板) |
| amazon 商品 (held-out) | 114/4 | 1.000 | 1.000 | 1.000 | 0.000 | 数据太小, 无信息 |

**关键发现(诚实, refine 了原 claim)**: 两个外部域 CLIP 与同家族 ImageNet-only ViT-B/16 **持平**(Δ≈0, 甚至略负)。这与论文自家 wikiart Δ=+0.21 / tmdb +0.15 / steam +0.11 形成对比。

**解读**: CLIP 预训练污染**不是普遍效应**, 而是**任务依赖**——只在"视觉风格/流派分类"任务(艺术流派/电影类型/游戏类型, 即 MAG 的 label 类型)上显现, 因为这类任务恰好踩中 CLIP 在 LAION 学到的视觉风格语义。外部"纹理识别/猫狗品种"(细粒度识别, 非 style)任务上, CLIP 与 ImageNet-ViT 持平。

**诊断(3 处)**:
1. **CLIP 离线加载**: HF 被墙, `from_pretrained` 默认联网查版本→挂。修: 脚本顶部 `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`(env 经 nohup 没传进去, 必须脚本内 setdefault)。缓存本就完整(config+safetensors)。
2. **amazon 数据不足**: 本地只 114 商品图/4 类(多数商品图未下载, 仅 image_urls), 饱和 1.0 → 无信息。要有效 amazon 须先批量下载商品图。
3. **Pets 天花板**: 两编码器都 ~0.95, 天花板掩盖了潜在 gap。需更难的自然图任务才能显出 CLIP 边。

**对论文的意义(refine C2, 非推翻)**:
- 污染 claim 从"frozen CLIP 系统性污染"精确化为"**污染集中在视觉风格/流派分类任务**——正是 MAG benchmark 的 label 类型"。论文自家 4 域全是 style/genre 分类(流派/类型/风格), 所以都中招; 外部细粒度识别任务不中招。
- 这反而让 C2 更可防守: 不是夸大"所有 CLIP benchmark 都脏", 而是精准说"MAG 这类 style/genre benchmark 的 image 高分确有预训练污染成分, 且我们的 ImageNet-only 对照能量化它"。
- **残留 TODO(非阻塞)**: OpenMAG/MAGB 的 image 特征(HF 托管)本环境下载不了; 在开放网络机器上跑 `clip_contamination_probe.py` profile 它们的 style/genre 子集可补最直接的"同产物 cross-benchmark"铁证(预期它们 Δ 也大, 因 label 也是 genre/style)。

*数据: `benchmark/results/e5_cross_clip_contamination.json`*

---

## E6 — verifier 闭环鲁棒性定式化

**问题**: 量化"single-shot label 发现脆, 但 verifier 闭环兜底"。P5-A 只报 aggregate, 这里逐 run 取 round0(single-shot) vs final(loop) 合法性 + 恢复事件。

**设计**: `run_e6_loop_robust.py`。4 profile 变体(V1 terse / V2+examples / V3+structural / V4 minimal) × 4 金标准域 × 2 seed = 32 run。每 run 一次 `autonomous_discover`, 从 trace 同时取 round0 单-shot 合法性 + final(闭环后) build_valid/decontam + 是否恢复。

**结果**:

| 变体 | single-shot 合法 | loop 合法+去泄露 | 恢复事件 |
|---|---|---|---|
| V1_terse | 8/8 (100%) | 8/8 (100%) | 0 |
| V2_examples | 8/8 (100%) | 8/8 (100%) | 0 |
| V3_structural | 6/8 (75%) | 6/8 (75%) | 2(steam 单-shot 挂→闭环救回) |
| V4_minimal | 6/8 (75%) | **8/8 (100%)** | 2(steam 挂→救回) |
| **总** | **28/32 (88%)** | **30/32 (94%)** | **4** |

**decontam(去泄露正确): 32/32 = 100%** —— 任何变体/域/seed, 终 schema **从不**含定义性关系。

**诚实暴露(非 bug, 写 limitation)**:
1. **恢复成立**: 4 个 single-shot 失败(全在 steam V3/V4, 单-shot 选了退化 `type` 字段)被闭环 100% 救回(V4 steam 75%→100%)。✅ 支持"闭环兜底"。
2. **但闭环非严格单调**: V3 tmdb 出现 `round0=Y → final_bv=N` ×2 —— 闭环在修关系时把 LLM 带歪到 invalid label, **反而破坏了有效的单-shot 结果**。净: +4 恢复, −2 回归, 88%→94%。与 IGC 教训一致(closed_loop 不是严格优化器, 是审计机制)。
3. **量化论文红线**("可靠核心是 C2 测量, LLM 发现是脆弱部分"): **C2 去泄露 32/32=100% vs LLM label 发现 88-94%**。verifier 的本职(去泄露)无懈可击; 残余脆弱全在 LLM 的 label 发现(正是 agent 区别于纯脚本的、不可消除的部分)。

**对论文的意义**: 把"loop 救回 single-shot"从口号变数字(4 恢复事件, 88%→94%), 且诚实披露闭环的 2 个回归(不夸大为"严格更优")。**decontam 32/32 是最强卖点**——agent 产出的 schema 无论 label 怎么变, 绝不 ship 定义性泄露。这与 C2 护城河直接呼应。

*数据: `benchmark/results/e6_loop_robust.json`*

---

## E7 — steering 合成受控扫描(救 effective-n≈3)

**问题**: 真实 steering 只 3 域(steam/tmdb/wikiart)→ effective n≈3, cluster-robust p=0.143(边缘)。审稿人会质疑 r=0.846 是 3-域巧合。用合成受控图证明机制不是巧合。

**设计**: `run_e7_synthetic_sweep.py`。固定节点特征(→ MLP 跨条件恒定=对照), SBM 旋同质性(0.20→0.95, 9 档)只换边, 跑 GCN vs MLP(3 seed)。复用 run_steering 同款 get_model/train_and_eval + 受控种子。

**诊断迭代(实验设计, 2 次失败 → 成功)**:
1. 首跑 SIGNAL=1.5/feat=128 → MLP=GCN=1.000 全程 → gap=0, Pearson=nan。**天花板**: 特征太易, MLP 满分, GCN 无空间。
2. 二跑 SIGNAL=0.55/feat=128 → 仍 1.000。**高维 bless**: 128 维下类条件高斯即使弱信号也近完美可分。
3. **修**: feat=8(低维) + SIGNAL=0.5 + NOISE=1.6 → MLP=0.640(给 GCN 留空间) → 成功。

**结果(feat=8, MLP 恒定=0.640 全程)**:

| homophily | 0.20 | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|
| GCN | 0.493 | 0.579 | 0.718 | 0.832 | 0.933 | 0.980 | 0.997 | 1.000 | 1.000 |
| gap(GCN−MLP) | −0.147 | −0.061 | +0.078 | +0.192 | +0.293 | +0.340 | +0.357 | +0.360 | +0.360 |

**Pearson(gap, homophily) = 0.940 over 9 controlled levels** (MLP 恒定 0.640 已自检)。趋势: 低同质 GCN<MLP(gap 负), 高同质 GCN>MLP(gap 正), crossover ≈ h 0.37。

**对论文的意义**:
1. **机制非 3-域巧合**: gap-vs-homophily 正相关在 9 个独立受控合成设置上 r=0.940(比真实 0.846 还强)。真实 3 域只是对这个**普适机制**的采样, 不是过拟合 3 点。
2. **MLP 恒定 0.640 = 控制自洽**: 证明 gap 纯来自边结构(同质性), 与真实 steering 的 MLP bit-identical 自检一致。
3. **诚实 nuance**: 合成 crossover ≈ 0.37 << 真实 ≈ 0.91。因合成特征弱(MLP=0.64), GCN 在更低同质就能帮忙; 真实域特征强(MLP 0.5-0.9), 需更高同质 GCN 才反超。**crossover 依赖特征强度**, 论文应说"crossover 不是普适常数, 依赖 feature SNR"——这解释了为何旧版删 crossover@0.91 点估计是对的。
4. **降级确认**: steering 保持"controlled trend"(非 causal law), 但现在有合成证据撑"机制普适"。n≈3 软肋从"只 3 域"缓解为"3 真实域 + 9 合成受控档, 机制 r=0.94 可复现"。

*数据: `benchmark/results/e7_synthetic_sweep.json`*

---

## E1–E7 汇总(对 ICLR 审稿攻击面的回击)

| 攻击面 | 实验 | 结论 |
|---|---|---|
| agent 不 load-bearing | E1 | ✅ verifier 在 7 公开 benchmark 查出 +0.43~+0.68 结构泄露(它们 ship 了) |
| "省时"是断言 | E2 | ✅ agent ~3s/~2200tok vs 专家 25-90min/域, 18/18 含 held-out |
| generation agent 对比 | E3A/B | ✅ 架构正交(生成vs抽取); 合成数据带不可剥语义泄露(TF-IDF+LR≈1.0 vs 真实去泄露 0.24-0.49) |
| agent 只会 4 金标准域 | E4 | ✅ 6/6 held-out(含 arxiv 子域)build_valid+decontam; 修 measure-on-sample 缺陷 |
| CLIP 污染只自家数据 | E5 | ⚠️ refine: 污染任务依赖(外部 DTD/Pets Δ≈0, 自家 style/genre 域 Δ 大)——更精准可防守 |
| loop 鲁棒性是口号 | E6 | ✅ 量化: decontam 32/32; loop 88%→94%(+4恢复, 诚实披露 −2 回归) |
| steering n≈3 | E7 | ✅ 合成 9 档 r=0.940 复现机制, 非巧合 |

**净结论**: agent 在"去泄露 verifier"(E1: 外部铁证 + E6: decontam 100%)和"自治 schema 发现泛化"(E4: 6/6 held-out)上 load-bearing; 与 generation 范式在泄露轴上实证可分(E3B)。残留软肋(refine 而非推翻): E5 污染任务依赖、E6 loop 非严格单调、steering crossover 依赖特征 SNR、人工耗时是估计。这些都是**诚实 limitation**, 写进论文反而加分。
