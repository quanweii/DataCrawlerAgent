# P1 协议 — Exp3 / Exp4b / Exp5b（2026-07-31；环境就绪，仅 Exp3/4b 需真人）

> **本机实测（2026-07-31 纠正前述误判）**：`.env` 配齐；LLM 端点可达（dashscope.aliyuncs.com/Qwen → 200）；HF 经镜像可达（hf-mirror.com/modelscope.cn → 200）。仅直连 huggingface.co/github.com 超时（镜像全覆盖）。
> 故 **Exp5b 本机可跑**（经镜像）；**fresh agent 调用可跑**。**唯一真实约束**：Exp3/Exp4b 是 user study，需 3 名真人受试者（与网络/机器无关）。
> Exp5b 由 Claude 自治执行（见实验线 §20.2）；本文档聚焦 Exp3/Exp4b 招人协议 + Exp5b 的开放复现细节。

---

## Exp5b — OpenMAG / MAGB 真图 leak-profile（堵"leak-free 自证循环"）

**动机**：E1 只证 C2 **探测器**在 7 公开 PyG benchmark 有效，未对**同产物（MAG）**裁定。审稿人："你用自家测量定义 leak-free，又用自家测量证明 leak-free = 循环。" Exp5b 对现存 MAG benchmark（MAGB KDD'25 D&B / OpenMAG）直接跑 C2，对照自家 4 域。

**步骤**（开放网络机器）：
1. 下载 MAGB 图（HF `MAGBench/MAGB` 或论文 repo）+ OpenMAG 图（arXiv 2602.05576 repo）。MAGB 含 7 数据集（movie/book/music/...）。
2. 对每个图跑论文 C2 三件套（复用 `benchmark/data/splits.py`）：
   - `label_propagation_ceiling(y, ei_full, tr, te)` → structure-leakage ceiling (original)
   - `prune_cross_split_edges` 后再算 → ceiling (pruned)
   - label-field strip 检查（regex 恢复 label 字段，`diag_label_field_recover.py` 模板）
3. 对照：raw ceiling − majority = 结构泄露量；与自家 4 域（wikiart 0.921→0.024 等）+ E1 的 7 PyG benchmark 同表。
4. **预期**：MAGB/OpenMAG 若 ship 结构泄露 → raw ceiling ≫ majority（同 Amazon/Coauthor 模式）→ 证"领域金标准系统性漏掉结构泄露"对**同产物**也成立，堵循环指控。

**脚本入口**：模板 `benchmark/run_public_leakage_localnpz.py`（已支持 PyG npz）；为 MAGB 格式加一个 loader adapter（读其节点/边/label → 转 PyG `Data`）即可复用全部 C2 函数。产物：`benchmark/results/public_leakage_magb_openmag.json`。

**诚实 fallback**：若 MAGB 图本身已 pruned/无跨切分边（ceiling ≤ majority），则结论改为"现存 MAG benchmark 已无结构泄露，本框架价值在前置 leak-profile + 自治 + 模态"，仍可防守（不依赖"别人都漏"）。

---

## Exp3 — 真 human baseline（替换 E2"专家估计"）

**动机**：E2 的"agent ~3s vs 专家 25–90 min"是**专家估计非 user study**——审稿人必质疑。Exp3 给真实 n=3×2 计时 + 产出质量对比。

**受试者**：3 名懂 GNN/图学习的研究生（**不接触本框架**），各自独立。

**任务（每人 2 个 held-out 原始源）**：给受试者一份原始数据样本（如 amazon 5438 商品 raw json、reddit 7257 帖 raw json）+ 任务说明（"为这个源设计一个 MAG benchmark：选一个 NC label + 枚举关系 + 标注哪些关系对 label 是定义性的需剔除"），要求产出一份 schema（YAML 或表格）+ 说明。**计时**从拿到数据到交 schema（含查定义性）。

**记录**：每人每域 wall time、产出 schema（label/relations/去泄露判断）、是否需要查文档/试错。

**对照 agent**（已在盘 `e4_scale.json`）：agent discovered schema + wall ~3s + tokens ~2200。

**评测**（双盲）：
- **效率**：human 中位时间 vs agent ~3s（报告比值 + 范围）。
- **label 一致性**：human 选的 label vs agent discovered label（amazon→main_cat / reddit→subreddit）一致率。
- **质量**：对 human 与 agent 各自的 schema，建图 + 跑报告卡（`generate_report_card.py`）→ 比 correctness（去泄露是否正确）+ quality（label 是否退化）。
- **诚实 caveat 写进论文**：n=3 非大样本；受试者水平差异；agent 此处只测 schema 发现（不含双方共享的编码+跑批）。

**物料**：`docs/p1_subject_task_sheet.md`（受试者任务卡，待写：2 域 raw 样本 + 指示 + 交付模板）+ `benchmark/results/exp3_human_baseline.json`（结果模板：per-subject per-domain time/schema/report-card）。

---

## Exp4b — 外部用户复用（"开源快速复用"金标准）

**动机**：C4 复用证据目前是 spotify 0-LOC（自测）+ SLOC manifest（静态）。审稿人要**外部用户**真复用证据。

**步骤**：Exp3 的 3 名受试者，任务结束后给本框架（README + 1 个零代码 YAML 范例 spotify）+ 1 个新原始源（如 Goodreads 书评 / 某新 API），要求用框架造出 MAG。**计时** + 记录：是否成功（build_valid+decontam）、wall time、是否写 Python（LOC）、产出报告卡质量。

**对照**：受试者 Exp3 手建时间 vs Exp4b 框架复用时间（同一个人的 speedup，配对）→ "框架省时省力"的金标准证据。

**产物**：`benchmark/results/exp4b_external_user_reuse.json`（per-subject：domain/time/LOC/report-card/success）。

---

## 跑完后的回填清单
- [ ] Exp5b 结果 → 实验线 §20.2 补 MAGB/OpenMAG leak-profile 表 + 逻辑线 §4.5「leak-free 靠 C2」行。
- [ ] Exp3 结果 → 实验线 §18.2（E2 成本表）补 human baseline 列 + 逻辑线 §4.5「E2 专家估计」行。
- [ ] Exp4b 结果 → 实验线 §20 补外部用户复用 + 逻辑线 C4 证据。
- [ ] 三项完成后，P1 全清，paper §6（复用性）+ §8（limitation：n=3/受试者）有实证。

> 优先级：**Exp5b > Exp3 > Exp4b**（Exp5b 堵最大指控"循环去泄露"；Exp3 次之；Exp4b 锦上添花）。
