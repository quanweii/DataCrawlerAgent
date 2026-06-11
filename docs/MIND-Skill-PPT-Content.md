# MIND-Skill 论文讲解 PPT 内容稿
## Quality-Guaranteed Skill Generation via Multi-Agent Induction and Deduction

> 演讲时长：20-25分钟
> 论文来源：arXiv:2605.08670 (2026年5月)
> 作者：Yixuan Li, Mingshu Cai, Ziyang Xiao, Wanyuan Wang, Yanchen Deng, Bo An
> 单位：南洋理工大学, 早稻田大学, 浙江大学, 东南大学

---

## Slide 1: 封面

**标题：** MIND-Skill: 基于多智能体归纳与演绎的质量保证技能生成

**副标题：** Quality-Guaranteed Skill Generation via Multi-Agent Induction and Deduction

**演讲人信息 + 日期**

---

## Slide 2: 目录 / 大纲

1. **背景与动机** — 为什么需要自动生成Agent技能？
2. **核心问题** — 现有方法的三大缺陷
3. **MIND-Skill 框架** — 归纳智能体 + 演绎智能体
4. **三大文本损失函数** — 重构损失、结果损失、评分损失
5. **闭环优化** — TextGrad 驱动的迭代优化
6. **实验结果** — AppWorld & BFCL-v3
7. **消融实验与深入分析** — 每个组件的贡献
8. **案例研究** — 弱模型为何能匹敌强模型
9. **实践落地** — 如何在我的 DataCrawlerAgent 项目上构建技能系统
10. **总结与展望**

---

## Slide 3: 背景 — AI Agent 与技能（Skills）

**讲解要点（约1.5分钟）：**

LLM 驱动的 AI 智能体（Agent）已经成为自主问题求解的重要范式。但 Agent 在处理复杂的多步骤真实任务时仍然面临巨大挑战。

**什么是 Agent 技能（Skill）？**

- 技能是**可复用的程序性知识文档**，将成功的解题策略和标准操作流程封装为结构化的 Markdown 文档 + 脚本
- 类比：就像人类专家把自己的经验写成操作手册，其他人照着手册就能完成类似任务
- 与普通"工具调用"的关键区别：工具是"单个动作"（如调用一个API），技能是"一整套流程"（如"如何用分页API获取所有数据"）

**技能的生命周期：**
```
表示(Representation) → 获取(Acquisition) → 检索(Retrieval) → 演化(Evolution)
```

**举例说明：**
- ❌ 工具层面：`download_pdf(arxiv_id)` — 只是一个函数调用
- ✅ 技能层面：完整的"arXiv论文批量下载与多模态提取"流程 — 包含错误重试、GPU分配、结果验证等策略

---

## Slide 4: 核心问题 — 现有技能生成方法的三大缺陷

**讲解要点（约1.5分钟）：**

现有技能生成方法（零样本生成、轨迹蒸馏、终身演化）都存在**质量无法保证**的根本问题：

### 缺陷 1：缺少闭环验证
> 直接从任务描述或轨迹生成技能，**没有验证生成的技能是否真的有用**
>
> 类比：写了一份菜谱，但从没实际炒过菜验证菜谱对不对

### 缺陷 2：忽视文档质量
> 生成的技能文档通常**结构混乱、冗长或缺失关键信息**
>
> 技能是要给其他 Agent 甚至人类使用的，文档质量直接决定可用性

### 缺陷 3：抽象过程忠实度未验证
> 从执行轨迹蒸馏技能是**有损压缩**，可能过度泛化导致关键细节丢失
>
> 类比：你做了一个实验记录，但写总结时把关键步骤漏掉了，别人照着总结做就会失败

**论文引用数据（SkillsBench）支持：**
- 零样本生成的技能**平均无益**（no benefit on average）
- 低质量技能甚至会**显著降低** Agent 性能
- 只有精心编写的人工技能才能稳定优于无技能基线

---

## Slide 5: MIND-Skill 核心思想 — 一句话总结

**核心公式：**

$$P_I^* = \arg\min_{P_I} \left( L_{\text{outcome}}(\hat{\tau}, t),\ L_{\text{recon}}(\tau, \hat{\tau}, t),\ L_{\text{rubric}}(s, t) \right)$$

**约束条件：**
$$s = A_I(t, \tau; P_I), \quad \hat{\tau} = A_D(t, s; P_D)$$

**用人话说：**
1. **归纳智能体（Induction Agent）**：从成功的执行轨迹中提取可复用技能
2. **演绎智能体（Deduction Agent）**：只靠这个技能，尝试重新执行任务
3. **如果演绎智能体成功了** → 说明技能质量好
4. **如果演绎智能体失败了** → 反馈给归纳智能体改进技能
5. **反复迭代直到收敛** → 得到高质量技能

**关键设计决策：** 演绎智能体的 Prompt **冻结不变**，确保它除了技能之外得不到任何额外指导。

---

## Slide 6: 框架总览图 — Figure 1 详解

**【展示论文 Figure 1】**

**讲解要点（约2分钟）：**

整个框架的工作流如下：

```
输入轨迹 τ → [归纳智能体 A_I] → 生成技能 SKILL.md
                                            ↓
                                    [演绎智能体 A_D] → 重构轨迹 τ̂
                                            ↓
                            ┌───────────────┼───────────────┐
                            ↓               ↓               ↓
                      重构损失        结果损失        评分损失
                     L_recon        L_outcome       L_rubric
                            ↓               ↓               ↓
                            └───────────────┼───────────────┘
                                            ↓
                                    [TextGrad 优化器]
                                            ↓
                                   更新归纳智能体的 Prompt P_I
```

**四个关键组件：**

| 组件 | 作用 | 是否优化 |
|------|------|---------|
| 归纳智能体 A_I | 从轨迹提取技能 | ✅ 优化其 Prompt P_I |
| 演绎智能体 A_D | 靠技能重构轨迹 | ❌ 冻结 Prompt P_D |
| 三个损失函数 | 评估技能质量 | — |
| TextGrad 优化器 | 更新归纳 Prompt | — |

**为什么要冻结演绎智能体？**
> 如果演绎智能体也在变化，我们无法判断重构成功是因为技能变好了，还是因为演绎智能体本身变强了。冻结它，就能**隔离出技能质量的纯净信号**。

---

## Slide 7: 归纳智能体（Induction Agent）详解

**讲解要点（约2分钟）：**

### 输入输出
- **输入**：任务描述 t + 成功的 ReAct 轨迹 τ = {(thought_m, code_m, observation_m)}
- **输出**：结构化技能文档 s（SKILL.md 格式）

### 核心挑战：控制抽象级别

归纳智能体需要在**两个极端**之间找到平衡：

```
过于具体                              ✅ 恰到好处                        过于抽象
├── 保留实例级细节                    ├── 只保留非平凡的                  ├── 只有高层意图
├── 如：具体API路径                   │   程序性模式                      ├── 等于没说
├── 只对该任务有效                    ├── 跨任务可复用                    └── 不比任务描述多任何信息
└── 泛化能力差                        └── 包含关键策略
```

### 三类知识的分类法（Taxonomy）

归纳智能体的 Prompt P_I 编码了一个分类法，将候选知识分为三类：

| 类别 | 说明 | 示例 | 处理方式 |
|------|------|------|---------|
| **(1) 程序性惯例** | 跨任务通用但不从任务描述中显而易见的知识 | "分页请求直到返回为空" | ✅ **保留** |
| **(2) 指令可推断** | 从任务描述 t 就能推导出的知识 | "聚合任务意味着需要计数或分组" | ❌ 剔除 |
| **(3) 真值泄露** | 只有看过轨迹才知道的细节 | 具体的响应 schema、硬编码阈值 | ❌ 剔除 |

> **类比**：写一份"如何做红烧肉"的菜谱
> - (1) 应该保留："冰糖要先炒成糖色" — 不看食谱不知道的关键步骤
> - (2) 不该写："需要用刀切肉" — 看菜名就知道
> - (3) 不该写："精确需要350g五花肉" — 过于具体，照搬不了

---

## Slide 8: 演绎智能体（Deduction Agent）详解

**讲解要点（约1.5分钟）：**

### 工作方式
- **输入**：技能 s + 任务描述 t
- **过程**：在真实环境中执行多步 ReAct 循环
  ```
  每一步：思考 → 写代码 → 执行 → 观察结果 → 下一思考
  ```
- **输出**：重构轨迹 τ̂

### 关键设计

1. **Prompt 冻结**：P_D 在整个优化过程中完全不变
2. **无源轨迹访问**：演绎智能体**永远看不到**原始轨迹 τ
3. **技能作为唯一策略来源**：技能被注入为"程序性操作手册"

> **为什么这样设计？**
> 想象你是一个新员工（演绎智能体），老板给了你一份操作手册（技能），让你完成一个任务。
> - 如果你成功了 → 手册写得不错
> - 如果你失败了 → 手册有问题，需要改进
> - 关键：你不能去问原来写手册的老员工（看不到源轨迹），只能靠手册本身

---

## Slide 9: 三大文本损失函数 — 总览

**【展示对比表格】**

**讲解要点（约3分钟）：**

| 损失函数 | 评估什么 | 信号来源 | 分数范围 | 核心作用 |
|---------|---------|---------|---------|---------|
| **重构损失 L_recon** | 重构轨迹 vs 源轨迹的策略对齐度 | LLM 评判 | [0,10] | 保证技能忠实保留关键步骤 |
| **结果损失 L_outcome** | 重构轨迹的执行正确性 | 环境执行 | [0,1] | 保证技能能让任务真正成功 |
| **评分损失 L_rubric** | 技能文档的质量 & 抽象级别 | LLM 评判 | [0,10] | 保证技能可复用、可迁移 |

**优化目标 — 字典序最小化（Lexicographic Minimization）：**

$$P_I^* = \arg\min_{P_I} \left( \underbrace{L_{\text{outcome}}}_{\text{主目标}},\ \underbrace{L_{\text{recon}}}_{\text{第一打破平局}},\ \underbrace{L_{\text{rubric}}}_{\text{第二打破平局}} \right)$$

> **字典序的含义**：先比结果损失，谁的结果损失小就选谁；如果结果损失相同，再比重构损失；如果还相同，再比评分损失。
> 这确保了：**任务正确性 > 策略忠实度 > 文档质量**。

---

## Slide 10: 重构损失（Reconstruction Loss）详解

**公式：**
$$L_{\text{recon}}(\tau, \hat{\tau}, t) = (\ell_{\text{recon}}, f_{\text{recon}}) = A_J(\tau, \hat{\tau}, t; P_{\text{recon}})$$

**讲解要点：**

- **评估方式**：一个 LLM 评判器（Judge）对比原始轨迹 τ 和重构轨迹 τ̂
- **关键原则**：评估的是**策略层面的等价性**，而非步骤层面的相似性
  - ✅ 使用不同 API 端点、不同循环构造、不同中间变量 → 只要策略逻辑相同就算对齐
  - ✅ 同样的"先检索再聚合"模式、同样的分页策略 → 算对齐
- **输出**：
  - $\ell_{\text{recon}} \in [0, 10]$：轨迹差异的量化分数
  - $f_{\text{recon}}$：自然语言批评，指出具体的不匹配之处

> **类比**：两个厨师做同一道菜，一个用铁锅一个用不粘锅，一个先放盐一个后放盐——只要最终做出来的菜味道和做法的"策略"是一致的，就算重构成功。

---

## Slide 11: 结果损失（Outcome Loss）详解

**公式：**
$$L_{\text{outcome}}(\hat{\tau}, t) = (\ell_{\text{outcome}}, f_{\text{outcome}}) = \text{EnvExec}(\hat{\tau}, t)$$

**讲解要点：**

- **唯一的真实信号**：直接在真实环境中执行重构轨迹
- **不依赖 LLM 判断**：基于实际执行结果
- **输出**：
  - $\ell_{\text{outcome}} \in [0, 1]$：任务失败程度
  - $f_{\text{outcome}}$：环境反馈（错误信息、执行追踪）

**为什么需要结果损失？**
> 重构损失说"策略看起来对了"，但策略看起来对不代表执行一定成功。可能存在：
> - 隐藏的 API 失败
> - 静默错误（程序没报错但结果不对）
> - 运行时异常
>
> 结果损失是**唯一能捕获这些问题的信号**。

---

## Slide 12: 评分损失（Rubric Loss）详解

**公式：**
$$L_{\text{rubric}}(s, t) = (\ell_{\text{rubric}}, f_{\text{rubric}}) = A_J(s, t; P_{\text{rubric}})$$

**五个评分维度：**

| 维度 | 含义 | 评估什么 |
|------|------|---------|
| **GT-Independence** (真值独立性) | 技能中有多少是不看源轨迹就写不出来的？ | 防止技能记忆具体实现 |
| **Actionability** (可操作性) | 拿到技能 + API文档，能写出可工作的代码吗？ | 保证技能有足够指导性 |
| **Transferability** (可迁移性) | 技能能用到其他领域的类似任务上吗？ | 保证技能的泛化能力 |
| **Completeness** (完整性) | 完整的程序链路是否都覆盖了？ | 防止遗漏关键步骤 |
| **Conciseness** (简洁性) | 是否信息密集，没有冗余废话？ | 防止技能过于冗长 |

**评分损失的两大作用：**
1. **评估文档质量** — 确保技能是可复用、可移植的文档
2. **正则化抽象级别** — 防止归纳智能体往技能里塞实例特定的细节来"作弊"

> **为什么评分损失是正则化器？**
> 如果没有评分损失，归纳智能体可能会把具体的 API 路径、硬编码阈值写进技能，这样重构和结果损失都会很好（因为技能在"作弊"），但技能完全不能泛化到新任务。评分损失中的 GT-Independence 维度专门防止这种情况。

---

## Slide 13: 闭环优化 — TextGrad 驱动

**【展示 Algorithm 1】**

**讲解要点（约2分钟）：**

### 完整算法流程

```
初始化：最佳技能 s* = nil，各损失初始值 = ∞

for q = 0, 1, ..., Q-1:  (最多8轮迭代)
    ① 归纳：s ← A_I(t, τ; P_I^(q))          // 从轨迹提取技能
    ② 演绎：τ̂ ← A_D(t, s; P_D)              // 用技能重构轨迹
    ③ 评估：计算三个损失
       - (ℓ_recon, f_recon) ← L_recon(τ, τ̂, t)
       - (ℓ_outcome, f_outcome) ← L_outcome(τ̂, t)
       - (ℓ_rubric, f_rubric) ← L_rubric(s, t)
    ④ 追踪最优：字典序比较，保留当前最佳技能
    ⑤ 优化：
       - g ← GradientLLM(P_I^(q), t, s, τ̂, f_recon, f_outcome, f_rubric)
       - P_I^(q+1) ← OptimizerLLM(P_I^(q), g)

return s*  // 返回整个过程中找到的最佳技能
```

### 两个关键设计

**设计1：梯度 LLM 不看源轨迹 τ**
> 梯度 LLM 只看重构轨迹 τ̂，不看源轨迹 τ。关于 τ 的信息只通过重构反馈 $f_{\text{recon}}$ 间接传达。
> 这防止优化器直接把 τ 的实现细节"抄"进 Prompt。

**设计2：两步 TextGrad**
1. **梯度 LLM**：分析失败模式，输出"文字梯度" g（诊断问题 + 提出修改建议）
2. **优化器 LLM**：根据梯度 g 更新归纳 Prompt P_I

> **类比**：
> - 梯度 LLM = 质检员：看了产品缺陷报告后，指出"哪里有问题、为什么有问题"
> - 优化器 LLM = 工程师：根据质检报告，修改生产流程（Prompt）

### Prompt 演化示例（Figure 7）

**初始 Prompt P_I^(0)（~530 tokens）：**
```
Role: You are an expert at extracting reusable procedural strategies.
Rules:
1. Describe ONLY solving strategy and structural patterns
2. Do NOT include task-specific info
3. Focus on NON-OBVIOUS structural knowledge
Output: Valid SKILL.md with YAML frontmatter
```

**优化后 Prompt P_I^(5)（~2000 tokens）：**
```
[原有规则保留] +
+ Identifier & Scope Resolution: 通过父容器端点发现数据源...
+ Constraint-Based Data Filtering: 按层级遍历构建子集...
+ Output Construction & Completion Signals...
+ STRICT ABSTRACTION RULE: 永不包含具体端点、字段名或布尔值
  Bad: "调用 show_song_privates 检查 like 是否为 true"
  Good: "调用暴露用户特定状态标志的端点"
+ Common Pitfalls: 幻觉端点、假设全局列表匹配约束...
```

> TextGrad 在 4 轮迭代后将 Prompt 从 530 tokens 增长到 ~2000 tokens，新增的规则全部来自**梯度反馈中观察到的失败模式**，不是人工编写的。

---

## Slide 14: 实验设置

**讲解要点（约1分钟）：**

### 基准测试

| 基准 | 类型 | 规模 | 评估指标 |
|------|------|------|---------|
| **AppWorld** | 交互式编码 Agent 基准 | 9个日常应用，457个API；训练90任务，测试168(normal)+417(challenge) | TGC (任务目标完成率), SGC (场景目标完成率) |
| **BFCL-v3** | 多轮函数调用基准 | 200个实例，50训练+150测试 | ACC (准确率) |

### 对比方法

| 方法 | 类型 |
|------|------|
| **ReAct** | 无技能基线，只有单个示例 |
| **ICL** | 上下文学习，提供多样任务示例 |
| **Skill-extract** | 与 MIND-Skill 相同归纳智能体，但**不做迭代优化**（单次提取） |
| **ACE** | 终身演化方法，通过生成-反思-策展循环积累策略 |
| **Trace2Skill** | 轨迹蒸馏方法，并行分析+层次合并 |

### 实现细节
- 推理模型：Qwen3.5-122B-A10B
- 最大优化迭代：Q = 8
- 检索技能数：K = 3
- 跨模型实验：技能生成用 GPT-5.4，推理仍用 Qwen

---

## Slide 15: 主要结果 — Table 1

**【展示 Table 1】**

### Qwen3.5-122B-A10B 生成技能的结果

| 方法 | Normal TGC | Normal SGC | Challenge TGC | Challenge SGC | BFCL ACC | 平均 |
|------|-----------|-----------|--------------|--------------|---------|------|
| ReAct | 55.4 | 35.7 | 36.7 | 23.0 | 63.3 | 42.8 |
| ICL | 58.9 | 39.3 | 42.7 | 26.6 | 64.7 | 46.4 |
| Skill-extract | 63.7 | 46.4 | 45.1 | 30.9 | 68.7 | 51.0 |
| ACE | 65.5 | 55.4 | 51.1 | 34.5 | 74.0 | 56.1 |
| Trace2Skill | 67.3 | 55.4 | 46.8 | 33.1 | 72.7 | 55.1 |
| **MIND-Skill** | **71.4** | **55.4** | **51.8** | **39.6** | **77.3** | **59.1** |

**三大发现：**

### 发现1：MIND-Skill 全面领先
- 平均分 59.1，超过 ACE（56.1）+3.0，超过 Trace2Skill（55.1）+4.0
- 特别是在 AppWorld-Challenge SGC 上：**39.6 vs 34.5（ACE）**，领先 5.1 个百分点
- 唯一在**两个 AppWorld 分割上同时领先**的方法

### 发现2：闭环优化贡献巨大
- MIND-Skill 比 Skill-extract（相同归纳智能体但不迭代）：
  - Normal TGC：71.4 vs 63.7（**+7.7**）
  - Challenge TGC：51.8 vs 45.1（**+6.7**）
  - 平均：59.1 vs 51.0（**+8.1**）
- 证明：即使归纳智能体本身很强，单次提取也远远不够

### 发现3：弱模型 = 强模型（惊人发现！）
| 技能生成模型 | 推理模型 | 平均分 |
|------------|---------|--------|
| Qwen3.5-122B（弱） | Qwen3.5-122B | **59.1** |
| GPT-5.4（强） | Qwen3.5-122B | **58.9** |

> 弱模型生成的技能（59.1）竟然与强模型生成的技能（58.9）**几乎一样好**！
> 这意味着：**你不需要昂贵的强模型来生成高质量技能。**

---

## Slide 16: GPT-5.4 生成技能的结果

**【展示 Table 1 下半部分】**

| 方法 | Normal TGC | Normal SGC | Challenge TGC | Challenge SGC | BFCL ACC | 平均 |
|------|-----------|-----------|--------------|--------------|---------|------|
| Skill-extract (GPT) | 63.7 | 44.6 | 47.7 | 33.1 | 69.3 | 51.7 |
| ACE (GPT) | 69.0 | 58.9 | 47.2 | 33.8 | 72.7 | 56.3 |
| Trace2Skill (GPT) | 70.2 | 51.8 | 52.8 | 36.7 | 71.3 | 56.6 |
| **MIND-Skill (GPT)** | **70.8** | **57.1** | 50.6 | **37.4** | **78.7** | **58.9** |

**关键观察：**
- MIND-Skill 在 BFCL-v3 上达到 78.7，是所有方法中最高的
- 在 AppWorld-Challenge 上，Trace2Skill 的 TGC 更高（52.8 vs 50.6），但 MIND-Skill 的 SGC 更高（37.4 vs 36.7）
- 这说明 MIND-Skill 更擅长捕获**场景级别的程序性模式**，而不是任务特定的捷径

---

## Slide 17: 优化过程分析 — Figure 2 & 3

**【展示 Figure 2 前两列 + Figure 3】**

**讲解要点（约2分钟）：**

### Figure 2：性能随迭代的变化

**Normal 分割：**
- TGC 从迭代 0（相当于 Skill-extract）的 63.7 → 最终 71.4，**提升 7.7 个百分点**
- SGC 从 46.4 → 55.4，**提升 9.0 个百分点**

**Challenge 分割：**
- TGC 从 45.1 → 51.8，**提升 6.7 个百分点**
- SGC 从 30.9 → 39.6，**提升 8.7 个百分点**

**关键洞察：**
- **前3轮迭代贡献了大部分提升**（粗粒度的程序性缺陷被修复）
- **后续迭代解决更细微的边缘情况**（难任务的收益集中在后期）
- 简单任务在早期就饱和了，困难任务持续受益

### Figure 3：损失值随迭代的变化

```
迭代 0-2：结果损失急剧下降 → 接近 0（任务基本都能执行成功了）
迭代 3-8：重构损失和评分损失持续缓慢下降（策略对齐和文档质量持续改善）
```

> 这符合直觉：先让任务能跑通（结果损失），再追求策略忠实度和文档质量。

---

## Slide 18: 技能检索数量 K 的影响 — Figure 2 第三列

**【展示 Figure 2 第三列】**

**讲解要点（约1.5分钟）：**

| K值 | Normal TGC | Normal SGC | Challenge TGC | Challenge SGC |
|-----|-----------|-----------|--------------|--------------|
| 1 | 较低 | 较低 | 较低 | 较低 |
| 2 | 大幅提升 | 大幅提升 | 大幅提升 | 大幅提升 |
| **3** | **最优平衡** | 良好 | **最优平衡** | **最优平衡** |
| 5 | 略降 | **62.5 (最高)** | 略降 | 较好 |

**分析：**
- **K=1 表现差**：单个技能无法覆盖测试任务的完整程序范围
- **K=1→3 大幅提升**：多个互补技能拓宽了程序性覆盖范围
- **K=5 继续提升 Normal SGC**：更多技能帮助场景级一致性
- **最终选择 K=3**：在整体性能上取得最佳平衡

**按难度分析：**
- 简单任务：K≥2 就接近天花板
- 中等/困难任务：从 K=3 中受益最大（额外覆盖度很重要）

---

## Slide 19: 消融实验 — Table 2

**【展示 Table 2】**

| 方法 | Normal TGC | Normal SGC | Challenge TGC | Challenge SGC |
|------|-----------|-----------|--------------|--------------|
| Skill-extract（无优化） | 63.7 | 46.4 | 45.1 | 30.9 |
| w/o outcome（去掉结果损失） | 68.5 | 55.4 | 48.0 | 34.5 |
| w/o reconstruction（去掉重构损失） | 66.1 | 46.4 | 45.8 | 31.6 |
| w/o rubric（去掉评分损失） | 64.3 | 46.4 | 46.5 | 31.6 |
| **MIND-Skill（完整）** | **71.4** | **55.4** | **51.8** | **39.6** |

**三大结论：**

### 去掉重构损失：影响最大（Challenge TGC: 51.8 → 45.8）
- 没有轨迹对比，优化器**缺乏细粒度的程序性反馈**
- 无法识别缺失的关键步骤和有缺陷的工作流
- Challenge TGC 几乎退化到 Skill-extract 水平（45.1）

### 去掉评分损失：Normal TGC 影响最大（71.4 → 64.3）
- 没有抽象级别正则化，优化器倾向于往技能里塞实例特定细节
- 可能在某些 Challenge 任务上碰巧有帮助，但**损害整体泛化能力**

### 去掉结果损失：影响相对温和
- **但即使没有真实执行反馈**，仅靠重构损失+评分损失（68.5 Normal, 48.0 Challenge）
- 已经**超过 Trace2Skill**（55.4 Normal TGC, 46.8 Challenge TGC 不可比但参考）
- 结果损失捕获的是运行时错误和静默 API 失败，文字判断无法发现的问题

---

## Slide 20: Token 效率分析 — Figure 4

**【展示 Figure 4】**

**讲解要点（约1分钟）：**

| 方法 | AppWorld 注入 Tokens | BFCL 注入 Tokens |
|------|---------------------|-----------------|
| ACE | **~11.6K** | ~6.32K |
| Trace2Skill | ~6.43K | ~5.54K |
| **MIND-Skill** | **~1.89K** | **~1.72K** |

**MIND-Skill 生成的技能比 ACE 紧凑 3-6 倍！**

**为什么这么紧凑？**
- 评分损失中的 **Conciseness 维度** 显式惩罚冗余
- 遵循**渐进披露原则**：每个技能只覆盖与其匹配任务相关的程序性知识
- ACE 和 Trace2Skill 把所有训练知识塞进一个巨大的单一体中，不管是否与当前任务相关

> 这在实际应用中意味着：**更低的 API 调用成本、更快的推理速度、更少的上下文污染。**

---

## Slide 21: 案例研究 — 技能质量对比（Figure 5）

**【展示 Figure 5】**

**讲解要点（约2分钟）：**

### ACE 的问题：任务特定记忆化（Task-specific Memorization）
```python
# ACE 把训练任务 60d0b5b 的解法直接记住了：
"Venmo Refund Workflow: When a task requires refunding
 an accidental 'payment request' that was approved:
 (1) use show_sent_payment_requests with status='approved'
 (2) filter by receiver's email
 (3) sort by created_at descending
 (4) take the most recent request
 (5) refund via create_transaction"
```
> 问题：触发条件几乎是原任务描述的改写，5步流程是具体解法的记忆化。任何偏离此场景的测试任务都会误触发。

### Trace2Skill 的问题：结构畸形（Malformed Structure）
```
## When to Apply
- Filesystem-style tasks: navigating ...
- Social / messaging tasks: posting ...
  3a. Trust tool responses: ...       ← 程序步骤出现在"适用条件"下！
  3b. Calculate derived values ...     ← 位置完全错误
  (3c-3m: 还有10个程序步骤)
## Procedure                            ← Procedure 反而是空的
1. Per-turn intent extraction: ...
```
> 问题：13个程序步骤被错误地放在了"When to Apply"标题下，结构混乱导致下游 Agent 解析困难。

### MIND-Skill 的产出：结构清晰、概念化、可迁移
```
## When to Apply
- Target API requires auth token via a separate login step.
- List endpoint returns data in pages, requiring a loop.
- Task involves identifying one item by a label or attribute...

## Procedure
1. Authenticate: Retrieve token.
2. Paginate and Collect: Loop until empty.
3. Identify Target: Find item by specific label.
4. Apply Specific Update: Modify target.
5. Apply Bulk Update: Disable the rest.
6. Verify: Re-fetch and confirm.

## Key Patterns
- Pagination Loop: while-loop, break on empty page.
- Selective Mutation: unique update for target, uniform for rest.

## Common Pitfalls
- Failing to loop through all pages.
- Updating target incorrectly or failing to exclude it.
```
> MIND-Skill 使用**概念化占位符**（"label or attribute"、"time shift"）而非具体的值，结构严格遵循 SKILL.md 规范。

---

## Slide 22: 案例研究 — 弱模型为何匹敌强模型（Figure 6）

**【展示 Figure 6】**

**讲解要点（约2分钟）：**

### 对比：同一个训练任务（302c169_1），两个模型生成的技能

| | Qwen 自生成技能 | GPT 生成的技能 |
|---|---|---|
| **净贡献** | **+12** (翻转12个测试任务从失败到成功) | +1 (仅翻转1个) |
| **词汇风格** | 朴素标签："Pagination Loop"、"Selective Mutation" | 教科书风格："Doc-first execution"、"Credential bootstrap" |
| **详细程度** | 6步精简流程 | 10步详细流程，包含防御性检查 |

**核心发现：自我训练效应（Self-training Effect）**

> Qwen 生成的技能用的是 Qwen 自己"习惯的语言"，GPT 生成的技能用的是 GPT 的"方言"。
> 当推理时把技能注入 Qwen 的 Prompt：
> - Qwen 自写的技能 → 自然解码，直接理解
> - GPT 写的技能 → 需要"隐式方言翻译"，削弱程序性信号
>
> 这不是正确性问题——GPT 的标签可以说更精确——但**精确的外语方言不如朴素但熟悉的母语好使**。

**另一个因素：能力校准**
- GPT 的技能推荐细粒度检查（如检查 partial-update 语义），GPT 自己能做但 Qwen 做不到
- Qwen 不会去利用这些超出它能力范围的建议
- 自我训练避免了这种"能力不匹配"

**启示：**
> 经过质量保证的优化后，**风格对齐（style alignment）比技能生成器的原始推理能力更重要**。
> 这对实践意义重大：不需要购买昂贵的强模型 API，用自己部署的模型就能生成同样好的技能。

---

## Slide 23: 实践落地 — 在 DataCrawlerAgent 上构建技能系统

**讲解要点（约3-4分钟）：**

### 我的项目：DataCrawlerAgent 简介

一个多智能体学术数据采集系统，包含5个专业 Agent：

```
ArxivBatchDownloader → PaperCollector → DataCleaner → ReferenceGraphBuilder → DatasetFormatter
```

| Agent | 功能 |
|-------|------|
| ArxivBatchDownloader | 从arXiv元数据筛选论文 |
| PaperCollector | 下载PDF + 多模态提取（文本/图像/公式/表格） |
| DataCleaner | 规则清洗Markdown |
| ReferenceGraphBuilder | 引用关系抽取与匹配 |
| DatasetFormatter | 组装多模态属性图数据集 |

### 用 MIND-Skill 方法构建技能系统

**步骤1：收集成功轨迹**
- 运行完整 pipeline，记录每个 Agent 的执行轨迹
  ```
  τ = {
    (thought_1, tool_call_1, observation_1),   # 思考 + 调用 + 观察结果
    (thought_2, tool_call_2, observation_2),
    ...
  }
  ```
- 对于 DataCrawlerAgent，收集以下场景的成功轨迹：
  - 单篇论文处理（从PDF到清洗后的Markdown）
  - 批量论文处理（多GPU并行提取）
  - 引用图谱构建（跨论文引用匹配）

**步骤2：定义技能层级**

```
Skill: arxiv_multimodal_dataset_construction
│
├── Primitive Skills（原子技能）:
│   ├── download_pdf:         下载单篇arXiv论文PDF
│   ├── extract_multimodal:   从PDF提取文本/图像/公式/表格
│   ├── clean_markdown:       清洗提取的Markdown文本
│   └── parse_references:     从PDF中解析引用关系
│
├── Procedure Skills（流程技能）:
│   ├── paper_collect_and_extract:  下载+多模态提取的完整流程
│   ├── data_clean_pipeline:        清洗流水线（去重+修公式+修引用）
│   └── ref_graph_build:            引用图谱构建（抽取+匹配+验证）
│
├── Strategy Skills（策略技能）:
│   ├── batch_extraction_strategy:  多GPU并行分配策略
│   ├── quality_assurance:          数据质量检查策略
│   └── error_recovery:             错误恢复与重试策略
│
└── Policy Skill（顶层策略）:
    └── academic_dataset_construction:  完整学术数据集构建的端到端策略
```

**步骤3：用 MIND-Skill 归纳智能体提取技能**

以 `paper_collect_and_extract` 为例：

```markdown
# SKILL.md - paper_collect_and_extract

## Overview
从arXiv下载论文PDF并提取多模态数据（文本、图像、公式、表格）。

## When to Apply
- 需要从arXiv获取论文并提取结构化内容
- PDF需要经过深度学习模型处理（非简单文本提取）
- 支持单篇或批量处理

## Procedure
1. 验证输入：检查arXiv ID格式，确认输出目录可写
2. 下载PDF：使用arXiv标准URL，支持重试和缓存
3. GPU分配：根据可用GPU数量，将论文分配到不同GPU批次
4. 多模态提取：调用pdf-extract-kit，执行布局检测+OCR+公式识别
5. 结果验证：检查输出文件完整性（.md + .npy + images/）
6. 错误处理：对失败论文记录日志，尝试降级提取

## Key Patterns
- Batch Loading: 每个GPU批次加载一次模型（~30s），而非每篇论文加载
- Skip-Already-Extracted: 检测已有输出文件，跳过已完成的论文
- Fallback Extraction: 批量失败时降级为逐篇提取

## Common Pitfalls
- PaddleOCR不遵守CUDA_VISIBLE_DEVICES，需要额外设备设置
- 论文名含特殊字符导致文件路径问题
- 大批量处理时内存不足，需要控制batch_size
```

**步骤4：用演绎智能体验证技能**
- 冻结演绎智能体的 Prompt
- 只给技能文档 + 任务描述，让演绎智能体重新执行
- 对比重构轨迹与原始轨迹，计算三大损失

**步骤5：TextGrad 迭代优化**
- 根据三大损失的文本反馈，更新归纳智能体的 Prompt
- 重复最多 8 轮，选择字典序最优的技能

### 预期收益

| 维度 | 当前状态 | 引入技能后 |
|------|---------|----------|
| **可复用性** | 每次需要手动调用5个Agent | 一条指令自动执行完整pipeline |
| **可迁移性** | 绑定arXiv特定逻辑 | 技能可适配其他学术数据库（Semantic Scholar, DBLP） |
| **可维护性** | 策略分散在代码中 | 集中在SKILL.md中，人类可读可审 |
| **鲁棒性** | 缺乏质量保证 | 闭环验证确保技能有效性 |
| **Token效率** | 不适用 | 紧凑技能，最小化注入开销 |

---

## Slide 24: 技能检索与执行 — 端到端演示

**讲解要点（约1.5分钟）：**

### 用户视角的技能使用流程

```
用户输入：
"帮我从arXiv上爬取cs.AI领域最新的50篇论文，提取多模态数据并构建引用图谱"

         ↓

[技能检索] → 从技能库中检索 K=3 个最相关的技能：
  1. arxiv_batch_download (匹配: arXiv + 批量 + 爬取)
  2. paper_collect_and_extract (匹配: 论文 + 提取)
  3. ref_graph_build (匹配: 引用 + 图谱)

         ↓

[技能注入] → 将3个技能注入 Agent 的 Prompt

         ↓

[Agent执行] → 按照技能中的 Procedure 步骤：
  Step 1: 筛选 cs.AI 论文（ArxivBatchDownloaderAgent）
  Step 2: 下载PDF + 多GPU提取（PaperCollectorAgent）
  Step 3: 清洗Markdown（DataCleanerAgent）
  Step 4: 构建引用图（ReferenceGraphBuilderAgent）
  Step 5: 组装数据集（DatasetFormatterAgent）

         ↓

输出：multimodal_attribute_graph.json
```

### 与 MIND-Skill 的对应关系

```
MIND-Skill 框架          →  DataCrawlerAgent 落地
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
成功轨迹 τ               →  完整pipeline的成功运行日志
任务描述 t               →  "爬取cs.AI论文并构建数据集"
归纳智能体 A_I           →  从运行日志提取SKILL.md
演绎智能体 A_D           →  用新论文测试技能是否可复用
重构损失 L_recon         →  对比技能执行 vs 原始执行的步骤对齐
结果损失 L_outcome       →  检查最终数据集是否正确生成
评分损失 L_rubric        →  评估SKILL.md的文档质量
```

---

## Slide 25: 总结与启示

**论文贡献总结：**

1. **MIND-Skill 框架**：首个通过多智能体归纳-演绎实现技能质量保证的方法
2. **三大文本损失**：重构损失（策略忠实度）+ 结果损失（执行正确性）+ 评分损失（文档质量 & 抽象级别正则化）
3. **闭环优化**：TextGrad 驱动的迭代优化，冻结演绎智能体隔离技能质量信号
4. **SOTA 性能**：AppWorld 和 BFCL-v3 上全面领先
5. **弱模型友好**：经过质量保证后，风格对齐比模型能力更重要

**对我项目的启示：**

1. **技能是可以自动生成的** — 不需要手动编写每个SKILL.md
2. **质量保证是关键** — 闭体验证确保生成的技能真的有用
3. **三层损失缺一不可** — 尤其是评分损失防止技能"作弊"
4. **用自己项目的模型就行** — 不需要强模型API，自我训练效应保证效果
5. **技能库规模可控** — 我的5个Agent约产生10-15个技能，远低于相变阈值（κ≈90）

**未来方向：**
- 在 DataCrawlerAgent 上实现完整的 MIND-Skill pipeline
- 将技能系统扩展到其他学术数据源（Semantic Scholar, PubMed）
- 探索技能的终身演化（随着新论文格式、新提取工具出现不断更新技能）

---

## Slide 26: Q&A

**谢谢！欢迎提问。**

---

## 附录：演讲时间分配建议

| 部分 | Slides | 建议时间 |
|------|--------|---------|
| 背景 & 动机 | 3-4 | 3分钟 |
| 核心问题 | 4 | 1.5分钟 |
| 框架总览 | 5-6 | 3.5分钟 |
| 归纳智能体 | 7 | 2分钟 |
| 演绎智能体 | 8 | 1.5分钟 |
| 损失函数 | 9-12 | 3分钟 |
| 闭环优化 | 13 | 2分钟 |
| 实验结果 | 14-18 | 4分钟 |
| 消融实验 | 19-20 | 2分钟 |
| 案例研究 | 21-22 | 3分钟 |
| **项目落地** | **23-24** | **4.5分钟** |
| 总结 | 25 | 1.5分钟 |
| **总计** | — | **~28分钟** |

> 建议根据实际时间灵活调整，如果时间紧可以快速过消融实验部分，重点放在框架设计和项目落地上。
