# G2Text / G2Image 决策（T3⑭）—— 从贡献剔除生成，改写为诚实局限

> 2026-06-23

## 现状（实测）
- **G2Text**：arxiv 单域、gcn 单模型、text 单模态、2 个 PEFT 变体有真实结果：
  - peft-none:  BLEU-4=0.005, ROUGE-L=0.072, CIDEr=0.007
  - peft-lora:  BLEU-4=0.006, ROUGE-L=0.077, CIDEr=0.012
- **G2Image**：结果目录为空（代码 `benchmark/tasks/g2image_generation.py` 就绪，未训）。

## 判断
数值近乎为零（BLEU-4 0.005 ≈ 随机生成）。**生成不是本文强项**，且 OpenMAG 在生成任务（G2T/G2I）上有完整结果与专用指标，正面竞争必输。

## 决策：从贡献中剔除生成，改写为「诚实局限」一条
- **摘要 / 贡献 / 引言**：不出现任何 "graph-to-text / graph-to-image generation" 字样。
- **Limitations 章节**写一句：
  > "We explored graph-conditioned text generation on arxiv (graph-to-text via GNN-encoded node embeddings fed to a decoder), but obtained near-zero BLEU-4 (0.005–0.006) and CIDEr (0.007–0.012), indicating that frozen-CLIP node embeddings + a light decoder are insufficient for generation; dedicated graph-to-sequence architectures (as in OpenMAG's G2T/G2I suite) are out of scope. Generation is therefore not a task this benchmark claims."
- **附录**（可选）：附上这两个数，证明「试过、量化过、主动剔除」，而非藏起来。

## 为什么这样最稳
审稿人对「声称生成却无结果/弱结果」零容忍（常见拒因）。主动把弱结果定位成「我们试了、发现不够、故不声称」，把范围严格收敛到证据（NC/LP/边关系/跨域/可控性），是 D&B 最安全的姿态。这与论文整体「诚实护城河」策略一致。

## 如需加强（camera-ready，非投稿门槛）
要把生成做成卖点需：① 多域（不止 arxiv）；② 专用 graph-to-sequence 解码器（非 frozen CLIP + 浅解码头）；③ 与 OpenMAG G2T/G2I 同口径指标。工程量大、收益不确定，不建议作为主线。
