# OpenMAG 完整实验运行指南

## 项目路径
```
/home/dev_qw/DataCrawlerAgent
```

## 环境准备（已完成）
- Python 3.9 + PyTorch 2.8 + PyG
- CLIP-ViT-Large-Patch14 (768d) 已下载到本地缓存
- 768d 编码已完成: text/image/formula 各 45677/45672/45677 节点, dim=768
- 多模态组合已生成: text+image, text+formula, image+formula, text+image+formula
- 旧编码已备份: text_embeddings_384d.pt, image_embeddings_512d.pt, formula_embeddings_384d.pt

## 已实现的全部模型 (14个)
| 类别 | 模型 | 说明 |
|------|------|------|
| Baseline | mlp | 无图结构基线 |
| 传统GNN | gcn, gat, graphsage, gin, gt | 标准图神经网络 |
| 改进GNN | revgat | 可逆GAT |
| Graph-enhanced | dgf | 双图滤波(分离同配/异配信号) |
| Graph-enhanced | dmgc | 分离多关系图聚类 |
| Multimodal-enhanced | mmgcn | 多模态GCN(独立视图传播) |
| Multimodal-enhanced | mgat | 多模态门控注意力GAT |
| Multimodal-enhanced | lgmrec | 局部+全局图学习(超边) |
| Multimodal-enhanced | mhgat | 多关系异构GAT |

## 实验运行命令

### 单GPU顺序运行所有实验
```bash
cd /home/dev_qw/DataCrawlerAgent
HF_ENDPOINT=https://hf-mirror.com nohup python3 benchmark/run_openmag_experiments.py --phase all --gpu 0 > benchmark/results/openmag/all_experiments.log 2>&1 &
```

### 多GPU并行运行（推荐，8卡可用）
```bash
cd /home/dev_qw/DataCrawlerAgent
export HF_ENDPOINT=https://hf-mirror.com

# GPU 0: Q1+Q2 (必要性)
nohup python3 benchmark/run_openmag_experiments.py --phase q1_necessity --gpu 0 > benchmark/results/openmag/q1.log 2>&1 &

# GPU 1: Q3+Q4 (数据质量)
nohup python3 benchmark/run_openmag_experiments.py --phase q3_finetune --gpu 1 > benchmark/results/openmag/q3.log 2>&1 &
nohup python3 benchmark/run_openmag_experiments.py --phase q4_encoding --gpu 1 > benchmark/results/openmag/q4.log 2>&1 &

# GPU 2: Q5 (有效性 - 最大实验)
nohup python3 benchmark/run_openmag_experiments.py --phase q5_effectiveness --gpu 2 > benchmark/results/openmag/q5.log 2>&1 &

# GPU 3: Q6+Q2
nohup python3 benchmark/run_openmag_experiments.py --phase q6_peft --gpu 3 > benchmark/results/openmag/q6.log 2>&1 &
nohup python3 benchmark/run_openmag_experiments.py --phase q2_neighbors --gpu 3 > benchmark/results/openmag/q2.log 2>&1 &

# GPU 4: Q7 (鲁棒性 - 6模型×3噪声)
nohup python3 benchmark/run_openmag_experiments.py --phase q7_robustness --gpu 4 > benchmark/results/openmag/q7.log 2>&1 &

# GPU 5: Q8 (边噪声 - 6模型)
nohup python3 benchmark/run_openmag_experiments.py --phase q8_edge_noise --gpu 5 > benchmark/results/openmag/q8.log 2>&1 &

# GPU 6: Q9+Q10 (效率)
nohup python3 benchmark/run_openmag_experiments.py --phase q9_q10_efficiency --gpu 6 > benchmark/results/openmag/q9_q10.log 2>&1 &
```

### 逐个运行（最稳定）
```bash
cd /home/dev_qw/DataCrawlerAgent
export HF_ENDPOINT=https://hf-mirror.com

python3 benchmark/run_openmag_experiments.py --phase q1_necessity --gpu 0
python3 benchmark/run_openmag_experiments.py --phase q2_neighbors --gpu 0
python3 benchmark/run_openmag_experiments.py --phase q3_finetune --gpu 0
python3 benchmark/run_openmag_experiments.py --phase q4_encoding --gpu 0
python3 benchmark/run_openmag_experiments.py --phase q5_effectiveness --gpu 0
python3 benchmark/run_openmag_experiments.py --phase q6_peft --gpu 0
python3 benchmark/run_openmag_experiments.py --phase q7_robustness --gpu 0
python3 benchmark/run_openmag_experiments.py --phase q8_edge_noise --gpu 0
python3 benchmark/run_openmag_experiments.py --phase q9_q10_efficiency --gpu 0
```

## 各实验详情（完全对齐论文第四章）

### Q1: Necessity - 单模态 vs 多模态 (对应论文 Figure 2)
- **内容**: NC + LP 在 text-only / image-only / text+image+formula 上的表现
- **模型(NC)**: mlp, gcn, gat, graphsage × 3种模态 × 3次运行 = 36组实验
- **模型(LP)**: mlp, gcn, gat × 3种模态 × 3次运行 = 27组实验
- **指标**: NC用 Acc±F1, LP用 MRR±Hits@3
- **结果文件**: `results/openmag/q1_necessity.json`

### Q2: Necessity - 邻居数消融 (对应论文 Table 2)
- **内容**: GCN在邻居数K=0,2,4,6,8时的NC准确率
- **模型**: GCN × 5种邻居数 × 3次运行
- **模态**: text+image+formula (768d×3=2304d)
- **结果文件**: `results/openmag/q2_neighbor_ablation.json`

### Q3: Data Quality - 微调对比 (对应论文 Figure 3)
- **内容**: MLP + 4种编码器冻结/训练配置
- **配置**: T-Frozen+V-Frozen, T-Train+V-Frozen, T-Frozen+V-Train, T-Train+V-Train
- **模态**: text+image (768d×2=1536d)
- **结果文件**: `results/openmag/q3_finetune.json`

### Q4: Data Quality - 编码策略 (对应论文 Figure 4)
- **内容**: 独立编码器 vs 统一编码器的跨模态检索质量
- **策略**: CLIP-Large(unified), SBERT+CLIP-Base(independent), SPECTER+CLIP-Large
- **任务**: text-to-image / image-to-text 检索
- **指标**: MRR, Hits@1/5/10/20
- **结果文件**: `results/openmag/q4_encoding_comparison.json`

### Q5: Effectiveness - 全模型对比 (对应论文 Table 3)
- **内容**: 9个模型在 NC / LP / Clustering 三个任务上的全面对比
- **模型**: mlp, gcn, gat, dgf, mmgcn, mgat, lgmrec, dmgc, mhgat
- **NC**: 9模型 × 3次 = 27组, 指标 Acc±F1
- **LP**: mlp, gcn, gat, dgf, lgmrec × 3次 = 15组, 指标 MRR±Hits@3
- **Clustering**: 9模型 × 3次 = 27组, 指标 NMI±ARI
- **模态**: text+image+formula
- **结果文件**: `results/openmag/q5_effectiveness.json`
- **注意**: 这是最耗时的实验（约70组训练），预计2-4小时

### Q6: Effectiveness - PEFT策略 (对应论文 Table 5)
- **内容**: GCN上 None / Adapter / LoRA(r=8) / LoRA(r=16)
- **模态**: text+image+formula
- **结果文件**: `results/openmag/q6_peft.json`

### Q7: Robustness - 特征/标签噪声 (对应论文 Figure 5a-c)
- **内容**: 6个模型在 text/image/label 噪声下的性能退化
- **模型**: gat, graphsage, dgf, mmgcn, mgat, lgmrec (论文Figure 5)
- **噪声比**: 0.0~0.8 (9级) × 3噪声类型 × 6模型 × 3次运行 = 486组
- **结果文件**: `results/openmag/q7_robustness.json`
- **注意**: 第二耗时的实验，预计3-6小时

### Q8: Robustness - 边噪声 (对应论文 Figure 5d)
- **内容**: 6个模型在边噪声(随机添加虚假边)下的性能
- **噪声比**: 0.0~0.8 × 6模型 × 3次 = 162组
- **结果文件**: `results/openmag/q8_edge_noise.json`

### Q9-Q10: Efficiency - 效率分析 (对应论文 Table 6 + Figure 6/7)
- **Q9**: 9个模型的理论复杂度表 (O符号)
- **Q10**: 每个模型实测: 训练时间/epoch, 峰值GPU显存, 参数量
- **结果文件**: `results/openmag/q9_q10_efficiency.json`

## 结果汇总与可视化

实验完成后运行：
```bash
cd /home/dev_qw/DataCrawlerAgent
python3 benchmark/aggregate_results.py benchmark/results/openmag benchmark/results/openmag/tables
```

### 结果文件说明
所有JSON结果在 `benchmark/results/openmag/` 下：
- `q1_necessity.json` - 单模态vs多模态
- `q2_neighbor_ablation.json` - 邻居消融
- `q3_finetune.json` - 微调对比
- `q4_encoding_comparison.json` - 编码策略
- `q5_effectiveness.json` - 模型有效性(NC+LP+Clustering)
- `q6_peft.json` - PEFT策略
- `q7_robustness.json` - 噪声鲁棒性(6模型)
- `q8_edge_noise.json` - 边噪声(6模型)
- `q9_q10_efficiency.json` - 效率分析

### 可视化脚本
```python
# 在Python中生成所有图表
import sys; sys.path.insert(0, '.')
import json
from benchmark.analysis.visualization import *

results_dir = "benchmark/results/openmag"
out_dir = "benchmark/results/openmag/figures"

# Q7-Q8: 噪声鲁棒性曲线
q7 = json.load(open(f"{results_dir}/q7_robustness.json"))
plot_noise_robustness(q7, f"{out_dir}/noise_robustness.pdf")

# Q2: 邻居消融
q2 = json.load(open(f"{results_dir}/q2_neighbor_ablation.json"))
plot_neighbor_ablation(q2, f"{out_dir}/neighbor_ablation.pdf")

# Q3: 微调对比
q3 = json.load(open(f"{results_dir}/q3_finetune.json"))
plot_finetune_comparison(q3["results"], f"{out_dir}/finetune_comparison.pdf")

# Q6: PEFT对比
q6 = json.load(open(f"{results_dir}/q6_peft.json"))
plot_peft_comparison(q6["results"], f"{out_dir}/peft_comparison.pdf")

# Q10: 效率散点图
q910 = json.load(open(f"{results_dir}/q9_q10_efficiency.json"))
plot_efficiency(q910["empirical_efficiency"], f"{out_dir}/efficiency.pdf")

# Q5: 有效性柱状图
q5 = json.load(open(f"{results_dir}/q5_effectiveness.json"))
plot_effectiveness_table(q5["node_classification"], q5["link_prediction"],
                          q5["node_clustering"], f"{out_dir}/effectiveness.pdf")
```

## 预计运行时间
| 实验 | GPU时间估计 | 组数 |
|------|-------------|------|
| Q1 | 30-60min | 63 |
| Q2 | 10-15min | 15 |
| Q3 | 10-15min | 12 |
| Q4 | 5min | 3 |
| Q5 | 2-4h | 69 |
| Q6 | 15-20min | 12 |
| Q7 | 3-6h | 486 |
| Q8 | 1-2h | 162 |
| Q9-Q10 | 15min | 9 |
| **总计** | **8-14h单GPU** | |

多GPU并行可压缩到3-5小时。
