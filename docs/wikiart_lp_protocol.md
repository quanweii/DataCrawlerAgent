# WikiArt 链路预测协议规范

> 本文档规定 WikiArt 多模态属性图上链路预测 (Link Prediction, LP) 实验的**标准评估协议**。
> 使用本数据集做 LP 实验前**必须**阅读并遵循本文档,否则将得到虚高、不可发表的 AUC。
>
> 适用工具:[`scripts/lp_hard_negative_sampler.py`](../scripts/lp_hard_negative_sampler.py)

---

## 1. 问题:定义性标签泄漏

WikiArt 图的标签 (label) 是**艺术风格 (style)**,边由两类关系构成:

| 边关系 | 边数 | 同质性 (homophily) | 性质 |
|--------|------|------|------|
| `same_artist` | 443,554 | 0.876 | 学到的关联(艺术家多在同一风格内创作) |
| `shared_style` | 77,016 | **1.000** | **定义性同质** |

**`shared_style` 边按定义连接同一风格的画作,而风格就是节点标签** → 这类边 100% 连接同类节点,这是图构建的产物,不是模型学到的信号。

整体同质性 0.894,意味着绝大多数正样本边连接同类节点。

## 2. 随机负采样的失败

传统 LP 评估用**随机负采样**:从全体节点对中均匀抽不存在的边作为负样本。在同质性高的图上,这会让任务退化——

随机抽出的负对几乎总是**跨标签**(因为正样本几乎都是同标签)。于是模型只需学会「同标签 → 预测有边」这一平凡规则即可获胜,不需要任何图表征学习。

实测诊断(见下表,`label_agreement` = 负样本中恰好同标签的比例):

| 数据集 | 正样本同标签率 | 随机负样本同标签率 | 随机负采样下的 baseline AUC 估计 |
|--------|------|------|------|
| Steam | 0.526 | 0.313 | 0.765 |
| TMDb | 0.230 | 0.124 | 0.881 |
| **WikiArt** | **0.894** | **0.020** | **0.947** |

WikiArt 最严重:正样本 89% 同标签,随机负样本仅 2% 同标签。两者几乎完全可分,**一个不看图、只看标签是否相同的平凡分类器就能拿 0.947 AUC**。在随机负采样下,WikiArt 的 LP 是退化的、无意义的。

> ⚠️ **禁止**:对 WikiArt 只报告随机负采样的 LP 结果。该结果反映的是标签泄漏,不是模型能力。

## 3. 标准协议:硬负采样 (Hard Negative Sampling)

**主协议**:用**同标签约束**的负样本(hard negatives)。即负样本对的两端必须**属于同一风格**,且不是图中的真实边。

这样正负样本在「标签是否一致」这一捷径特征上**完全一致**(均为 1.0),模型无法靠它取胜,被迫学习真正的图表征——哪些同风格的画作对之间存在连接(本质是 `same_artist` 等潜因子 + 图像/文本相似性)。

本工具还提供更严格的 `hard_degree`(同标签 + 同度数桶),同时消除「度数」捷径,供进一步消融。

### 三种负采样策略(本工具实现)

| 策略 | 定义 | 用途 |
|------|------|------|
| `random` | 均匀随机非边 | baseline 对照,**必须同时报告**,以暴露泄漏 |
| `hard` | 同标签非边 | **主协议**,论文主结果 |
| `hard_degree` | 同标签 + 同度数桶非边 | 最严格,消融/稳健性 |

## 4. 工具使用

```bash
cd /home/dev_qw/DataCrawlerAgent

# 三个数据集全部生成 (默认 seed=42, 负:正=1:1, 80/10/10 切分)
python scripts/lp_hard_negative_sampler.py --domain all

# 单独跑 WikiArt
python scripts/lp_hard_negative_sampler.py --domain wikiart --seed 42

# 自定义负正比 (如 1:1 是 AUC 标准, 可改为 100 用于排序指标)
python scripts/lp_hard_negative_sampler.py --domain wikiart --ratio 100
```

### 输出目录 `data/{domain}_lp/`

```
train_pos.csv              训练正边
train_neg.csv              训练负边 (random)
val_pos.csv   val_neg.csv  验证 (random)
val_neg_hard.csv           验证硬负 (同标签)
val_neg_hard_degree.csv    验证硬负 (同标签+同度数桶)
test_pos.csv  test_neg.csv 测试 (random)
test_neg_hard.csv          测试硬负 (同标签)        ← 主结果用这个
test_neg_hard_degree.csv   测试硬负 (同标签+同度数桶)
protocol.json              策略/seed/统计, 供实验复现
```

每个 CSV 两列 `source,target`,直接可作为 PyG `LinkNeighborLoader` / DGL 的边标签。

## 5. 论文中如何报告

对每个数据集的 LP 实验,建议同时报告两列结果以体现严谨性:

| 负采样 | 含义 | 预期 |
|--------|------|------|
| Random | 暴露泄漏的 baseline | WikiArt 会很高(虚高),TMDb 较低 |
| **Hard (同标签)** | **主结果** | 真实反映模型能力,各模型间有区分度 |

- **主结果表**用 `test_neg_hard.csv` 评估(报告 Hits@K、MRR 或 AUC)。
- **补充材料/消融**额外给 random 对照,并注明差距来源是标签泄漏(引用本文档的诊断数据)。
- 复现说明里写明 seed、负正比、切分比例(均可从 `protocol.json` 读取)。

## 6. 为什么这不削弱数据集的价值

- 边本身合法:`shared_style` 边是真实的图结构(同风格画作确实共现),保留在图里用于消息传递完全正确。
- 问题只在**评估协议**:如何采样负样本。用硬负采样后,任务变为「在同风格画作中预测哪些对真实相连」,这是一个需要多模态 + 结构信号的有意义的任务。
- **同质性谱是 benchmark 的优势**:TMDb(0.24,异质,hard)→ Steam(0.53,平衡)→ WikiArt(0.89,同质,需 hard 负采样),覆盖不同难度,正好展示方法在三种 regime 下的表现。

## 7. 复现检查清单

- [ ] 用 `lp_hard_negative_sampler.py` 生成 `data/wikiart_lp/`(及其他两域)
- [ ] LP 主结果基于 `test_neg_hard.csv`
- [ ] 补充报告 `test_neg.csv`(random)对照,并说明差距
- [ ] `protocol.json` 中的 `label_agreement` 数据写入论文(证明 random 泄漏、hard 消除捷径)
- [ ] 注明 seed=42, 负正比 1:1, 切分 80/10/10
