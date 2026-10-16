# Dataset Licensing & Reproducibility Strategy

> 各域数据发布前必须给出可操作的获取路径；本文档与英文版
> `DATASET_LICENSING_en.md` 保持一致。

## 1. 数据来源与版权分类

| 域 | 原始数据 | 版权状态 | 可再发布内容 |
|---|---|---|---|
| **arxiv** | Open-access 论文 (arXiv.org) | CC/BY 许可, 无版权障碍 | ✅ 全集: 节点/边/文本/图像/公式/嵌入 |
| **steam** | Steam 游戏元数据 (Steamworks API + 公开页面) | Valve 版权, 不可再发布原始文本 | ⚠️ 图结构 + 嵌入 + 标签 + 构建脚本 |
| **tmdb** | TMDb 电影元数据 (TMDb API v3) | TMDb 版权, 不可再发布原始文本/图像 | ⚠️ 图结构 + 嵌入 + 标签 + 构建脚本 |
| **wikiart** | WikiArt 画作元数据 (公开爬取) | 画作图像各有版权, 元数据可争议 | ⚠️ 图结构 + 嵌入 + 标签 + 构建脚本 |
| **spotify** | Tidy Tuesday "Spotify Songs" 公开 CSV | Tidy Tuesday 开放数据, 衍生自 Spotify | ✅ 图结构 + 文本嵌入 + 标签 (仅元数据, 无音频) |

## 2. 发布策略

### Tier 1: 直接发布 (arXiv)
- 完整 `arxiv_multimodal_graph_agent.json`
- 节点: 45,672 论文 ID + 标签 + 特征
- 边: 57,045 引用关系
- 文本/图像/公式嵌入 (CLIP ViT-L/14, 768d)
- 原始 .md + images/ 目录 (*open access → 可发布*)

### Tier 2: 结构 + 嵌入发布 (Steam / TMDb / WikiArt)
- `*_multimodal_graph_agent.json` (仅图结构: node_ids, labels, edge lists)
- 预计算 CLIP 嵌入 (`.pt` files, N×768 浮点矩阵, 不可逆推原始文本/图像)
- 图统计与协议 manifests (节点掩码、跨划分边计数、分组重叠) 及完整性卡
- 评估划分与暴露定义 (各协议视图的 LP 消息边与负采样规则)
- **不包含**原始 .md 文本, 原始图像文件, API 原始响应

### Tier 2-lite: spotify (零代码展品)
- `spotify_multimodal_graph_agent.json` + 文本嵌入 + 评估划分
- 源 CSV 公开 (Tidy Tuesday) → 从 `domains/spotify/domain.yaml` 完全可复现

### Tier 3: 构建脚本 (所有域)
- `python main.py --domain-config domains/<name>/domain.yaml` 完整可复现
- 域 YAML 配置 + `crawlers/` + `benchmark/` 流水线
- 新增域撰写指南: `domains/README.md`

## 3. Reproducibility Statement (论文正文)

```
Reproducibility. The multi-agent framework and all benchmark code are
released under the MIT License at [URL]. The arxiv and spotify datasets
are released in full (open-access / public metadata). For Steam, TMDb,
and WikiArt, we release the graph structure (node IDs, labels, edge
lists), pre-computed CLIP embeddings, protocol view manifests, and
evaluation splits. Raw text and images are not redistributable due to
copyright restrictions; however, the complete datasets can be
reproduced by running:

    python main.py --domain-config domains/<name>/domain.yaml

Each command executes the full multi-agent pipeline (collector →
schema → graph-builder → formatter → verifier), producing the canonical
*_agent.json files and protocol views. Benchmark suites are reproduced
with:

    python benchmark/generate_protocol_views.py
    python benchmark/run_newdomain_nc.py
    python benchmark/run_four_protocol_graph_tasks.py
    python benchmark/run_four_protocol_modality_tasks.py
```

## 4. 常见问题

**Q: 其他 3 个域为什么不直接发布原始数据?**
A: Steam/TMDb/WikiArt 的原始数据因版权不可直接再发布。我们发布:
  (a) 图结构 JSON (节点 ID + 标签 + 边 + 嵌入), 即 benchmark 可直接消费的最小集合
  (b) 完整构建脚本 → 有 API key 的用户可逐位复现

**Q: 预计算嵌入是 CLIP-only, 换 encoder 怎么办?**
A: 编码脚本 `benchmark/encode_domain.py` 可重跑; 论文口径为冻结
  CLIP ViT-L/14 (文本塔处理文本/公式, 视觉塔处理图像)。

**Q: 数据集是否包含个人身份信息 (PII)?**
A: 不包含。Steam 游戏/TMDb 电影/WikiArt 画作均为公共文化产品元数据,
  不涉及个人信息。arxiv 论文为公开发表的学术文献。

## 5. 交付清单 (Camera-Ready)

- [ ] Zenodo/HuggingFace dataset DOI (arxiv full + 3 域 structure-only)
- [ ] GitHub repo: MIT LICENSE, English README
- [ ] Reproducibility section in paper
- [ ] 新增域教程 (`domains/README.md` 已含快速上手)
