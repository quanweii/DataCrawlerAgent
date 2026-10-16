# Dataset Licensing & Reproducibility Strategy

> P0 F1: 审稿人评 "releasing 4 datasets" 时必须看到可操作的获取路径。

## 1. 数据来源与版权分类

| 域 | 原始数据 | 版权状态 | 可再发布内容 |
|---|---|---|---|
| **arxiv** | Open-access 论文 (arXiv.org) | CC/BY 许可, 无版权障碍 | ✅ 全集: 节点/边/文本/图像/公式/嵌入 |
| **steam** | Steam 游戏元数据 (Steamworks API + 公开页面爬取) | Valve 版权, 不可再发布原始文本 | ⚠️ 图结构 + 嵌入 + 标签 + 构建脚本 |
| **tmdb** | TMDb 电影元数据 (TMDB API v3) | TMDB 版权, 不可再发布原始文本/图像 | ⚠️ 图结构 + 嵌入 + 标签 + 构建脚本 |
| **wikiart** | WikiArt 画作元数据 (公开爬取) | 画作图像各有版权, 元数据可争议 | ⚠️ 图结构 + 嵌入 + 标签 + 构建脚本 |

## 2. 发布策略

### Tier 1: 直接发布 (arXiv)
- 完整 `arxiv_multimodal_graph_agent.json` (13 MB)
- 节点: 45,672 论文 ID + 标签 + 特征
- 边: 57,045 引用关系
- 文本/图像/公式嵌入 (CLIP ViT-L/14, 768d)
- 原始 .md + images/ 目录 (*open access → 可发布*)

### Tier 2: 结构 + 嵌入发布 (Steam / TMDb / WikiArt)
- `*_multimodal_graph_agent.json` (仅图结构: node_ids, labels, edge lists)
- 预计算 CLIP 嵌入 (`.pt` files, N×768 浮点矩阵, 不可逆推原始文本/图像)
- 图统计 (同质性, 度分布, 标签分布)
- LP 负采样 CSV (train/val/test pos/neg/hard/hard_degree)
- **不包含**原始 .md 文本, 原始图像文件, API 原始响应

### Tier 3: 构建脚本 (所有域)
- `main.py --domain {steam,tmdb,wikiart,arxiv}` 完整可复现
- 域配置文件 (`*_config.json`)
- crawlers/ 图构建源码
- 编码脚本 + benchmark pipeline

## 3. Reproducibility Statement (论文正文)

```
Reproducibility. The multi-agent framework and all benchmark code are
released under the MIT License at [URL]. The arxiv dataset is released
in full (open-access papers). For Steam, TMDb, and WikiArt, we release
the graph structure (node IDs, labels, edge lists), pre-computed CLIP
embeddings, and link-prediction negative samples. Raw text and images
are not redistributable due to copyright restrictions; however, the
complete datasets can be reproduced by running:

    python main.py --domain steam --config steam_config.json
    python main.py --domain tmdb --config tmdb_config.json
    python main.py --domain wikiart --config wikiart_config.json

using the provided API key placeholders and crawler scripts. Each
command executes the full multi-agent pipeline (collector → graph-builder
→ formatter → verifier), producing the canonical *_agent.json files.
All benchmark tables (NC headline §3.1, LP headline §3.2) are
reproducible with:

    python benchmark/run_newdomain_nc.py --domain {domain} --gpu 0
    python benchmark/run_newdomain_lp.py --domain {domain} --gpu 0
```

## 4. 对审稿人的 FAQ

**Q: 你说 releasing 4 datasets, 但我只看到 1 个 (arxiv)。其他 3 个在哪里?**
A: Steam/TMDb/WikiArt 的原始数据因版权不可直接再发布。我们发布:
  (a) 图结构 JSON (节点 ID + 标签 + 边 + 嵌入), 即 benchmark 可直接消费的最小集合
  (b) 完整构建脚本 → 有 API key 的审稿人可逐位复现
  这遵循 MAG benchmark 的标准实践 (OpenMAG 同样不发布 TMDb/WikiArt 原始数据)。

**Q: 预计算嵌入是 CLIP-only。如果我想换 encoder 怎么办?**
A: 编码脚本 `benchmark/encode_domain.py --encoder-family {clip,siglip,dino}` 支持多族。
  我们发布嵌入是为了方便直接跑 benchmark; 换 encoder 只需重跑编码 (离线, 每域 ~20-40 min)。

**Q: 数据集是否包含个人身份信息 (PII)?**
A: 不包含。Steam 游戏/TMDb 电影/WikiArt 画作均为公共文化产品元数据,不涉及个人信息。
  arxiv 论文为公开发表的学术文献。

## 5. 交付清单 (Camera-Ready)

- [ ] Zenodo/HuggingFace dataset DOI (arxiv full + 3 域 structure-only)
- [ ] GitHub repo: MIT LICENSE, pyproject.toml, English README
- [ ] Reproducibility section in paper (§X)
- [ ] `add_new_domain.md` tutorial (step-by-step guide for community contributors)
- [ ] Static leaderboard (GitHub Pages / HF Space) showing NC + LP headline tables
