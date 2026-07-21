# Canonical Statistics (single source of truth) — 2026-07-11

> P0-2 产物。四源（agent.json / manifest.json / statistics.json / graph_data.pt）对齐结果。


## 权威表（per-domain）

| domain | nodes | classes | homophily | modalities | agent.json edges (raw, benchmark loads) | manifest edges_unique (deduped) |
|---|---|---|---|---|---|---|
| steam | 2,760 | 8 | 0.5295 | text,image | 54,642 | 54,424 |
| tmdb | 13,037 | 18 | 0.2371 | text,image | 193,475 | 193,346 |
| wikiart | 21,968 | 64 | 0.8941 | text,image | 520,330 | 520,570 |
| arxiv | 45,672 | 4 | 0.9332 | text,image,formula | 57,045 | 57,045 |

## 边数差异说明（footnotes，reviewer 会问）

- **agent.json edges** = release artifact 内的原始边数（benchmark 加载此文件；arxiv 例外，见下）。
- **manifest edges_unique** = 去重后的唯一边数。两者小差异（steam 54642 vs 54424；tmdb 193475 vs 193346；wikiart 520456 vs 520570）= agent.json 含少量重复/自环，manifest 已去重。
- **arxiv 特例**：arxiv benchmark **不走 agent.json**，而走 `benchmark/processed/graph_data.pt`（edge_index=57590）。agent.json release edges=57045（去重/过滤后）。graph_data.pt 57590 为 benchmark 工作副本（含少量待去重边）。论文 arxiv 边数统一报 **57045 (release) / 57590 (benchmark)**，脚注说明。
- **steam avg_degree 约定**：avg_degree = 2E/N（无向，与 steam metadata 一致）。

## statistics.json 修正

- num_classes: 7 → **4**（旧值 7 误算 7-subcat 方案；NC benchmark 实际用 4 域 cs_si/cs_lo/math_lo/physics_socph，label_distribution 本就列 4）。

## 同质性归因（防过度声称）

- 4 域同质性值（0.237/0.530/0.894/0.933）是**自然源属性**（arxiv 引用天然同质，tmdb 共流派天然异质），**不是调参设计**。
- 仅 **steering 实验**（同域换边）才因果受控同质性。论文措辞用 "spectrum spanning natural sources"，"controlled/causal" 只留给 steering。