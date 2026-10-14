# MAG-Forge / DataCrawlerAgent

MAG-Forge 是一个把真实多模态来源转换为可审计多模态属性图（MAG）benchmark 的多智能体框架。当前论文主线以 arxiv、Steam、TMDB、WikiArt 四个成熟域为核心，并用 Spotify 等域检查声明式复用。

## 六模块框架

```text
Real-world Multimodal Sources
  → Autonomous Schema Discovery
  → Schema-Guided MAG Construction
  → Protocol-Aware Integrity Auditing
  → Integrity-Guided Representation Encoding
  → Protocol-Conditioned Benchmarking & Integrity Reporting
```

生产执行链为：

```text
Collector → SchemaDiscoveryAgent → GraphBuilder → DatasetFormatter → GraphVerifier
```

- `SchemaDiscoveryAgent` 对字段做画像，确认 YAML schema 或请求 LLM 提议 schema，然后确定性检查 ID、label、关系字段和 target provenance。
- Builder 只消费已经发布的 confirmed schema。
- Verifier 检查重复实体、关系语义、覆盖率和 topology-only signal，并生成协议派生视图。
- 原始 MAG 不被协议实验覆盖；target-derived 关系只从 benchmark 派生视图中排除。
- LLM 可以规划可选步骤，但不能省略 schema gate 和最终 verifier。

## Evaluation protocols

成熟四域同时发布：

1. `transductive`：分层 60/20/20，保留完整 sanitized topology；
2. `split_disjoint`：与 transductive 使用相同 node masks，只保留 split 内边；
3. `temporal`：按真实时间排序，同一时间桶不拆，只保留 split 内边；
4. `group_inductive`：真实 primary group 整体分配，只保留 split 内边。

普通 transductive train–test 边、高 homophily 或 topology-only 高分本身都不是 leakage 结论。协议必须有真实 provenance；没有可信 timestamp 的新域会把 temporal 标成不适用，而不是伪造。

## 环境

服务器已验证环境：

```bash
cd /home/dev_qw/DataCrawlerAgent
source /home/dev_qw/miniconda3/bin/activate agent_env
```

LLM 驱动的真实流水线还需要在 `.env` 中设置：

```text
API_KEY=...
MODEL_BASE_URL=...
MODEL_NAME=...
```

## 运行声明式域

YAML 本身是可执行域声明：

```bash
python main.py --domain-config domains/steam/domain.yaml
python main.py --domain-config domains/tmdb/domain.yaml
python main.py --domain-config domains/wikiart/domain.yaml
python main.py --domain-config domains/spotify/domain.yaml
```

可用 JSON 覆盖运行参数：

```bash
python main.py --domain-config domains/steam/domain.yaml --config task_override.json
```

详细字段见 `domains/README.md`。

## 核心 benchmark 与复核

生成协议视图及运行四协议六任务：

```bash
PYTHONPATH=. python benchmark/generate_protocol_views.py
PYTHONPATH=. python benchmark/run_newdomain_nc.py --help
PYTHONPATH=. python benchmark/run_four_protocol_graph_tasks.py --help
PYTHONPATH=. python benchmark/run_four_protocol_modality_tasks.py --help
```

当前最终证据已经完成，不需要为了验证结果重新训练 GPU 模型。可直接重新计算报告和一致性检查：

```bash
PYTHONPATH=. python benchmark/generate_report_card.py
PYTHONPATH=. python benchmark/validate_final_evidence.py
PYTHONPATH=. python benchmark/aggregate_validate_four_protocol_tasks.py
PYTHONPATH=. python benchmark/run_reusability_audit.py
PYTHONPATH=. python -m pytest -q tests
```

已验证结果：

- 64 个 dataset×protocol×modality cells；
- 384 个 model cells；
- 1,920 个五初始化种子训练结果；
- 480 个 label-free LP/clustering 联合训练种子、120 个 modality-matching 种子；
- 24 个 retrieval/alignment protocol×pair cells；
- 400 次 label permutation 与 40 次 degree-preserving rewires；
- 最终证据审计 1,739/1,739 checks PASS；
- 六任务审计 2,665/2,665 checks PASS；
- 自动测试 11/11 PASS；
- 四个成熟域 integrity cards 全部 PASS；
- Steam、TMDB、WikiArt、held-out Spotify 的生产复用合同全部 PASS。

## 关键产物

```text
agents/schema_discovery.py                 production schema gate
agents/graph_verifier.py                   provenance/protocol integrity audit
benchmark/data/protocols.py                protocol view generator
benchmark/run_newdomain_nc.py              protocol-conditioned NC
benchmark/run_four_protocol_graph_tasks.py protocol-aware LP + clustering
benchmark/run_four_protocol_modality_tasks.py retrieval + alignment + matching
benchmark/aggregate_validate_four_protocol_tasks.py six-task evidence audit
benchmark/run_topology_nulls.py            topology null diagnostics
benchmark/generate_report_card.py          current-only integrity cards
benchmark/validate_final_evidence.py        final numerical consistency audit
benchmark/sync_protocol_manifests.py        guarded manifest/result metadata sync
benchmark/run_reusability_audit.py          production YAML reusability audit
```

结果与论文权威文件：

```text
docs/paper_logic_DnB_FINAL_2026-07-11.md
docs/experiment_results_FINAL_decontaminated_2026-07-16.md
benchmark/results/newdomain_nc/protocol_matrix/
benchmark/results/four_protocol_tasks/
benchmark/results/protocol_views/
benchmark/results/topology_nulls.{json,md}
benchmark/results/report_cards.{json,md}
benchmark/results/final_evidence_audit.{json,md}
benchmark/results/reusability_audit.{json,md}
```

## 结论边界

- 构建与评估正确性由 provenance、重复检查、边端点与 split invariants 证明，不由模型高分证明。
- 模型成绩用于说明 benchmark 的效用、难度与协议敏感性。
- 多模态融合不是普遍占优：Steam/TMDB 四协议均为正，Arxiv 多数 view 由 text 更强，WikiArt group-inductive 中 image 单模态更强。
- WikiArt temporal 是 label-space emergence 与 severe distribution shift 叠加的 stress test：train-support test mass 仅 0.3885、可见类 prior TVD 为 0.965，不能当作一般 closed-set temporal 结果。
- sample-level pretraining overlap、真人效率研究和真实外部用户复用仍需要外部资源/受试者，代码不能伪造这些证据。
