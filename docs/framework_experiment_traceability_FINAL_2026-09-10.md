# MAG-Forge Framework–Experiment Traceability (2026-09-11)

本表用于防止论文主张、生产代码与实验产物脱节。`PASS` 表示已有可重复检查；`EXTERNAL` 表示不能由当前服务器自行补造的外部证据。

| Module / claim | Production implementation | Direct evidence | Status |
|---|---|---|---|
| 1. Real-world multimodal sources | domain collectors, raw JSON/PDF and `cs_lo` assets | four mature raw datasets and graph data-quality audit | PASS |
| 2. Autonomous schema discovery | `agents/schema_discovery.py`; confirm/discover modes; deterministic validator | schema lineage tests; production YAML reusability audit | PASS |
| 3. Schema-guided MAG construction | `DomainGraphBuilderAgent`; `build_graph_from_config`; confirmed schema from blackboard | Steam/TMDB/WikiArt/Spotify graph contract checks | PASS |
| 4. Protocol-aware integrity auditing | `GraphVerifierAgent`; duplicate/entity audit; provenance exclusions; protocol generation | manifests; 0 group overlap; 0 cross-split edges in inductive views | PASS |
| 5. Integrity-guided representation encoding | frozen per-node encoders; pre-encoding target-field removal; label-free mean fusion | 64-cell modality matrix and recorded representation policy | PASS for current frozen/mean pipeline |
| 6. Protocol-conditioned benchmarking | six tasks over four protocol views; task-specific exposure contracts | NC 1,920 seeds + LP/clustering 480 joint seeds + matching 120 seeds; 2,665/2,665 new checks | PASS |

## Research-question closure

| RQ | Evidence | Supported conclusion | Boundary |
|---|---|---|---|
| RQ1 Automation | stored E2 cost records + executable YAML pipeline | schema/audit automation uses few LLM calls in stored runs | human-time values remain expert estimates |
| RQ2 Integrity | relation lineage, entity/edge audits, protocol invariants | target-derived relations are deterministically excluded from benchmark views | high topology score/homophily is diagnostic only |
| RQ3 Protocol sensitivity | 16 richest-modality headline cells + paired comparisons | accuracy and rankings vary materially by deployment protocol | WikiArt temporal combines open-set label emergence with severe prior/distribution shift |
| RQ4 Representation robustness | all 64 protocol×modality cells | fusion value is domain- and protocol-dependent | no universal multimodal-gain claim |
| RQ5 Reusability | production audit on 3 mature YAML domains + held-out Spotify | standard file-backed co-attribute domains reuse one code path | not a real external-user study |
| RQ6 Task portability | six tasks × four protocols; 4 mature domains | one sanitized protocol suite supports classification, prediction, clustering, retrieval, alignment and matching | task metrics establish applicability, not construction correctness |

## Numerical closure

- Dataset×protocol×modality cells: **64**.
- Model cells: **384**.
- Seeded training runs: **1,920**.
- Additional label-free LP/clustering training runs: **480**; modality-matching runs: **120**.
- Total current four-protocol seeded training runs: **2,520**.
- Retrieval/alignment: **24** protocol×modality-pair cells and **48** retrieval directions.
- Topology nulls: **400** label permutations and **40** degree-preserving rewires; every rewire passed degree-sequence verification.
- Final recomputation: **1,739/1,739 checks PASS**.
- Six-task recomputation, isolation and document audit: **2,665/2,665 checks PASS**.
- Framework tests: **8/8 PASS**.
- Current integrity cards: **4/4 mature domains PASS**.
- Production YAML reusability contract: **4/4 tested domains PASS**.

## Multi-agent logic closure

The mandatory production order is:

```text
Collector
  → SchemaDiscoveryAgent (profile → propose/confirm → validate → publish)
  → GraphBuilder (consume confirmed schema)
  → DatasetFormatter
  → GraphVerifier (audit → derived protocol views → integrity artifacts)
```

An LLM-generated orchestration plan may choose optional acquisition or cleaning actions, but `_ensure_required_stages` restores the schema and verifier gates if omitted. Target-derived raw relations do not trigger destructive rewriting: the verifier uses lineage to exclude them from derived benchmark views. Actual graph-coverage defects can still request a bounded Builder reprocess.

## Claim boundary and remaining external work

`PASS` now covers the six tasks drawn in the framework under the current sanitized four-protocol views. It does not cover every possible benchmark variant: legacy few-shot, cross-domain, noise, neighbor, steering and IGC tables remain outside the current core unless separately rerun under the same contract.

The following remain `EXTERNAL` and must not be fabricated:

- sample-level pretraining overlap, which needs a searchable training-sample index or fingerprint corpus;
- a real participant efficiency study;
- a real external-user reuse study.

Spotify is valid held-out configuration evidence, not a substitute for an external user. Because its source lacks a trustworthy timestamp, temporal evaluation is correctly unavailable rather than synthesized.
