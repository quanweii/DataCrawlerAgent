# Reproducibility Statement

> Enables reproduction of the released code, protocol views, and benchmark
> procedures. Aligned with the current paper (four protocols, six tasks,
> validation-AP selection for LP/clustering, frozen diagnostics).

## 1. Software environment (pinned)

| Component | Version |
|---|---|
| Python | 3.9 |
| torch | 2.8.0+cu128 |
| torch-geometric | 2.6.1 |
| transformers | 4.52.4 |
| GPU | a single NVIDIA RTX 4090 (24 GB) |

Full pinned dependency list: [`requirements-pinned.txt`](../requirements-pinned.txt).

```bash
pip install -r requirements-pinned.txt
```

LLM-driven stages (schema discovery in discovery mode, orchestration planning)
read `API_KEY` / `MODEL_BASE_URL` / `MODEL_NAME` from `.env`
(see `.env.example`). The schema-comparison experiments used `qwen-plus`,
temperature 0.2, 1,200-token output caps.

## 2. Encoder provenance

| Encoder | Revision | Notes |
|---|---|---|
| CLIP-ViT-Large-Patch14 (768-d) | HF `openai/clip-vit-large-patch14`, snapshot `32bd64288804d66eefd0ccbe215aa642df71cc41` | text tower for text and LaTeX formulas (77-token limit); vision tower for images; frozen, no fine-tuning |

This is a **frozen-encoder benchmark**: encoding and mean fusion fit no
split-dependent state, so embeddings are reusable across protocols whenever
entity identities, retained content, and encoder settings are unchanged.

## 3. Splits, seeds, and selection

- **NC**: stratified 60/20/20 node masks, split seed 1; five initialization
  seeds (0–4) per cell; checkpoints selected on validation accuracy. A
  second-partition campaign (split seed 2) reruns the fused-modality grid.
- **LP / clustering**: shared label-free graph autoencoder (≤200 epochs);
  architecture and checkpoints selected by validation AP. LP message graph
  contains training positives only; one fixed uniform negative per positive
  excluding every true edge in the view.
- **Matching**: entity-disjoint train/validation/test with same-label
  different-entity negatives; validation-AP checkpointing.
- **Retrieval / alignment**: deterministic evaluations over frozen
  representations (24 protocol–pair cells); no training.
- Historical architecture maxima summarized by mean test accuracy are
  descriptive, not deployment estimates; per-model values are retained by the
  runners below.

## 4. Artifact integrity

Protocol manifests record eligible node counts, missing-time coverage,
exclusions, group overlap, cross-split edge counts, and unavailable protocols
alongside source fields and parent artifacts. SHA-256 checksums for released
data artifacts accompany the tiered data release (see
`DATASET_LICENSING_en.md`); raw source data is not redistributed.

## 5. Reproducing each component

| Component | Command |
|---|---|
| domain graph | `python main.py --domain-config domains/<name>/domain.yaml` |
| protocol views | `python benchmark/generate_protocol_views.py` |
| NC suite | `python benchmark/run_newdomain_nc.py` |
| LP / clustering | `python benchmark/run_four_protocol_graph_tasks.py` |
| retrieval / alignment / matching | `python benchmark/run_four_protocol_modality_tasks.py` |
| second-partition replication | `python benchmark/run_split_seed2_nc.py` |
| held-out contract builds | `python benchmark/run_heldout_contract.py` |
| topology nulls | `python benchmark/run_topology_nulls.py` |
| schema/gate mechanisms | `python benchmark/run_framework_mechanisms.py` && `python benchmark/validate_framework_mechanisms.py` |
| release stress test | `python benchmark/run_artifact_release_study.py` && `python benchmark/validate_artifact_release_study.py` (opt-in gate: `benchmark/artifact_release.py`) |
| report cards | `python benchmark/generate_report_card.py` |
| final evidence audit | `python benchmark/validate_final_evidence.py` |
| six-task consistency audit | `python benchmark/aggregate_validate_four_protocol_tasks.py` |
| production YAML reuse audit | `python benchmark/run_reusability_audit.py` |
| unit tests | `pytest tests/` (protocols, framework integrity, orchestrator feedback) |

## 6. What is not claimed (honest scope)
- No matched-tools/budget end-to-end comparison of multi-agent coordination
  against a single agent or scripted workflow is included.
- The release stress test covers known, explicitly injected contract
  violations; it is not an estimate of natural failure rates, unseen-fault
  robustness, or human error prevalence.
- Frozen encoders may contain unverified pretraining overlap with benchmark
  content; no sample-level overlap evidence is claimed.
- Temporal/group results inherit source coverage and label shifts; the WikiArt
  temporal view is an open-set condition (18 unseen test classes).
