# MAG-Forge

**MAG-Forge** is a multi-agent framework for constructing **auditable multimodal
attributed graph (MAG) benchmarks** from real-world sources. Instead of a
one-off data pipeline, a research requirement becomes a declarative schema —
entity identifiers, a prediction target, and supported relation rules — and the
framework carries that contract through acquisition, graph building, formatting,
verification, and protocol-conditioned evaluation.

```
Real-world multimodal sources (APIs / files)
   → Autonomous schema discovery or researcher YAML
   → Schema-guided MAG construction
   → Protocol-aware integrity auditing (lineage, duplicates, invariants)
   → Integrity-guided frozen encoding + fusion
   → Protocol-conditioned benchmarking & integrity cards
```

Production execution chain:

```
Collector → SchemaDiscoveryAgent → GraphBuilder → DatasetFormatter → GraphVerifier
```

- **Schema as a contract.** Builders only materialize relations declared in an
  accepted schema; a deterministic gate checks IDs, target/relation fields, and
  class support before anything is built.
- **Lineage-based auditing.** Relations derived directly from the target field
  (e.g., shared-genre edges for a genre target) are excluded from derived views;
  feature exclusions follow recorded provenance. The raw MAG artifact is never
  rewritten.
- **Named evaluation protocols.** Each release publishes transductive,
  split-disjoint, temporal, and group-inductive views when provenance supports
  them, with manifests recording cross-split edges, group overlap, and
  unavailable protocols. Sources without trustworthy timestamps mark temporal
  *unavailable* rather than fabricating an order.
- **Integrity cards.** Every release ships a card separating checked
  correctness (identity, lineage, protocol invariants) from applicability
  (label support, open-set exposure, modality dependence).

## Repository layout

| Path | Contents |
|---|---|
| `agents/` | Role agents: collector, schema discovery, graph builders, formatter, verifier, orchestrator (`react_agent.py`, `orchestrator_agent.py`, `schema_discovery.py`) |
| `crawlers/` | Source adapters (config-driven + Steam/TMDb/WikiArt/arXiv) |
| `domains/` | Declarative per-domain YAML schemas (steam, tmdb, wikiart, spotify, amazon, reddit, cleveland) |
| `tools/` | Executable tools invoked by the agents |
| `prompts/` | Role prompts |
| `main.py`, `orchestrator.py` | Entry points |
| `benchmark/` | Protocol generation, six-task evaluation suite, audits, release gate (`artifact_release.py`), validation runners |
| `scripts/` | Data preparation / rebuild / verification helpers |
| `configs/` | Example legacy JSON crawl configs |
| `tests/` | Protocol, framework-integrity, orchestrator-feedback tests |
| `docs/` | Reproducibility statement, dataset card, licensing, ethics |

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # or use conda
pip install -r requirements.txt                     # or requirements-pinned.txt for exact versions
cp .env.example .env                                # fill in your LLM endpoint
```

Pinned versions, encoder provenance, and per-artifact SHA-256 checksums are
listed in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Usage

Build a benchmark from a domain declaration:

```bash
python main.py --domain-config domains/steam/domain.yaml
```

Add a new file-backed source by writing a `domain.yaml` (identifier, target,
relation rules, caps) — the schema gate, builders, audits, and protocol views
run through the same production path. Run the test suite:

```bash
pytest tests/
```

Reproduce the benchmark suite (protocol views → six tasks → audits):

```bash
python benchmark/generate_protocol_views.py
python benchmark/run_newdomain_nc.py
python benchmark/run_four_protocol_graph_tasks.py
python benchmark/run_four_protocol_modality_tasks.py
python benchmark/validate_final_evidence.py
```

## Data release and licensing

See [`docs/DATASET_LICENSING_en.md`](docs/DATASET_LICENSING_en.md) for the
full tiered plan. In brief:

- **arXiv / Spotify** — full release (open-access / public metadata).
- **Steam / TMDb / WikiArt** — graph structure + frozen CLIP embeddings +
  labels + rebuild scripts; raw text/images are not redistributable, but the
  complete dataset is reproducible via `python main.py --domain-config ...`.
- **All domains** — deterministic build scripts and audit trail.

This product uses the TMDB API but is not endorsed or certified by TMDB.

## License

Code is released under the [MIT License](LICENSE). Released data artifacts
follow the per-domain terms in `docs/DATASET_LICENSING_en.md`; source data
remains subject to the respective access/licensing terms.
