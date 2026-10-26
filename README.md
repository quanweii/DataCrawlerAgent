# MAG-Forge

MAG-Forge is an open-source multi-agent system for constructing auditable
multimodal attributed graph benchmarks. A coordinator connects source collection,
schema discovery, deterministic construction, formatting and integrity checks.
The `build-mag-dataset` skill helps a researcher author a validated domain YAML
and trial-build a graph. Four frozen benchmarks support four protocols and six
downstream tasks.

This release accompanies **MAG-Forge: Multi-Agent Construction of Auditable
Multimodal Graph Benchmarks**. The resource version is `www2027-20261027`.

## Start here

| Goal | Entry point |
|---|---|
| Try graph construction without an API or model download | [Offline skill example](#offline-skill-example) |
| Inspect or use the skill | [skills/build-mag-dataset/SKILL.md](skills/build-mag-dataset/SKILL.md) |
| Obtain the four datasets and experiment records | [Artifact files and checksums](artifacts/README.md) |
| Check published data and retained results without fitting models | `python scripts/verify_release.py` |
| Reproduce training and evaluation | [Reproducibility guide](docs/REPRODUCIBILITY.md) |
| Locate evidence cited in the paper | [Anonymous review guide](README_ANONYMOUS.md) |

## Installation

Use Python 3.10 or 3.11. The tested project environment uses PyTorch 2.8.0 and
torch-geometric 2.6.1. A CPU installation is sufficient for unit tests, the skill
example and saved-result verification:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
```

For GPU training, install the matching PyTorch GPU wheel and then use
`requirements-pinned.txt`. LLM-driven construction additionally reads `API_KEY`,
`MODEL_BASE_URL` and `MODEL_NAME` from your own `.env`; see `.env.example`.
Frozen artifact loading, saved-result checks and the example below do not make
LLM calls.

## Offline skill example

The bundled example is deliberately synthetic software test data, separate from
the research datasets. From the repository root:

```bash
python scripts/skill_dataset.py inspect --source examples/skill_demo/items.json
python scripts/skill_dataset.py trial --config examples/skill_demo/domain.yaml --output trial_output
```

Expected result: 40 nodes, 140 edges, two classes and a valid schema contract.
The structural trial creates temporary label-free placeholders, never frozen
features or scientific evaluation scores. To make the skill discoverable in an
assistant, install the `skills/build-mag-dataset` folder in its skill directory;
the same entry point is also supplied under `.claude/skills/`.

## Frozen research resources

| Dataset | Nodes | Classes | Modalities | Raw edges |
|---|---:|---:|---|---:|
| arXiv | 45,672 | 4 | text, image, formula | 57,045 |
| Steam | 2,760 | 8 | text, image | 54,642 |
| TMDB | 13,037 | 18 | text, image | 193,475 |
| WikiArt | 21,968 | 64 | text, image | 520,330 |

The repository includes the real frozen resources as hash-checked archive parts,
not download placeholders or Git LFS pointers. Install both resource groups:

```bash
python scripts/install_artifacts.py
python scripts/verify_release.py
```

The installer only reads files shipped with this repository. It restores graph
IDs/labels/edges, the original 768-dimensional CLIP vectors, protocol tensors,
source-derived time/group metadata, manifests, evidence and saved checkpoints.
It verifies each part, reconstructed archive and installed file. Raw source text
and images are excluded. Existing different local files are not overwritten
without `--force`.

The protocols are Trans, Split, Temporal and Group; the tasks are node
classification, link prediction, clustering, retrieval, alignment and matching.
Current LP code enforces disjoint negative prediction objects as well as disjoint
positive splits. Historical results are retained with their source/version
labels; [the review guide](README_ANONYMOUS.md) identifies the active LP records.

## Construct a domain

Use the skill to inspect a file-backed source and confirm its ID, scalar target,
relations and class support. Full execution uses the shared production pipeline:

```bash
python main.py --domain-config domains/steam/domain.yaml
```

This full run requires source records and extracted per-entity text/images at
the configured locations, plus an LLM endpoint. A new API source additionally
needs acquisition/extraction adapters. See [domains/README.md](domains/README.md).
arXiv PDF extraction uses an external backend configured by
`PDF_EXTRACT_KIT_PATH` and optional `PDF_EXTRACT_PYTHON`.

## Code and data terms

Code is covered by the [MIT License](LICENSE). Source-derived research artifacts
retain the per-source access and attribution conditions documented in
[docs/DATASET_LICENSING_en.md](docs/DATASET_LICENSING_en.md).
This product uses the TMDB API but is not endorsed or certified by TMDB.

Frozen-row alignment details are recorded in [artifacts/DATA_NOTES.md](artifacts/DATA_NOTES.md).
