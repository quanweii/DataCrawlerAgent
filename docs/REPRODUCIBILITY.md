# Reproducibility

Resource version: `www2027-20261027`. Run commands from the repository root.

## Environment

Use Python 3.10 or 3.11 and `requirements-pinned.txt`; tests additionally use
`requirements-dev.txt`. CPU suffices for artifact checks and the skill example.
The paper uses frozen CLIP ViT-L/14 (`openai/clip-vit-large-patch14`, revision
`32bd64288804d66eefd0ccbe215aa642df71cc41`), with 768-dimensional vectors and
77-token text/formula inputs. Using released features requires no encoder download.

## Install and verify fixed resources

```bash
python scripts/install_artifacts.py
python scripts/install_artifacts.py --verify-only
python scripts/verify_release.py
```

`artifacts/manifest.json` specifies all archive parts and restored-file SHA-256
values. `review_output/RELEASE_PROVENANCE.json` maps source records to anonymized
release copies. Graph exports retain IDs, labels and edges and omit raw titles,
descriptions and images; feature files and protocol tensors are byte-identical
to their source snapshots. The original feature archives retain extra IDs or
missing modalities: align by ID. TMDB image features omit ID `348877`, which the
original loader zero-fills; paired tasks use observed nonzero modality rows.

## Generate protocol views

```bash
python benchmark/generate_protocol_views.py --domains arxiv,steam,tmdb,wikiart --seed 1 --output-dir trial_output/protocol_views
```

The installed time/group metadata cache reconstructs the original protocol
assignments without raw descriptions or images. To use fresh source metadata,
give an explicit metadata configuration through the source pipeline.

## Training commands

Each runner takes one domain. The following is a full-fusion example; scientific
NC coverage also evaluates each single modality and, for arXiv, each two-modality
combination. Keep the retained per-cell records and use validation selection.

```bash
for domain in arxiv steam tmdb wikiart; do
  modality=text+image
  if [ "$domain" = arxiv ]; then modality=text+image+formula; fi
  python benchmark/run_newdomain_nc.py --domain "$domain" --modalities "$modality" --protocols all --init-seeds 5 --split-seed 1
  python benchmark/run_four_protocol_graph_tasks.py --domain "$domain" --modality "$modality" --protocols all --seeds 5 --split-seed 1
  python benchmark/run_four_protocol_modality_tasks.py --domain "$domain" --protocols all --seeds 5
done
```

NC uses split seed 1 and initialization seeds 0-4, validation-accuracy checkpoint
and architecture selection. Graph tasks train on training positives and select
the shared LP/clustering encoder by mean validation AP. All positive and negative
prediction objects are disjoint. Matching uses entity-disjoint masks and selects
validation AP. Retrieval and CKA are deterministic frozen-feature diagnostics.

For all NC modalities, use `text`, `image`, `text+image`; arXiv additionally uses
`formula`, `text+formula`, `image+formula`, `text+image+formula`. Models are
`mlp,gcn,gat,graphsage,gin,gcnii`; `gcnii` names the custom initial-residual model
called ResGCN in the paper. The second-partition script is
`python benchmark/run_split_seed2_nc.py --help`; inspect its options and output
directory before scheduling the full campaign.

Historical aggregation scripts remain for record inspection. The maintained
release check is `scripts/verify_release.py`; it validates the frozen records
and active repaired LP selections rather than assuming all historical files
were regenerated with the current sampler. Do not overwrite historical results
with a new run before copying or versioning them.

## Construction and extraction

`scripts/skill_dataset.py` performs source inspection and a structural trial.
`main.py --domain-config domains/<name>/domain.yaml` runs production agents and
requires configured source/media paths plus an LLM endpoint. New APIs need
collectors/extractors. arXiv extraction requires a separately installed
pdf-extract-kit backend, configured by `PDF_EXTRACT_KIT_PATH` and optional
`PDF_EXTRACT_PYTHON`; the code contains no machine-specific interpreter path.

## Checks

```bash
python -m pytest -q tests
python scripts/skill_dataset.py trial --config examples/skill_demo/domain.yaml --output trial_output
python scripts/verify_release.py
```

These checks do not train models, call an LLM, or fetch raw source media.

Frozen-row alignment details are recorded in [artifacts/DATA_NOTES.md](../artifacts/DATA_NOTES.md).
