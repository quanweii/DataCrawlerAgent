# MAG-Forge Dataset Card — Multimodal Attributed Graph Benchmarks

> Version 2.0 · 2026-10-16 · Datasheets-for-Datasets compliant.
> Aligned with the current paper: declarative schemas, lineage-based auditing,
> four named evaluation protocols, six tasks, integrity cards.

## 1. Summary
Real-source multimodal attributed graphs (MAGs) constructed by the MAG-Forge
multi-agent framework from declarative domain schemas. Four mature benchmark
domains plus three held-out contract-reuse domains that were exercised at the
schema/construction stage only.

| Domain | Nodes | Classes | Modalities | Time / group provenance | Excluded before all views |
|---|---|---|---|---|---|
| arxiv | 45,672 | 4 | text+image+formula | month / first author | none (citations retained) |
| steam | 2,760 | 8 | text+image | release / developer | shared genre |
| tmdb | 13,037 | 18 | text+image | release / director | shared genre |
| wikiart | 21,968 | 64 | text+image | year / artist | shared style |
| spotify (held-out) | 10,000 | 6 | text | — (no trusted timestamp) | — |
| amazon / reddit (held-out) | 5,364 / 7,257 | 20 / 10 | title-only stubs | — | — |

**Tasks (six)**: node classification, link prediction, node clustering,
modality retrieval, modality alignment, modality matching.
**Protocols (four)**: transductive, split-disjoint, temporal, group-inductive.
Unavailable protocols are marked as such rather than synthesized (e.g.,
temporal is unavailable for sources without trustworthy timestamps).

## 2. Motivation
Multimodal graph benchmarks couple entity identities, modality associations,
targets, and relation definitions. Existing benchmarks fix these choices at
dataset construction; researchers who need newly available data, different
modality combinations, or their own edge rules must modify the underlying
pipeline. MAG-Forge makes construction configurable around research needs
while keeping the coupling auditable: an accepted schema is a construction
contract, deterministic audits use recorded lineage to exclude target-derived
inputs, and named evaluation protocols specify which entities and edges are
observable during learning.

## 3. Composition
- **Per node**: frozen CLIP ViT-L/14 embeddings — a 768-d text vector (text
  tower; also used for arXiv LaTeX formulas, 77-token limit) and, where
  present, a 768-d image vector (vision tower). Node-local mean fusion of
  L2-normalized modality vectors, followed by an outer normalization. Missing
  modalities are zero-filled; all-missing nodes retain zero vectors with
  task-specific eligibility.
- **Per edge**: a `relation` attribute plus its recorded source field
  (lineage). Edges are built by deterministic rules — observed citations
  (arXiv) or capped co-attribute groups (per-group connection caps, frequency
  filters); self-loops and exact duplicates are removed and endpoints
  validated. No LLM is involved in edge materialization.
- **Labels**: declared target fields — arXiv = 4 research fields;
  Steam/TMDB = the configured first genre (`genres[0]`); WikiArt = 64 art
  styles (predicted principally from artwork-title text and images).

## 4. Collection process
- **arxiv**: open-access arXiv metadata + per-paper markdown/images; citation
  edges from public citation records; specialized citation pipeline with a
  fixed category schema.
- **steam / tmdb**: public storefront / official APIs (Steam, TMDb API v3);
  co-attribute edges from metadata fields with configured caps.
- **wikiart**: public WikiArt.org artwork pages.
- **spotify**: public Tidy Tuesday "Spotify Songs" CSV (derived from Spotify);
  built through the same declarative config path.
- **amazon / reddit**: file-backed stubs transcribed from stored discovery
  records; construction-interface reuse only (no encoding/training).

## 5. Preprocessing and protocol views
The raw MAG is immutable. Lineage-based exclusions are applied **before every
view**: relations whose source field resolves to the target field (shared
genre in Steam/TMDB, shared style in WikiArt) are removed from derived views;
arXiv citations are retained (not target-derived). Feature exclusions follow
the same recorded lineage (e.g., structured genre lines removed for
Steam/TMDB; artist/style/year/biography removed for WikiArt, retaining
principally titles).

- **transductive**: stratified 60/20/20 node masks; full sanitized topology.
- **split-disjoint**: identical masks; cross-split edges removed.
- **temporal**: whole time buckets ordered chronologically (monthly for
  arXiv/Steam/TMDB, annual for WikiArt); nodes missing timestamps are excluded
  (4 Steam, 1 TMDB, 3,078 WikiArt).
- **group-inductive**: whole primary groups assigned together; nodes missing
  primary groups receive disclosed singleton keys (350/2/4 arXiv/Steam/TMDB).
- **Link prediction**: transductive edges split 60/20/20; other views use
  within-partition positives. The message graph contains training positives
  only; one fixed uniform negative per positive, excluding every true edge in
  the view. Non-transductive held-out endpoints have no message neighbors.
- Seeds: NC split seed 1 with five initialization seeds (0–4).

## 6. Distribution / license
| Domain | Source | License |
|---|---|---|
| arxiv | arXiv | CC0 (metadata); paper text under each arXiv license |
| steam | Steam Store | Steam Subscriber Agreement (research use) |
| tmdb | TMDb | **CC-BY (attribution required)**: "This product uses the TMDB API but is not endorsed or certified by TMDB." |
| wikiart | WikiArt.org | artwork images © respective owners (research/display) |
| spotify | Tidy Tuesday / Spotify | Tidy Tuesday open data; track metadata derived from Spotify |

See `DATASET_LICENSING_en.md` for the tiered release plan.

## 7. Transparency notes
- **WikiArt temporal is open-set**: 18 test classes are unseen in training and
  seen classes cover 38.85% of test nodes; temporal scores must be interpreted
  with label-support analysis, not as pure "future prediction" failure.
- **Structural access matters**: with partitions and features fixed, removing
  cross-split edges changes GraphSAGE accuracy differently by domain (WikiArt
  −9.75 points, replicated as −10.02 under a second partition; smaller
  differences are partition-sensitive).
- **Integrity cards** accompany every release, separating checked correctness
  (identity, lineage, protocol invariants) from applicability (label support,
  open-set exposure, modality dependence).
- Topology-only probes and label permutations are descriptive diagnostics,
  not leakage verdicts; leakage determinations require contradicted lineage or
  invariants.

## 8. Intended use / misuse
- **Intended**: multimodal graph model evaluation under explicit exposure
  conditions, protocol sensitivity studies, controlled construction
  experiments, benchmark customization and reuse.
- **Misuse**: reading transductive accuracy as evidence about unseen-group
  generalization; treating descriptive test-score maxima as deployment
  estimates (validation-selected and test-maximum architectures agree in only
  7/16 domain–protocol cells); interpreting an audit pass as task-suitability;
  claiming a causal multi-agent advantage (not established — see the paper's
  scope statements).

## 9. Maintenance
Maintained by the authors. To add a new domain: author one
`domains/<name>/domain.yaml` (identifier, target, relation rules, caps), run
`python main.py --domain-config domains/<name>/domain.yaml`, then encode and
benchmark through the same production path.
