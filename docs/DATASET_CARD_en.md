# AMMAG Dataset Card — Multimodal Attributed Graph Benchmark (English)

> Version 1.1 · 2026-07-11 · Datasheets-for-Datasets compliant.
> Auto-generated fields: see `benchmark/results/dataset_card/manifest.json`. Canonical statistics: `docs/canonical_statistics_2026-07-11.md`. Frozen splits: `data/{domain}_splits/`.

## 1. Summary
A set of cross-modal multimodal attribute graphs (MAGs) spanning a natural homophily spectrum (0.24 → 0.93), built by a single reusable multi-agent framework. Four primary domains plus one zero-code demonstration domain (spotify) that proves the framework's extensibility.

| Domain | Nodes | Unique edges | Classes | Homophily | Modalities | Edge type |
|---|---|---|---|---|---|---|
| arxiv | 45,672 | 57,045 | 4 | **0.933** | text+image+**formula** | citation (cites) |
| steam | 2,760 | 54,424 | 8 | 0.529 | text+image | shared_genre/same_developer/same_publisher/shared_category |
| tmdb | 13,037 | 193,346 | 18 | **0.237** | text+image | shared_genre/same_director/shared_actor |
| wikiart | 21,968 | 520,570 | 64 | 0.894 | text+image | same_artist/shared_style |
| spotify | 10,000 | 17,440 | 6 | 0.741 | text | same_artist/same_album |

**Tasks**: Node Classification (NC), Link Prediction (LP; random/hard/hard_degree negatives + standard Hits@k over 1000 filtered negatives), edge-relation-type prediction, cross-domain transfer.

## 2. Motivation
Existing MAG benchmarks (e.g., OpenMAG, ICML'26) ship fixed, uncontrollable graph structures and cannot answer "how does GNN behavior change if we change the edge-construction rule?" We make **homophily a controllable axis** (E1 steering: fix nodes/features/labels/splits/seeds, swap only the edge relation) and automate/audit construction via a multi-agent framework + declarative YAML + an interactive skill.

## 3. Composition
- **Per node**: CLIP-ViT-L/14 768-d text embedding + 768-d image embedding (arxiv additionally has a 768-d formula embedding; spotify is text-only). Missing modalities → zero vector.
- **Per edge**: a `relation` attribute (used for E1 steering filtering at load time).
- **Labels**: arxiv = 4 research fields; steam = 8 game genres; tmdb = 18 movie genres; wikiart = 64 art styles; spotify = 6 playlist genres.
- **Class imbalance**: e.g., steam Racing 58 vs Action 1403; tmdb History 84 vs Drama 2690; wikiart smallest class ~21.

## 4. Collection Process
- **arxiv**: public arXiv metadata + per-paper markdown/images; citation edges from `data/{sub}_graphs/*.csv`; 4 sub-fields merged. Built by the multi-agent pipeline (reference_graph_builder → formatter).
- **steam/tmdb**: public storefront / API (Steam Store, TMDb API v3); edges by deterministic co-occurrence + density-capped sampling rules.
- **wikiart**: WikiArt.org artworks; `same_artist` / `shared_style` edges.
- **spotify**: Tidy Tuesday "Spotify Songs" public CSV (GitHub raw: rfordatascience/tidytuesday), derived from Spotify; 10k stratified subsample of 28,356 unique tracks. Built **zero-code** via the `build-mag-dataset` skill + `config_driven.build_graph_from_config` (1 YAML + data, no framework Python).
- Edge construction is **deterministic rules, no LLM in semantic edge decisions** (code-verified). The LLM agent's load-bearing roles are semantic wrangling (citation fuzzy matching) and the reusable interface — see `docs/paper_logic_DnB_FINAL_2026-07-11.md` §3.

## 5. Preprocessing
- Features: each modality L2-normalized (CLIP convention); concatenation globally L2-normalized (for LP dot-product decoding).
- Graph: undirected, self-loops removed, deduplicated.
- Splits: NC = stratified per-class 60/20/20 (fixed seed=1); LP = OGB-style 80/10/10 edges + random/hard/hard_degree negatives.

## 6. Distribution / License
| Domain | Source | License |
|---|---|---|
| arxiv | arXiv | CC0 (metadata); paper text under each arXiv license |
| steam | Steam Store | Steam Subscriber Agreement (research use) |
| tmdb | TMDb | **CC-BY (attribution required)**: "This product uses the TMDB API but is not endorsed or certified by TMDB." |
| wikiart | WikiArt.org | artwork images © respective owners (research/display) |
| spotify | Tidy Tuesday / Spotify | Tidy Tuesday open data; track metadata derived from Spotify |

⚠ Scraped domains (steam/wikiart) require per-source ToS review before release; we recommend releasing graph structure + features rather than raw images to reduce copyright risk. See `DATASET_LICENSING_en.md` and `ETHICS_STATEMENT.md`.

## 7. Known Issues / Leakage (transparent) ⚠
- **wikiart label leakage**: `same_artist`/`shared_style` edges are **definitionally homophilic** (same artist/style ⇒ same label). Evidence: feature-free label-propagation ceiling on the raw graph = 0.921. **Fix**: after cross-split-edge pruning, LP-ceiling drops to **0.024** (< majority 0.025) → leakage eliminated. We report NC on the pruned (leak-free) graph and always co-report the LP-ceiling. Each domain ships a **leak-profile** (`benchmark/results/leak_profile_table.json`).
- **LP saturation**: on high-homophily graphs, random-negative LP degenerates to "same label ⇒ edge" — we use hard/hard_degree negatives and report Δ(random − hard_degree).
- **statistics.json num_classes**: an earlier arxiv `num_classes` was mis-recorded as 7 (correct: 4); **fixed 2026-07-11**.

## 8. Intended Use / Misuse
- **Intended**: GNN/MAG model evaluation, homophily/heterophily studies, cross-domain transfer, controlled graph experiments, extensibility demonstration.
- **Misuse**: reporting raw wikiart NC 0.97 as true performance (leakage); using random-negative LP as the headline (saturated); claiming the LLM agent outperforms rules at edge construction (falsified — `greedy=oracle`; see FINAL §3).

## 9. Maintenance
Maintained by the authors. To add a new domain, follow `docs/ADD_NEW_DOMAIN.md` (or run the `build-mag-dataset` skill): author one `domains/<name>/domain.yaml`, run `python main.py --domain-config domains/<name>/domain.yaml`, then encode + benchmark. Zero framework Python for standard co-attribute domains.
