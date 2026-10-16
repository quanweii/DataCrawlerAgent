# Dataset Licensing & Distribution Strategy (English)

> P0 deliverable. Reviewers evaluating "releasing N datasets" must see an actionable acquisition path per domain.

## 1. Source & copyright classification

| Domain | Raw data | Copyright | Redistributable |
|---|---|---|---|
| **arxiv** | Open-access papers (arXiv.org) | CC/BY licenses, no obstruction | ✅ Full set: nodes/edges/text/images/formula/embeddings |
| **steam** | Steam game metadata (Steamworks API + public pages) | Valve, raw text NOT redistributable | ⚠️ Graph structure + embeddings + labels + build scripts |
| **tmdb** | TMDb movie metadata (TMDb API v3) | TMDb, raw text/images NOT redistributable | ⚠️ Graph structure + embeddings + labels + build scripts |
| **wikiart** | WikiArt artwork metadata (public scrape) | Artwork images © owners; metadata debatable | ⚠️ Graph structure + embeddings + labels + build scripts |
| **spotify** | Tidy Tuesday "Spotify Songs" public CSV | Tidy Tuesday open data; derived from Spotify | ✅ Graph structure + text embeddings + labels (metadata only; no audio) |

## 2. Release tiers

### Tier 1 — Full release (arxiv)
- Full `arxiv_multimodal_graph_agent.json`
- Nodes: 45,672 paper IDs + labels + features
- Edges: 57,045 citation relations
- Text/image/formula embeddings (CLIP-ViT-L/14, 768-d)
- Raw `.md` + `images/` (open-access → redistributable)

### Tier 2 — Structure + embeddings (Steam / TMDb / WikiArt)
- `*_multimodal_graph_agent.json` (graph structure only: node IDs, labels, edge lists)
- Pre-computed CLIP embeddings (`.pt`, N×768 float matrices; not invertible to raw text/image)
- Graph statistics (homophily, degree, label distribution, **leak-profile**)
- LP negative-sample CSVs (train/val/test pos/neg/hard/hard_degree)
- **Excluded**: raw `.md` text, raw image files, raw API responses

### Tier 2-lite — spotify (zero-code exhibit)
- `spotify_multimodal_graph_agent.json` + text embeddings + LP splits
- Source CSV is public (Tidy Tuesday) → fully reproducible from `domains/spotify/domain.yaml`

### Tier 3 — Build scripts (all domains)
- `python main.py --domain-config domains/<name>/domain.yaml` fully reproducible
- Domain YAML configs + `crawlers/` + `benchmark/` pipeline
- The `build-mag-dataset` skill for adding new domains

## 3. Reproducibility statement (paper text)

> **Reproducibility.** The multi-agent framework and all benchmark code are released under the MIT License at [URL]. The arxiv and spotify datasets are released in full (open-access / public metadata). For Steam, TMDb, and WikiArt, we release the graph structure (node IDs, labels, edge lists), pre-computed CLIP embeddings, and link-prediction negative samples. Raw text and images are not redistributable due to copyright; however, the complete datasets can be reproduced by running `python main.py --domain-config domains/<name>/domain.yaml`. Exact pinned dependencies, encoder hashes, and SHA-256 checksums of every released artifact and frozen split are provided in `REPRODUCIBILITY.md`.

## 4. Per-source attribution
- **TMDb**: "This product uses the TMDB API but is not endorsed or certified by TMDB." (required by TMDb ToS)
- **arxiv**: arXiv metadata under CC0.
- **WikiArt**: artwork images © respective artists/estates; metadata used for research.
- **Spotify/Tidy Tuesday**: data derived from Spotify via the Tidy Tuesday project.
