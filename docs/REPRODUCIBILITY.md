# Reproducibility Statement

> P0-8 deliverable. Enables bit-level reproduction of every released artifact and benchmark number.

## 1. Software environment (pinned)

Exact versions used to produce all reported results:

| Component | Version |
|---|---|
| Python | 3.9.23 |
| torch | 2.8.0+cu128 (CUDA 12.8, cuDNN 9.1.0) |
| torch-geometric | 2.6.1 |
| transformers | 4.52.4 (tokenizers 0.21.4) |
| numpy | 2.0.2 · scikit-learn 1.6.1 · scipy 1.13.1 · pandas 2.3.3 |
| GPU | 8× NVIDIA RTX 4090 (24 GB) |

Full pinned dependency list: [`requirements-pinned.txt`](../requirements-pinned.txt). Install:
```bash
pip install -r requirements-pinned.txt
# torch wheel matched to CUDA 12.8; PyG 2.6.1 with torch-scatter/torch-sparse built for torch 2.8.0+cu128
```

## 2. Encoder provenance

| Encoder | Revision | Notes |
|---|---|---|
| CLIP-ViT-Large-Patch14 (768-d) | HF snapshot `32bd64288804d66eefd0ccbe215aa642df71cc41` | default for text+image (+formula via text branch); HF_HUB_OFFLINE=1 during encoding |
| SigLIP (1152-d), CLIP-base (512-d) | encoder-comparison arm | see `benchmark/processed/{domain}/` |

Encoder embeddings are frozen (no fine-tuning) — this is a **frozen-encoder benchmark**.

## 3. Split & seed protocol

- **NC**: stratified per-class 60/20/20, `split_seed=1`; headline = 5-init-seed mean±std (seeds 1/43/85/127/169). See `benchmark/results/dataset_card/manifest.json`.
- **LP**: OGB-style 80/10/10 edges (`seed=42`); negatives random/hard/hard_degree from `scripts/lp_hard_negative_sampler.py`. Standard Hits@k computed on-the-fly over 1000 filtered random negatives (file-based ratio=1 makes tabulated Hits@3/10 degenerate, so they are computed separately — see `benchmark/run_newdomain_lp.py::_hitsk_metrics`).
- NC splits for the 4 primary domains are frozen under `data/{domain}_splits/`. spotify NC uses the deterministic `split_seed=1` stratified split (regenerated identically each run; no stored file needed).

## 4. Artifact integrity (SHA-256)

[`checksums.sha256`](../checksums.sha256) lists SHA-256 for:
- All 5 released graphs: `data/{arxiv,steam,tmdb,wikiart,spotify}_datasets/*_multimodal_graph_agent.json`
- Frozen NC splits (`data/{domain}_splits/`)
- LP split files (`data/{domain}_lp/{test_pos,test_neg_hard_degree,train_pos}.csv`)
- Encoder embeddings (`benchmark/processed/**/*embeddings*.pt`)

Verify after download:
```bash
sha256sum -c checksums.sha256
```

## 5. Reproducing each artifact

| Artifact | Command |
|---|---|
| arxiv graph | multi-agent pipeline → `data/arxiv_datasets/arxiv_multimodal_graph_agent.json` (reference_graph_builder + formatter) |
| steam/tmdb/wikiart graph | `python main.py --domain-config domains/{steam,tmdb,wikiart}/domain.yaml` |
| spotify graph (zero-code) | `python main.py --domain-config domains/spotify/domain.yaml` (or `build-mag-dataset` skill) |
| embeddings | `python benchmark/encode_domain.py --domain {domain} --gpu 0 --modalities text,image` |
| LP negatives | `python scripts/lp_hard_negative_sampler.py --domain {domain}` |
| NC table | `python benchmark/run_newdomain_nc.py --domain {domain} --models mlp,gcn,gat,graphsage,gin,gcnii --seeds 5` |
| LP table (incl. Hits@k) | `python benchmark/run_newdomain_lp.py --domain {domain} --models mlp,gcn,gat,graphsage,gin,gcnii --seeds 5` |
| leak-profile | `benchmark/results/leak_profile_table.json` (aggregated) |

## 6. What is NOT claimed (honest scope)
- The LLM agent does **not** make construction/de-leaking decisions (`greedy=oracle`; see FINAL §3). Numbers are identical under a rule-based greedy builder.
- Encoder set is frozen CLIP-family (+ SigLIP/base for comparison); a trainable-encoder arm is out of P0 scope (optional P2-7).
- See `docs/paper_logic_DnB_FINAL_2026-07-11.md` §4.5 for the full overclaim-guard list.
