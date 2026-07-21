# tmdb Image Re-encoding → 768-d Unification (2026-07-12)

> Data-consistency fix requested by author: all benchmark headline embeddings unified to 768-d.

## The bug
`benchmark/processed/tmdb/image_embeddings.pt` was **512-d** (CLIP-base) while every other headline embedding (steam/tmdb/wikiart/arxiv text+image, arxiv formula, spotify text) is **768-d** (CLIP-large). `tmdb/image_clip_large_embeddings.pt` was also mislabeled 512-d. Root cause: tmdb posters were encoded by an older pipeline that used CLIP-base; the current `ImageEncoder` uses `CLIPModel.config.projection_dim=768`.

## The fix
1. Backed up the old 512-d files → `benchmark/processed/tmdb/_archive_512d_bug/`.
2. Re-encoded tmdb posters with the current `encode_domain.py --domain tmdb --modalities image` (CLIP-ViT-Large-Patch14, projection_dim=768): **N=13036, dim=768** ✓. `image_clip_large_embeddings.pt` is the same file (now correctly 768-d).
3. Verified **all 10 headline embedding tensors across 5 domains are 768-d** (steam/tmdb/wikiart/arxiv text+image, arxiv formula, spotify text).

## Cascade re-runs (tmdb text+image feature changed 1280-d → 1536-d)
All tmdb benchmark experiments that use image features were re-run; leaked/structural metrics (LP-ceiling, per-relation homophily) are unchanged by construction.

| experiment | key change (new 1536-d) |
|---|---|
| NC headline | gcnii+text+image 0.636 → **0.638** (≈unchanged) |
| LP headline | gcn random/hard/hard_degree = 0.946/0.927/0.889 |
| leak_profile | lp_ceiling_pruned **0.154 (unchanged, structure-only)**; nc_preserved mlp 0.672 / gcn 0.489 (updated) |
| per-seed / significance / multisplit | re-aggregated; tmdb split-std ≤0.021 (still stable) |
| clustering / few-shot / modality-missing / advanced / labelnoise | re-run; magnitudes ≈unchanged |
| **modality retrieval** | **N/A → R@1=0.783 / MRR=0.844** (was blocked by the dim mismatch; now the STRONGEST domain) |

## Key insight (confirms P1-14)
Re-encoding tmdb image 512→768 **barely changed NC/LP** (≤0.002) — because **text dominates** these tasks (P1-14: drop_text −0.12, drop_image ≈0). The image dim only mattered where the image embedding is the *object* of evaluation: **cross-modal retrieval**, which requires a shared-dim space. The fix therefore (a) unifies dims and (b) unlocks tmdb retrieval (now the best of the three visual domains, R@1=0.78 — movie posters signal genre strongly).

## What remains (optional)
- `encoder_comparison_tmdb.json`: its clip_large cell used the buggy 512-d image; for full consistency re-run `scripts/run_multi_encoder_nc.py --domain tmdb` (re-encodes clip_base/siglip image too). The headline comparison is unaffected; this is the secondary encoder study only.
- All headline numbers updated in the per-domain summary JSONs + `leak_profile_table.json` + significance/multisplit aggregates. Per-domain docs' tmdb numbers shifted by ≤0.01 (qualitative findings unchanged); the JSONs are the source of truth.
