# Cross-Modal Retrieval & Alignment — New Domains (P2-11, 2026-07-12)

> OpenMAG Data Quality dimension (modality retrieval via MRR/Hits@k). Uses existing
> frozen CLIP text+image embeddings — no training. Closes "modality tasks only on arxiv".
> Source: `benchmark/results/newdomain_modality/modality_all_summary.json`.

## Headline (text→image retrieval, Recall@k + MRR)

| domain | text-d / image-d | R@1 | R@5 | R@10 | MRR | CKA |
|---|---|---|---|---|---|---|
| steam | 768 / 768 | 0.469 | 0.702 | 0.771 | 0.576 | 0.006 |
| wikiart | 768 / 768 | 0.111 | 0.263 | 0.352 | 0.190 | 0.007 |
| **tmdb** | 768 / 768 | **0.783** | **0.917** | **0.940** | **0.844** | 0.005 |

(image→text retrieval is symmetric; full numbers in JSON. **tmdb image re-encoded to 768-d on 2026-07-12** — it was previously 512-d and retrieval was N/A; now it is the strongest domain.)

## Findings (honest)

- **tmdb: strongest cross-modal alignment** (R@1=0.78, MRR=0.84). Movie plot/genre text maps tightly to poster art in CLIP space — posters are designed to signal genre, so text→image retrieval is easy. (After the 768-d fix; previously N/A.)
- **steam: moderate alignment** (R@1=0.47, MRR=0.58). Game descriptions ↔ cover art are reasonably aligned — text carries genre/theme cues that match the image.
- **wikiart: weak alignment** (R@1=0.11, MRR=0.19). Painting *images* are hard to retrieve from their text (title/genre tags) — 64 fine-grained classes, and a painting's visual content is loosely tied to its text label.

**Cross-cutting:** CKA between raw text and image embeddings is low (~0.005) across all domains — expected, since CLIP's text and image towers, while aligned for CLIP's training distribution, are only weakly aligned for these domain-specific cultural-metadata contents. This corroborates the P1-14 finding: **text is the dominant, image the marginal modality** — the two are not tightly bound in these graphs.

## What this adds vs OpenMAG
- OpenMAG reports modality retrieval (MRR/Hits@k) as a Data Quality metric. We now provide it for 2/3 new visual domains (steam, wikiart) — tmdb blocked by an encoder-dim data issue (documented, fixable by re-encoding tmdb posters with CLIP-large).
- spotify is text-only (no image) → retrieval N/A by design.

## Honest scope
- Retrieval here is **frozen-CLIP zero-shot** (no learned projection). OpenMAG's retrieval is also frozen-encoder. A learned cross-modal projection would likely improve these numbers but is out of scope (frozen-encoder benchmark).
- The tmdb 512-d image encoder inconsistency is worth fixing before final release (re-encode with CLIP-large); flagged in the data-quality audit follow-up.
