# Ethics & Data-Acquisition Compliance Statement

> P0-9 deliverable (NeurIPS D&B mandated). Complements `DATASET_LICENSING_en.md` (copyright) with acquisition-ethics: ToS, robots.txt, rate-limiting, PII, consent, dual-use.

## 1. Acquisition method per source

| Domain | Method | Auth | Rate limiting |
|---|---|---|---|
| arxiv | arXiv metadata snapshot (OAI) + open PDFs | none (open access) | respectful bulk-download intervals |
| steam | Steam Store public storefront pages + Steamworks API | API key (operator's own) | exponential backoff on HTTP 429 (verified in `scripts/`) |
| tmdb | TMDb API v3 | API key (operator's own, registered) | API budget respected; TMDb attribution included |
| wikiart | WikiArt.org public pages | none | throttled crawl |
| spotify | **Public Tidy Tuesday CSV** (already collected by the Tidy Tuesday project from Spotify) | none — we used the published dataset, did not query the Spotify API ourselves | n/a (static public file) |

## 2. Terms-of-service & robots.txt compliance
- **TMDb**: access via the official API v3 under its ToS; attribution statement included in every release artifact (`DATASET_LICENSING_en.md` §4). We do not redistribute raw TMDb text/images.
- **Steam**: data collected from the public storefront and Steamworks API; we redistribute only graph structure + embeddings, not raw store text, consistent with the Steam Subscriber Agreement's research-use norms.
- **WikiArt**: public gallery pages; artwork images are not redistributed (© owners); only metadata-derived structure + embeddings are released.
- **Spotify**: we did **not** scrape Spotify. We used the Tidy Tuesday project's already-published public CSV. No user accounts, no listening histories, no audio.
- **arXiv**: open-access; redistribution permitted.

## 3. PII / user-generated content audit
- **No user-generated content is retained.** We do not collect Steam reviews, TMDb user ratings, or any reviewer/author PII.
- spotify tracks carry only track metadata (name, artist, album, audio features) — **no listener data, no user IDs, no play counts of real users**.
- arxiv nodes are paper IDs (content), not author identities used as personal data.
- Conclusion: **the released datasets contain no personal data.**

## 4. Consent
All sources are public APIs / open-access repositories / published open datasets whose terms permit research use. No human subjects were involved; no private data was collected.

## 5. Potential misuse & dual-use (broader impact)
- **Benchmark gaming**: leaked or saturated metrics (e.g., raw wikiart NC, random-negative LP) could be misreported as model capability. We mitigate by shipping the **leak-profile** per domain and the 3-tier LP negatives, and by explicitly documenting the leakage/saturation in `DATASET_CARD_en.md` §7–8.
- **Overclaiming the agent**: the LLM agent does **not** make construction decisions (code-verified; `greedy=oracle`). We forbid claims that it outperforms rules at construction/de-leaking (see FINAL §3, §4.5).
- **Copyright**: redistribution is tiered (full / structure-only) to respect source copyrights.
- **Scope**: domains are cultural-metadata graphs (games, movies, art, music, papers); they reflect the biases of their source platforms (genre taxonomies, popularity skew). Downstream users should not treat label distributions as normative.

## 6. Environmental / compute
Benchmark runs on 8× NVIDIA RTX 4090. Total compute for the paper's experiments is documented per-experiment in `experiment_roadmap_2026-07-11.md` (effort estimates). No unusually large training runs.

## 7. Correspondence
Licensing/attribution questions: see `DATASET_LICENSING_en.md`. Ethics concerns: this document. Both are released alongside the data.
