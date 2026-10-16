# Ethics & Data-Acquisition Compliance Statement

> Complements `DATASET_LICENSING_en.md` (copyright) with acquisition ethics:
> ToS, robots.txt, rate-limiting, PII, consent, dual-use.

## 1. Acquisition method per source

| Domain | Method | Auth | Rate limiting |
|---|---|---|---|
| arxiv | arXiv metadata snapshot (OAI) + open PDFs | none (open access) | respectful bulk-download intervals |
| steam | Steam Store public storefront pages + Steamworks API | API key (operator's own) | exponential backoff on HTTP 429 |
| tmdb | TMDb API v3 | API key (operator's own, registered) | API budget respected; TMDb attribution included |
| wikiart | WikiArt.org public pages | none | throttled crawl |
| spotify | public Tidy Tuesday CSV (already published by the Tidy Tuesday project) | none — the published dataset was used; the Spotify API was not queried | n/a (static public file) |
| amazon / reddit | file-backed records transcribed from stored discovery data | none | n/a |

## 2. Terms-of-service & robots.txt compliance
- **TMDb**: access via the official API v3 under its ToS; the attribution
  statement is included in every release artifact. Raw TMDb text/images are
  not redistributed.
- **Steam**: data collected from the public storefront and Steamworks API;
  only graph structure and embeddings are redistributed, not raw store text,
  consistent with the Steam Subscriber Agreement's research-use norms.
- **WikiArt**: public gallery pages; artwork images are not redistributed
  (© owners); only metadata-derived structure and embeddings are released.
- **Spotify**: Spotify was not scraped. The Tidy Tuesday project's already
  published public CSV was used. No user accounts, no listening histories, no
  audio.
- **arXiv**: open access; redistribution permitted.

## 3. PII / user-generated content audit
- **No user-generated content is retained.** No Steam reviews, TMDb user
  ratings, or reviewer/author PII are collected.
- Spotify tracks carry only track metadata — no listener data, no user IDs,
  no real-user play counts.
- arXiv nodes are paper IDs (content); first-author strings are used only as
  group provenance for the group-inductive protocol, not as personal data.
- Conclusion: **the released datasets contain no personal data.**

## 4. Consent
All sources are public APIs, open-access repositories, or published open
datasets whose terms permit research use. No human subjects were involved; no
private data was collected.

## 5. Potential misuse & dual-use (broader impact)
- **Benchmark gaming**: scores obtained under transductive exposure or on
  open-set temporal cohorts can be misreported as general capability. We
  mitigate by publishing named protocol views with explicit exposure rules and
  integrity cards that separate checked correctness from applicability, and by
  documenting label support and open-set conditions.
- **Overclaiming the agent**: deterministic tools enforce construction
  contracts; no causal advantage of multi-agent coordination over equally
  equipped single-agent or scripted workflows is claimed.
- **Copyright**: redistribution is tiered (full release vs. structure +
  embeddings) to respect source copyrights.
- **Scope**: domains are cultural-metadata graphs (games, movies, art, music,
  papers) reflecting the biases of their source platforms; label distributions
  should not be treated as normative.

## 6. Environmental / compute
Benchmark suites run on a single NVIDIA RTX 4090; per-domain training grids
complete in minutes. No unusually large training runs.

## 7. Correspondence
Licensing/attribution questions: see `DATASET_LICENSING_en.md`. Ethics
concerns: this document. Both are released alongside the code and data.
