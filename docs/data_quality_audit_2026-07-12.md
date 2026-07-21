# Data Quality Audit — Leak-free vs. Dedup / No-overlap

**Date:** 2026-07-12
**Scope:** the 5 released multimodal-attribute graphs
`data/{arxiv,steam,tmdb,wikiart,spotify}_datasets/*_multimodal_graph_agent.json`
and their LP splits under `data/{domain}_lp/`.
**Motivation:** Reviewer attack surface (paper_logic_DnB_FINAL_2026-07-11.md §4.5; experiment_roadmap §I):
*"leak-free ≠ 去重/无重叠"*. Reviewers want EXPLICIT audits of (A) duplicate nodes,
(B) transductive LP train/test node overlap, (C) multi-edge folding. This document
provides them with real, unmassaged numbers.
**Reproducible:** `python3 scripts/data_quality_audit.py` → `benchmark/results/data_quality_audit.json`.

## Headline conclusion (TL;DR)

The 5 released graphs are **clean** with respect to all three audited properties.

- **(A) Duplicate nodes:** 0 duplicate node ids, 0 duplicate entity keys in every domain.
  Node `id` always equals the canonical entity id (`arxiv_id`/`appid`/`movie_id`/`painting_id`/`track_id`),
  0 mismatches. Title/name collisions exist (tmdb 502, wikiart 1089) but are **legitimate
  distinct entities sharing generic titles** (e.g. "Untitled" x1040 by many artists;
  "Prey"/"Dracula" = different films), not duplicate entities.
- **(B) LP train/test node overlap:** fully transductive as expected (87–99.7% test-node
  overlap with train), and **0.00% exact edge leakage** in every domain (no test_pos edge
  reappears in train_pos). Inductive "tail" (test edges with both endpoints unseen in
  train) is tiny: arxiv 2.33%, spotify 5.33%, tmdb 0.02%, wikiart 0.00%.
- **(C) Multi-edge folding:** every graph is a **simple graph** — 0 self-loops, 0 duplicate
  undirected edges, 0 exact (src,tgt,relation) duplicates across all 5 domains. The builder
  deduplicates each node pair to a single edge (verified: steam's 54,642 undirected pairs
  each carry exactly 1 of 4 possible relations; 0 reciprocal directed pairs).

The released artifacts support a **leak-free + dedup** claim. One minor reproducibility
note (not a leak, not a defect): `protocol.json` edge-relation counts drift slightly from
the final graph (|Δ| ≤ 240 per relation); see Findings §4.

---

## (A) Duplicate nodes

Per domain:

| domain | #nodes | #unique ids | dup id keys | dup rate % | entity-key | dup entity keys | name collisions |
|--------|-------:|------------:|------------:|-----------:|------------|----------------:|----------------:|
| arxiv  | 45,672 | 45,672 | 0 | 0.000% | arxiv_id | 0 | n/a (no title) |
| steam  |  2,760 |  2,760 | 0 | 0.000% | appid | 0 | 1 |
| tmdb   | 13,037 | 13,037 | 0 | 0.000% | movie_id | 0 | 502 |
| wikiart| 21,968 | 21,968 | 0 | 0.000% | painting_id | 0 | 1,089 |
| spotify| 10,000 | 10,000 | 0 | 0.000% | track_id | 0 | 0 |

`entity_id_mismatch_count = 0` for every domain (node `id` ≡ entity key).

**Name/title collisions are NOT duplicate entities** (verified by inspection of the
top offenders):

- **wikiart** (1,089 colliding titles, 4,839 nodes): driven by generic art titles —
  "Untitled" x1040 (distinct artworks by 100+ different artists, each painting_id carries a
  unique artist-prefix + hash), "Composition" x119, "Hamlet" x55 (John Austen illustration
  series), "Peaceable Kingdom" x47 (Edward Hicks series), "Self-Portrait" x47, "Still Life" x41.
- **tmdb** (502 colliding titles, 1,061 nodes): generic/remake film titles — "Prey" (4 films),
  "The Enforcer", "The Three Musketeers", "The Island", "The Killer", "Dracula" (distinct
  TMDB movie_ids = distinct releases). Size distribution: 452 titles x2, 43 x3, 7 x4.
- **steam** (1 collision): "Total War: SHOGUN 2" appid 201270 (base game) vs 34330
  (*Fall of the Samurai* standalone expansion) — genuinely distinct Steam products.
- **spotify** (0): all track names unique (or empty — some `name` fields are `""`, excluded
  from the collision test).

**arxiv note.** The arxiv graph nodes carry only `label`/`label_index` (no title/abstract),
so fuzzy-title dedup is **not applicable from the released artifact alone**. Exact
`arxiv_id` dedup was performed: 0 duplicates across all 45,672 ids. Because each id is
unique and each maps to exactly one of the 4 sub-domain labels (cs_lo 11,012 / cs_si 11,929 /
math_lo 9,555 / physics_socph 13,176), **no paper appears in two sub-domains** — the
sub-domain collections are disjoint by construction.

---

## (B) Transductive LP train/test overlap

LP splits: `data/{domain}_lp/{train_pos,test_pos}.csv` (columns `source,target`).
Overlap metric: `|nodes(train_pos) ∩ nodes(test_pos)| / |nodes(test_pos)|`.

| domain | train edges | test edges | node overlap % | test edges fully-unseen % | exact edge overlap (undirected) |
|--------|------------:|-----------:|---------------:|--------------------------:|--------------------------------:|
| arxiv  | 45,637 | 5,704 | **87.04%** | 2.33% | **0** (0.00%) |
| steam  | 43,540 | 5,442 | **99.30%** | 0.00% | **0** (0.00%) |
| tmdb   | 154,678 | 19,334 | **99.62%** | 0.02% | **0** (0.00%) |
| wikiart| 416,456 | 52,057 | **99.65%** | 0.00% | **0** (0.00%) |
| spotify| 13,952 | 1,744 | **90.57%** | 5.33% | **0** (0.00%) |

Interpretation:
- **Node overlap (87–99.7%) documents the transductive setting honestly.** A transductive
  LP benchmark *should* have high node overlap (test edges among nodes seen at train time);
  these numbers confirm that and are reported as-is, not hidden.
- **0.00% exact edge leakage everywhere** is the key leak-freedom evidence: **no positive
  test edge is a duplicate of a train edge**, directed or undirected, in any of the 5 domains.
- The small **fully-unseen** fraction (test edges whose *both* endpoints never appear in
  train) is the genuinely-inductive tail. arxiv 2.33% and spotify 5.33% reflect those graphs'
  lower density / higher fragmentation; tmdb/wikiart are essentially pure transductive. This
  is reported transparently so reviewers can see there is no hidden fully-inductive leakage
  masquerading as transductive.

All 5 domains have LP split files present. (`test_neg_hard_degree` and `test_neg_random`
also exist as `test_neg.csv` / `test_neg_hard_degree.csv`; not needed for the overlap audit.)

---

## (C) Multi-edge folding (parallel edges & self-loops)

| domain | #edges | unique undirected | dup undirected edges | dup undirected % | exact (src,tgt,rel) dups | self-loops | multi-relation pairs |
|--------|-------:|------------------:|---------------------:|-----------------:|-------------------------:|-----------:|---------------------:|
| arxiv  | 57,045 | 57,045 | 0 | 0.000% | 0 | 0 | 0 |
| steam  | 54,642 | 54,642 | 0 | 0.000% | 0 | 0 | 0 |
| tmdb   | 193,475 | 193,475 | 0 | 0.000% | 0 | 0 | 0 |
| wikiart| 520,330 | 520,330 | 0 | 0.000% | 0 | 0 | 0 |
| spotify| 17,440 | 17,440 | 0 | 0.000% | 0 | 0 | 0 |

Every released graph is a **simple graph**: no parallel undirected edges, no self-loops,
no exact directed duplicates.

This is non-trivial for the 4 co-attribute domains (steam/tmdb/wikiart/spotify), where two
nodes could legitimately share *multiple* attributes (e.g. two Steam games with the same
genre, developer, and publisher) and thus generate parallel edges. The builder **resolves
this by emitting exactly one edge per node pair** — verified directly for steam: all 54,642
undirected pairs carry exactly 1 relation (`rels-per-pair = {1: 54642}`), 0 pairs carry >1
relation, 0 reciprocal directed pairs. (arxiv is single-relation citation, so the question
is moot there.)

**Edge-endpoint integrity:** 0 edges reference a missing node id in any domain (every
`source`/`target` is a valid node id).

---

## Findings & flags

1. **No data-quality defects found in the released graphs.** Zero duplicate nodes, zero
   duplicate/parallel edges, zero self-loops, zero train→test edge leakage across all 5
   domains. The "leak-free + dedup" claim is supported.

2. **Name/title collisions are NOT a problem** but should be pre-empted in the paper text:
   reviewers spot-checkping `title` fields will see 502 tmdb / 1089 wikiart shared titles.
   State explicitly that entity identity is keyed on `movie_id`/`painting_id` (all unique),
   and that title collisions are generic reused strings (remake films; "Untitled" etc.).

3. **arxiv fuzzy-title dedup was not possible** from the released graph (nodes lack title).
   This is a scope limitation to disclose: exact-id dedup is proven (0 dups); fuzzy
   near-duplicate (same title, different arxiv_id) is not auditable from the artifact. If a
   reviewer demands it, the raw paper metadata in `data/{cs_lo,cs_si,math_lo,physics_socph}_papers/`
   would need a separate title-similarity pass.

4. **Minor reproducibility note (not a defect, not a leak):** `data/{domain}_lp/protocol.json`
   `edge_relations` counts drift slightly from the final released graph:

   | domain | graph total | protocol sum | per-relation max |Δ| |
   |--------|------------:|-------------:|-------------------:|
   | steam  | 54,642 | 54,424 | 123 (same_publisher) |
   | tmdb   | 193,475 | 193,346 | 71 (shared_genre) |
   | wikiart| 520,330 | 520,570 | 240 (shared_style) |
   | spotify| 17,440 | 17,440 | 0 (exact) |

   The `protocol.json` was generated at split time on a slightly earlier edge set. The
   released graph is the source of truth for the paper's headline edge counts; the protocol
   file is a split-log. Recommend regenerating `protocol.json` from the final graph for
   consistency, or adding a one-line note that protocol counts are pre-final-build.

5. **Transductive setting is honest and well-documented.** No domain claims inductive
   purity it doesn't have; the fully-unseen tail (arxiv 2.33%, spotify 5.33%) is small and
   reported. If desired, these tails could be removed by filtering test edges to the
   transductive core, but at <6% this is optional polish, not a leak.

---

## Honest verdict

**The released graphs are clean enough to claim leak-free + dedup.** There are no duplicate
nodes, no duplicate/parallel/self-loop edges, and no train→test edge overlap in any of the
5 domains. The only items to address before submission are presentational: (i) disclose that
arxiv fuzzy-title dedup is out of scope for the artifact, (ii) pre-empt the title-collision
question with a sentence on entity-key identity, and (iii) optionally regenerate
`protocol.json` so its edge counts match the final graphs.
