# Cross-Domain LP Transfer — Scope Decision (P1-7, 2026-07-12)

> Closes roadmap P1-7 via the roadmap-sanctioned "scope 缩为 NC-only" option
> (the alternative — porting `run_newdomain_crossdomain.py` to LP — is declined, with justification).

## Decision

**The paper's cross-domain transfer claim is NC-only (node classification, linear-probe 12/12 negative; see P1-6). Cross-domain LP transfer is out of scope, by design, not by omission.**

## Why cross-domain LP is not a meaningful benchmark here (the structural argument)

Each released MAG's edges are **domain-specific co-attribute relations**, not a universal relation type:

| domain | edge relations (co-attribute) |
|---|---|
| steam | same_developer, same_publisher, shared_category, shared_genre |
| tmdb | same_director, shared_actor, shared_genre |
| wikiart | same_artist, shared_style |
| arxiv | citation (cross-subdomain) |

A link-prediction model learns to score **one domain's** relation semantics (e.g., "two games share a developer"). Transferring that LP encoder to another domain means scoring, e.g., tmdb's `shared_actor` edges with a model trained on steam's `same_developer` edges. These are **unrelated relation types** — there is no shared edge semantics to transfer. So cross-domain LP would test "does a developer-co-occurrence predictor generalize to actor-co-occurrence?", which is not a standard or well-posed transfer benchmark.

This is fundamentally different from **cross-domain NC transfer**, where the node feature space IS shared (all domains use the same frozen CLIP text/image encoder → same 768-d space), so transferring a feature→label mapping is well-defined (and, as P1-6 shows, fails: features are domain-bound).

## What the paper says (consistent with this scope)

FINAL §2 C4: "*跨域 linear-probe 12/12 负增益*" — NC only. No claim of cross-domain LP transfer is made. The benchmark's cross-domain experiment is NC transfer (the standard OGB-style transfer setting), reported under both linear-probe (12/12 negative) and full-finetune (neutral) protocols (P1-6).

## Reviewer guard

If a reviewer asks "why no cross-domain LP?": because LP edges are domain-specific co-attribute relations with no shared semantics across domains; the meaningful cross-domain transfer benchmark for this benchmark family is NC (shared feature space). Cross-domain LP transfer would conflate "feature transfer" with "relation-semantics transfer" and is not well-posed. The within-domain LP table (4 domains × 6 models × 3 negative-sampling regimes, with leak-free Hits@k) is the LP contribution; cross-domain LP is intentionally out of scope.

## Optional pilot (not run; documented for completeness)

A minimal 1-pair pilot (train LP GCN on steam edges → score tmdb edges with the frozen conv) would, by the above argument, be expected to fail (negative transfer). It is not run because the negative result is structurally guaranteed by the domain-specificity of the relations and would not add information beyond the NC transfer result. If a reviewer insists, the pilot is ~2h to implement via the existing `run_newdomain_crossdomain.py` transfer harness adapted to the LP decoder.
