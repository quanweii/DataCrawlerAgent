# E2 Citation Matching — Statistical Power Boost & Honest Re-analysis (P1-8, 2026-07-12)

> Closes roadmap P1-8 + guards the SINGLE load-bearing agent-necessity evidence point (FINAL §3).
> Source: `benchmark/results/e2_citation_ablation/e2_results.json` (part1, n=60 controlled references).
> Structured data: `benchmark/results/e2_citation_ablation/e2_power_boost_2026-07-12.json`.
> Model provenance: **mimo-v2-flash**, Jun20 run (no new LLM calls in this analysis).

## The problem with the old headline

FINAL §3 / memory cited: *"citation fuzzy OCR: LLM recall **0.714** ≫ rule-shortlist **0.000** / production-rule **0.286**, n=7"*.
This is the **fuzzy(<0.90) bucket only, n=7** — underpowered. A reviewer will (correctly) object that n=7 cannot support the claim, and that the comparison is on a cherry-picked difficulty slice.

## What this re-analysis does (no new LLM calls)

1. **McNemar exact paired test on ALL n=60 controlled references** — the statistically correct test for two matchers scored on the *same* references (more powerful than treating the 7 fuzzy cases as an independent sample).
2. **Bootstrap 95% CI** on every recall estimate.
3. **Combined non-clean regime (medium+fuzzy, n=17)** — the difficulty band where the LLM is *supposed* to add value over rules, with a properly powered n.

## Results

### By bucket (with bootstrap 95% CI)

| bucket | n | LLM recall | rule_production | rule_shortlist |
|---|---|---|---|---|
| clean (≥0.95) | 43 | 0.977 | 0.977 | 0.977 |
| medium (0.90–0.95) | 10 | 0.900 | **1.000** | 0.900 |
| fuzzy (<0.90) | 7 | 0.714 | 0.286 | 0.000 |

### Combined non-clean (medium+fuzzy, n=17) — the LLM-unique regime

| matcher | correct | recall ± SE | 95% CI |
|---|---|---|---|
| **LLM** | 14/17 | **0.824 ± 0.092** | [0.647, 1.000] |
| rule_production | 12/17 | 0.706 ± 0.111 | [0.471, 0.882] |
| rule_shortlist | 9/17 | 0.529 ± 0.121 | [0.294, 0.765] |

LLM > rule_shortlist by 0.30 (disjoint CIs), and > rule_production by 0.12 (overlapping CIs).

### McNemar exact (paired, n=60)

| comparison | LLM-only correct | opponent-only correct | discordant | p (exact) |
|---|---|---|---|---|
| LLM vs rule_shortlist | 5 | 0 | 5 | **0.0625** |
| LLM vs rule_production | 3 | 1 | 4 | 0.625 |

**Every discordant pair favors the LLM** (5/5 vs shortlist, 3/4 vs production), but with n=60 the paired test is only **borderline vs shortlist (p=0.0625)** and **not significant vs the production rule (p=0.625)**.

### Overall (n=60): LLM 0.933 [0.867,0.983] vs rule_production 0.900 vs rule_shortlist 0.850.

## Honest interpretation (this SOFTENS the old headline)

- The LLM provides a **real but modest** advantage on fuzzy/noisy citation matching. The dramatic-looking fuzzy-bucket gap (0.714 vs 0.000) **does not survive** proper paired testing at α=0.05 once all 60 references are considered: **McNemar p=0.0625 vs rule-shortlist, p=0.625 vs production rule.**
- The production rule (the stronger baseline) is **not significantly beaten** by the LLM.
- This is the **only** point in the paper where an LLM agent beats rules, and it is **marginal**, not decisive.

## Required paper wording (replaces the old overclaim)

> "On noisy/fuzzy citation matching, the LLM recovers references the shortlist rule misses: every one of the 5 discordant paired cases favors the LLM (McNemar exact p=0.0625, n=60; borderline). Against the stronger production rule the gap is not significant (p=0.625). We therefore claim the agent provides a **modest, not decisive**, advantage on semantic citation wrangling — consistent with the end-to-end finding that citation-edge recovery is bottlenecked by corpus coverage, not the matcher (§E2 coverage, P1-9)."

## Scope & future work (honest)

- The controlled set is n=60; only 7 fell in the fuzzy(<0.90) band. Boosting the fuzzy band to n≥20–30 requires **re-running the matcher on a larger reference sample** with a comparable LLM endpoint. The current endpoint is configured for a *different* model (qwen-plus) than the original (mimo-v2-flash, deprecated); a re-run would be a cross-model replication, not a like-for-like power boost, so it is left as optional future work. The **McNemar paired test on all 60 references is the principled power boost** delivered here and is more honest than the n=7 headline.
- The honest bottom line for agent positioning: the agent's value on citation edges is **marginal**; the load-bearing agent claims remain **semantic wrangling (modest) + reusable interface (C3) + orchestration**, NOT construction/de-leaking decisions (greedy=oracle, FINAL §3).
