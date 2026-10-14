# ICLR 2027 Submission Readiness Audit (updated 2026-09-11)

## Verdict

**代码与当前核心实验已闭环；论文尚未达到可直接提交状态。** 当前适合开始写 ICLR 主稿，不应把现有本科论文或旧 related-work 文档直接提交。

## What is closed

- Production five-agent chain is executable and protected by mandatory schema/verifier gates.
- Four mature domains have provenance-sanitized, four-protocol derived views.
- Core node-classification evidence contains 64 dataset×protocol×modality cells, 384 model cells and 1,920 seeded runs.
- All six tasks drawn in the framework now use the sanitized four-protocol suite: 480 joint LP/clustering training runs, 120 matching runs, and 24 retrieval/alignment cells.
- 400 label permutations and 40 verified degree-preserving rewires are complete.
- Temporal label support now separates label emergence from within-support prior shift.
- NC numerical audit is 1,739/1,739 PASS; the new six-task audit is 2,665/2,665 PASS; framework tests are 11/11 PASS.
- Current-only integrity cards pass for all four mature domains.
- Production YAML contract passes on Steam, TMDB, WikiArt and held-out Spotify.

## P0 blockers before submission

1. **No ICLR manuscript exists.** The repository contains an older Chinese undergraduate thesis, whose architecture and claims predate the current six-module logic. A new anonymous ICLR 2027 paper must be written using the official template and fit within the 9-page main-text limit.
2. **Related work is stale.** `related_work_FINAL_2026-08-06.md` still makes old structural-leakage, LP-ceiling, universal decontamination and exclusivity claims. It is explicitly marked stale and must be rebuilt from primary sources.
3. **Automation evidence is not yet a strong causal baseline.** Stored LLM cost figures show feasibility, but do not prove reduced human effort. The main paper must limit RQ1 accordingly or add a controlled rule-only/manual-intervention comparison. Expert time estimates cannot be presented as a user study.
4. **Reusability evidence is internal.** Spotify supports held-out declarative configuration, not external-user usability. C4/RQ5 must use “held-out configuration reuse,” unless a real external user study is completed.
5. **Dataset release compliance is not documented.** Before anonymous release, add source URLs, licenses/terms, redistribution policy, provenance, personally identifiable information review, removal procedure and per-domain dataset cards.
6. **Submission artifacts are missing.** Prepare an anonymous code/data package, environment lock, one-command reproduction routes, expected runtime/storage, hardware table and result hashes.
7. **Required policy text is missing.** ICLR 2027 requires an AI-use statement. Add a reproducibility statement and, because this work releases datasets/crawls real sources, a focused ethics/data-governance statement.

## Scientific positioning recommended for the paper

Use one central claim:

> MAG-Forge turns real multimodal sources into provenance-audited MAG benchmarks and exposes how one dataset changes under explicit deployment protocols.

Do not use model performance as proof of construction correctness. The contribution structure should be:

1. production schema-to-MAG multi-agent system;
2. deterministic provenance and integrity auditing;
3. protocol-conditioned benchmark release;
4. six-task empirical evidence that topology value, task difficulty and modality compatibility depend on domain and deployment protocol.

The WikiArt temporal result should be written as a stress test combining label-space emergence and severe distribution shift: coverable test mass 0.3885, observed accuracy 0.0101, seen-class prior TVD 0.965. It is not a leakage result.

## Recommended nine-page allocation

| Section | Pages |
|---|---:|
| Introduction and contributions | 1.0 |
| Related work | 0.75 |
| Six-module method and multi-agent guarantees | 2.0 |
| Protocol definitions and integrity invariants | 1.25 |
| Experimental setup | 0.75 |
| Main six-task protocol/modality results | 1.75 |
| Analysis, limitations and conclusion | 1.5 |

Put full 64-cell tables, per-seed values, schema audits, topology-null details, dataset cards and reproduction commands in the appendix/supplement.

## Official ICLR 2027 constraints checked

- Genuine abstract deadline: **2026-09-18 23:59 AOE**.
- Full-paper deadline: **2026-09-25 23:59 AOE**.
- Double blind; identity disclosure in paper or supplement can cause desk rejection.
- Main text limit: **9 pages** at initial submission; references excluded and appendices allowed.
- Mandatory AI-use statement; reproducibility and ethics statements are recommended/appropriate here.

Official sources:

- https://iclr.cc/Conferences/2027/AuthorGuidelines
- https://iclr.cc/Conferences/2027/CallForPapers
- https://iclr.cc/Conferences/2027/AIPolicyForAuthors

## Go/no-go

- **Go for writing and submission preparation:** yes.
- **Upload the current repository documents as the paper:** no.
- **Minimum defensible submission:** resolve P0 items 1–2 and 5–7, and strictly narrow RQ1/RQ5 claims.
- **Stronger competitive submission:** additionally add a controlled automation baseline and one prospective unseen-domain run frozen before seeing the outcome.
