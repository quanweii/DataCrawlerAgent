# Cross-Domain Transfer — Dual-Protocol Honest Report (P1-6, 2026-07-12)

> Closes roadmap P1-6 + reviewer attack surface FINAL §4.5 "6/6 跨域负但仅 linear_probe".
> Source: `benchmark/results/newdomain_crossdomain/transfer_ablation.json`
(4 domains: steam/tmdb/wikiart/arxiv; protocols linear_probe & full_finetune; modalities text & text+image).
Structured data: `benchmark/results/newdomain_crossdomain/dual_protocol_report_2026-07-12.json`.

## Headline (honest, dual-protocol)

| Protocol | Modality | Off-diag pairs | Negative Δ | Mean Δ vs scratch | Verdict |
|---|---|---|---|---|---|
| **linear_probe** | text | 12 | **12/12** | **−0.258** | frozen source features **do not transfer** (strongly negative) |
| linear_probe | text+image | 12 | 12/12 | −0.255 | same — adding image does not rescue transfer |
| full_finetune | text | 12 | 3/12 | **+0.001** | **neutral** — model relearns in target (scratch-level) |
| full_finetune | text+image | 12 | 8/12 | −0.0005 | neutral |

## What this means (the paper claim)

- **Linear-probe (frozen extractor): 12/12 off-diagonal transfers are negative** (mean −0.258 acc, worst −0.734). The CLIP-derived graph features learned in one domain carry **no usable signal** for another domain's labels — transferring them *hurts*. This is the strong, defensible headline: **the four MAGs are domain-specific; there is no free cross-domain feature transfer.**
- **Full-finetune: neutral** (mean ≈ 0, range [−0.005, +0.004]). When the target-domain GCN is fully retrained, source initialization gives ≈ scratch accuracy — the model simply **relearns** in the target domain. This is expected and not a contradiction: it says the *architecture* is reusable but the *learned features* are not.

## Why both protocols must be reported

A reviewer will catch a single-protocol claim: "12/12 negative" is only true for the frozen-extractor (linear-probe) setting. Under full fine-tuning the gap collapses to zero. Reporting only linear-probe would overclaim "transfer is harmful"; reporting only full-finetune would underclaim ("nothing transfers"). The honest statement is the **contrast**:

> Frozen source features fail to transfer (linear-probe 12/12 negative, −25.8 pp mean); full fine-tuning recovers to scratch-level, confirming the features are domain-bound but the architecture/training recipe is reusable.

This is consistent with the frozen-encoder benchmark design (FINAL §C2, REPRODUCIBILITY §6): encoders are frozen CLIP; cross-domain transfer at the *feature* layer is not expected.

## Cross-domain NC vs cross-subdomain (arxiv) — scope note

- This dual-protocol result is **cross-domain** (steam↔tmdb↔wikiart↔arxiv). It supports C4's "cross-domain negative result".
- The arxiv-internal `cd_*.json` files are **cross-subdomain** within arxiv (cs_lo↔cs_si↔math_lo↔physics_socph) — a different experiment (same encoder, related tasks). Do not conflate the two in the paper.

## Honest constraints

- n = 12 directed off-diagonal pairs per (protocol, modality) for 4 domains.
- Transfer gain = transfer_acc − scratch_acc (both 3-run mean). Linear-probe negatives are large and consistent; full-finetune deltas are within run-to-run noise (std ≈ a few ×10⁻³), so "neutral" is the honest read, not "marginally positive/negative".
- Model: 2-layer GCN, 768-d CLIP text (+512-d image concatenated for text+image), same hyperparameters across protocols for comparability.
