# Artifact release-boundary study (2026-09-18)

## Scope fixed before full campaign

This is a controlled software fault campaign, not a human-participant experiment,
not an estimate of naturally occurring defects, and not a full agent ablation.
The existing `save_protocol_suite` is a serializer, not a fail-closed publisher.
`benchmark/artifact_release.py` is a **new, opt-in extension**; no historical
result is retroactively attributed to this publisher. Production datasets and
earlier benchmark outputs remain immutable. Candidates are derived real-graph
subsets with frozen features, not new complete source acquisitions.

## Sampling and cases

- Domains: arXiv, Steam, TMDB, WikiArt.
- One fixed stratified subset per domain, seed 20260918, approximately 2,048
  nodes (per-class quota; no replacement; rare classes may reduce count).
- Perturbation/training seeds 0, 1, 2; one fault per candidate.
- Ten faults: declared target-derived relation; explicitly serialized target
  feature; duplicated entity across partitions; missing time coerced to zero;
  primary-group overlap; prohibited self-loop; dangling endpoint; label-sorted
  splitting falsely declared stratified; test-label checkpoint selection;
  mixed feature row/version alignment.
- Three nonfault controls: clean, harmless relation renaming, edge reordering.
- Same candidate per policy; no ground-truth fault identifier is passed to the
  release auditor. Post-readback evaluation uses injection-specific witnesses,
  not the auditor's verdict.

## Policy arms

1. Training-label-homophily heuristic: retain the relation with the highest
   training-only homophily (lexicographic ties), then export without auditing.
   This is an operational heuristic, not a measured human decision process.
2. Unchecked export: write the candidate without semantic checks.
3. New audited export: reject on artifact/contract violation, otherwise write.
   This is rejection, **not repair**. Report release yield and clean acceptance
   alongside escaped-fault counts, so rejecting everything cannot look useful.

Unchanged legacy serialization corresponds to unchecked export at this boundary;
earlier builder/verifier checks are not being ablated end to end.

## Fault meaning and exclusions

The stratification fault is a mismatch to an explicitly stratified contract,
not a claim that all random/non-stratified splits are invalid. Self-loops are
invalid under this bundle's simple-graph contract, not under all GNNs.
The target-feature intervention inserts a one-hot target-field feature into
a reserved block of the input. It is an explicit leakage stress test, **not**
an estimate of leaving genre words in a CLIP text encoder. Temporal/group
faults change cohorts; their score changes are not isolated leakage effects.

## Outcomes

- Primary: independently witnessed residual fault in a reloaded, actually
  published `.pt` file; denominator 12 candidates per fault and policy.
- Also report rejected faulty candidates, published clean controls, and actual
  artifact SHA-256. Rejected outputs receive no invented downstream score.
- Secondary: paired unchecked-faulty minus clean NC accuracy and LP AUC, in
  percentage points. Negative, zero, failed, and inapplicable results retained.
- NC: existing 2-layer GraphSAGE, hidden 64, dropout .2, Adam lr .005,
  weight decay .0005, 60 epochs, checkpoint selection by validation accuracy.
- LP: same backbone, embedding 32, existing positive/negative splitting,
  training-positive message graph, checkpoint selection by validation AP.
- Test-selection fault replaces the selection cohort with test data (NC) or
  test AP (LP). Curves are retained to inspect selection effects.
- Clean fits are cached only when seed, protocol, and chronology are identical.
- Fixed subset/perturbation seeds are not independent domains or human users.

## Reporting commitments

Do not prespecify 8/10, 5/10, or 0/10. Report observed results and false
rejections. Do not translate algorithmic export comparisons into human speedup,
human error rates, or evidence for a benefit of adding agents. Any zero-escape
result is conditional on these explicit fault families and trusted contracts.
This extends schema-level tests to serialized bundles; it does not establish
robustness to arbitrary hidden proxies, maliciously falsified manifests, or
unknown defect types.
