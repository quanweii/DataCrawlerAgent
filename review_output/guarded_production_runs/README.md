# Guarded production runs (12 episodes, 2026-10-15)

Full-role guarded controller (production orchestrator with the guarded patch in
production_patch.json: before/after SHA-256 of the two patched modules;
production_regression.json PASS) over the frozen design (DESIGN_FROZEN.json,
design.json) and frozen source snapshot (source_sha256 inside DESIGN_FROZEN.json).
Fault conditions: clean / write_retry / partial_edges / combined, three
repetitions each (episodes.csv, summary.json: 12/12 primary successes, 1176
checks, 36 reloaded bundles).

Per episode: task.json (confirmed schema request), action_trace.json (per-action
events), result.json/autonomous_result.json, and release/*.receipt.json +
protocol_manifest.json (hash receipts for the three protocol views). Binary
classifier/view checkpoints (.pt) are excluded for size; every receipt hash can
be recomputed from the manifest fields.

Replay of the paper's partial-edge trajectory (episodes/partial_edges_r0_multi_guarded/action_trace.json):
  1. injected fault: first builder save keeps a fixed 5% edge slice (175 edges);
  2. graph_verifier check_coverage -> connected 44 / isolated 856;
  3. report_issues -> "856/900 nodes isolated (95.1%) - collector or edge builder
     may be incomplete", reprocess_target=cleveland_graph_builder (priority 2);
  4. reset builder/formatter/verifier; rebuild with the same confirmed schema and
     the same frozen records (hash seed 0) -> 3,503 edges;
  5. re-audit: connected 694 / isolated 206 / issues 0 (remaining isolation is
     disclosed coverage: unknown creators), release receipts reissued.

Example reload: python -c "import json;d=json.load(open('episodes/partial_edges_r0_multi_guarded/action_trace.json'));[print(e['index'],e['stage'],e['action'],e.get('result',{}).get('issues','')) for e in d if e['stage']=='graph_verifier']"
All JSON is UTF-8; paths were rewritten to /PKG_ROOT placeholders.
