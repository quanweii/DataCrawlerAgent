# Anonymous review resources

This release accompanies **MAG-Forge: Multi-Agent Construction of Auditable
Multimodal Graph Benchmarks**. Install the artifacts before saved-result checks:

```bash
python scripts/install_artifacts.py
python scripts/verify_release.py
```

## Evidence map

| Paper claim | Released files |
|---|---|
| Four mature MAG datasets | `data/DATASET_RELEASE.json`, `data/<domain>_datasets/`, `benchmark/processed/<domain>/` |
| Four graph-access protocols | `benchmark/results/protocol_views/`, `data/protocol_metadata/` |
| 2,520 primary seeded fits | `review_output/primary_run_registry.csv` and `.json` |
| Active repaired LP results | `review_output/lp_active_results.json`, `lp_active_selection.json`, `benchmark/results/positive_evidence_20261022/lp_repair/` |
| Schema cases | `review_output/schema_fixture_tests.json`; current regression tests in `tests/test_release_contracts.py` |
| WikiArt temporal analysis | `review_output/wikiart_diagnosis.json`, per-seed saved checkpoints/predictions in the evidence archive |
| Twelve guarded recovery episodes | `review_output/guarded_production_runs/` |
| Frozen task matrices and source receipts | `benchmark/results/`, `review_output/RELEASE_PROVENANCE.json` |
| Figure 1 and Figure 3 numerical inputs | `review_output/figure1_paper_data.json`, `figure3_paper_data.json` |
| Table 4 row-level source audit | `review_output/table4_paper_data.json` |

Small JSON/CSV summaries are directly browsable in the repository. Large saved
tensors, checkpoints and traces are restored from the evidence archive. The
2,520-fit registry records the original training campaign; 90 repaired primary
Trans LP fits replace the corresponding historical rows. It is not a claim that
every NC checkpoint was saved; the retained per-seed metrics, matrix hashes and
the available checkpoint/prediction witnesses are separately identified.

Historical matrices and diagnostic snapshots are retained. The final manuscript
Table 4 uses active repaired LP selections for the four domains. Figure 1's
controlled-fault means and the appendix museum table retain their stated
historical evaluation snapshots; later repaired LP/content records are also
available with version labels and must not silently replace those values.
The skill example contains synthetic software fixtures and is not experimental
evidence or one of the four research datasets.
