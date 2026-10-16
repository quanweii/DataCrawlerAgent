#!/usr/bin/env python3
"""Fail-fast validator for the final four-domain evidence bundle."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark.data.protocols import PROTOCOLS, protocol_excluded_relations

R = ROOT / "benchmark" / "results"
MATRIX_DIR = R / "newdomain_nc" / "protocol_matrix"
DOMAINS = ("arxiv", "steam", "tmdb", "wikiart")
EXPECTED_MODALITIES = {"arxiv": 7, "steam": 3, "tmdb": 3, "wikiart": 3}
MODELS = {"mlp", "gcn", "gat", "graphsage", "gin", "gcnii"}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    checks, errors = {}, []

    def check(name, condition, detail=""):
        checks[name] = bool(condition)
        if not condition:
            errors.append(f"{name}: {detail}")

    total_cells = total_model_cells = total_seed_runs = 0
    matrices = {}
    for domain in DOMAINS:
        matrix = load(MATRIX_DIR / f"nc_{domain}_protocol_matrix.json")
        matrices[domain] = matrix
        modalities = matrix.get("modalities", [])
        cells = matrix.get("results", {})
        check(f"{domain}.protocols", set(matrix.get("protocols", [])) == set(PROTOCOLS))
        check(f"{domain}.modality_count", len(modalities) == EXPECTED_MODALITIES[domain])
        check(f"{domain}.cell_count", len(cells) == 4 * len(modalities))
        total_cells += len(cells)
        for key, cell in cells.items():
            check(f"{domain}.{key}.models", set(cell.get("models", {})) == MODELS)
            total_model_cells += len(cell.get("models", {}))
            for model_name, result in cell.get("models", {}).items():
                accs, f1s = result.get("accs", []), result.get("f1s", [])
                check(f"{domain}.{key}.{model_name}.five_seeds", len(accs) == len(f1s) == 5)
                check(f"{domain}.{key}.{model_name}.finite",
                      all(math.isfinite(x) and 0 <= x <= 1 for x in accs + f1s))
                if accs:
                    check(f"{domain}.{key}.{model_name}.acc_mean",
                          abs(sum(accs) / len(accs) - result["acc"]) < 1e-10)
                if f1s:
                    check(f"{domain}.{key}.{model_name}.f1_mean",
                          abs(sum(f1s) / len(f1s) - result["f1"]) < 1e-10)
                total_seed_runs += len(accs)

        manifest = matrix.get("protocol_manifest", {})
        pchecks = manifest.get("checks", {})
        check(f"{domain}.raw_immutable", manifest.get("raw_graph_immutable") is True)
        check(f"{domain}.all_views", set(manifest.get("views", {})) == set(PROTOCOLS))
        check(f"{domain}.group_overlap", pchecks.get("group_overlap") == 0)
        check(f"{domain}.cross_edges",
              pchecks.get("inductive_views_have_no_cross_split_edges") is True)
        check(f"{domain}.provenance_exclusions",
              set(manifest.get("provenance_excluded_relations", [])) ==
              set(protocol_excluded_relations(domain)))
        support = manifest.get("temporal_label_support") or {}
        temporal_stats = manifest.get("views", {}).get("temporal", {})
        check(f"{domain}.temporal_label_support_present", bool(support))
        check(f"{domain}.temporal_support_node_total",
              support.get("test_nodes") == temporal_stats.get("nodes", [None, None, None])[2])
        check(f"{domain}.temporal_support_mass",
              abs(support.get("coverable_test_mass", -1) +
                  support.get("unseen_test_mass", -1) - 1.0) < 1e-12)
        check(f"{domain}.temporal_train_missing_classes",
              support.get("train_missing_global_classes") ==
              temporal_stats.get("train_missing_classes"))
        check(f"{domain}.temporal_prior_tvd_range",
              0 <= support.get("seen_class_prior_tvd", -1) <= 1)
        check(f"{domain}.temporal_prior_js_range",
              0 <= support.get("seen_class_prior_js_nats", -1) <= math.log(2) + 1e-12)

    check("bundle.total_dataset_protocol_modality_cells", total_cells == 64, total_cells)
    check("bundle.total_model_cells", total_model_cells == 384, total_model_cells)
    check("bundle.total_seeded_runs", total_seed_runs == 1920, total_seed_runs)

    aggregate = load(MATRIX_DIR / "protocol_matrix_all.json")
    check("aggregate.headline_rows", len(aggregate.get("rows", [])) == 16)
    for row in aggregate.get("rows", []):
        matrix = matrices[row["domain"]]
        richest = matrix["modalities"][-1]
        cell = matrix["results"][f"{row['protocol']}/{richest}"]
        best_name, best = max(cell["models"].items(), key=lambda item: item[1]["acc"])
        check(f"aggregate.{row['domain']}.{row['protocol']}.best_model",
              best_name == row["best_model"])
        check(f"aggregate.{row['domain']}.{row['protocol']}.best_acc",
              abs(best["acc"] - row["best_acc"]) < 1e-12)

    modality = load(MATRIX_DIR / "protocol_modality_summary.json")
    check("aggregate.modality_rows", len(modality.get("rows", [])) == 16)
    for row in modality.get("rows", []):
        matrix = matrices[row["domain"]]
        cell_prefix = row["protocol"] + "/"
        single = [m for m in matrix["modalities"] if "+" not in m]
        richest = matrix["modalities"][-1]
        best_single = max(
            ((m, max(matrix["results"][cell_prefix + m]["models"].values(),
                     key=lambda value: value["acc"])) for m in single),
            key=lambda item: item[1]["acc"])
        full = max(matrix["results"][cell_prefix + richest]["models"].values(),
                   key=lambda value: value["acc"])
        check(f"modality.{row['domain']}.{row['protocol']}.single",
              row["best_single_modality"] == best_single[0])
        check(f"modality.{row['domain']}.{row['protocol']}.delta",
              abs(row["full_minus_best_single_acc"] -
                  (full["acc"] - best_single[1]["acc"])) < 1e-12)

    topology = load(R / "topology_nulls.json")
    for domain in DOMAINS:
        cell = topology[domain]
        check(f"{domain}.permutation_runs",
              cell["training_label_permutation_null"]["runs"] == 100)
        rewires = cell["degree_preserving_rewire_null"]
        check(f"{domain}.rewire_runs", rewires["runs"] == 10)
        check(f"{domain}.rewire_degree_verified", all(
            run.get("degree_sequence_verified") is True for run in rewires["swap_runs"]))

    cards = load(R / "report_cards.json")
    check("report_cards.domains", set(cards) == set(DOMAINS))
    check("report_cards.all_pass", all(card.get("status") == "PASS" for card in cards.values()))

    summary = {
        "status": "PASS" if not errors else "FAIL",
        "counts": {"dataset_protocol_modality_cells": total_cells,
                   "model_cells": total_model_cells, "seeded_runs": total_seed_runs,
                   "label_permutations": 400, "degree_preserving_rewires": 40},
        "checks_run": len(checks), "checks_passed": sum(checks.values()),
        "errors": errors, "checks": checks,
    }
    (R / "final_evidence_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# Final Evidence Audit", "", f"Status: **{summary['status']}**", "",
          f"- Checks: {summary['checks_passed']}/{summary['checks_run']}",
          f"- Dataset × protocol × modality cells: {total_cells}",
          f"- Model cells: {total_model_cells}",
          f"- Seeded training runs: {total_seed_runs}",
          "- Topology nulls: 400 label permutations + 40 degree-preserving rewires",
          "", "Errors: " + ("none" if not errors else "; ".join(errors))]
    (R / "final_evidence_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("status", "counts", "checks_run", "checks_passed")}))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
