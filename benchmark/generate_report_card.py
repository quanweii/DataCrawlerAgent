#!/usr/bin/env python3
"""Generate current-only integrity and benchmark utility cards.

Correctness is certified by source/protocol invariants. Predictive performance
is reported separately as evidence of utility and protocol sensitivity; a high
score is never treated as proof that the construction pipeline is correct.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark.data.protocols import PROTOCOLS, protocol_excluded_relations

RESULTS = ROOT / "benchmark" / "results"
DOMAINS = ("arxiv", "steam", "tmdb", "wikiart")
GNNS = ("gcn", "gat", "graphsage", "gin", "gcnii")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def graph_stats(domain: str):
    path = ROOT / "data" / f"{domain}_datasets" / f"{domain}_multimodal_graph_agent.json"
    graph = load(path) or {}
    stats = graph.get("metadata", {}).get("statistics", {})
    labels = stats.get("label_distribution", {})
    if not labels:
        for node in graph.get("nodes", []):
            label = str(node.get("attributes", {}).get("label", "?"))
            labels[label] = labels.get(label, 0) + 1
    counts = list(labels.values())
    n = sum(counts)
    entropy = -sum((c / n) * math.log(c / n) for c in counts if c and n)
    return {
        "nodes": len(graph.get("nodes", [])) or stats.get("node_count"),
        "edges": len(graph.get("edges", [])) or stats.get("edge_count"),
        "classes": len(counts),
        "max_class_fraction": max(counts) / n if counts and n else None,
        "normalized_entropy": (entropy / math.log(len(counts))
                               if len(counts) > 1 else None),
    }


def correctness(domain: str, audit: dict, manifest: dict):
    duplicate = audit.get("A_duplicate_nodes", {})
    edges = audit.get("C_multiedge_folding", {})
    endpoints = audit.get("edge_endpoint_integrity", {})
    checks = manifest.get("checks", {})
    expected = set(protocol_excluded_relations(domain))
    actual = set(manifest.get("provenance_excluded_relations", []))
    available = set(manifest.get("available_protocols", manifest.get("views", {})))
    result = {
        "unique_node_ids": duplicate.get("n_dup_id_keys") == 0,
        "unique_declared_entities": duplicate.get("n_dup_entity_keys") == 0,
        "no_self_loops": edges.get("n_self_loops") == 0,
        "no_exact_relation_duplicates": edges.get("n_dup_exact_srctgtrel") == 0,
        "all_edge_endpoints_exist": endpoints.get("n_edges_with_missing_endpoint_node") == 0,
        "all_four_protocols_materialized": available == set(PROTOCOLS),
        "group_disjoint": checks.get("group_overlap") == 0,
        "inductive_views_have_no_cross_split_edges":
            checks.get("inductive_views_have_no_cross_split_edges") is True,
        "raw_graph_declared_immutable": manifest.get("raw_graph_immutable") is True,
        "sanitized_base_declared": manifest.get("sanitized_base_graph") is True,
        "target_derived_exclusions_exact": actual == expected,
        "representation_policy_recorded": bool(manifest.get("representation_policy")),
    }
    return {
        "checks": result,
        "pass": all(result.values()),
        "failed_checks": [name for name, passed in result.items() if not passed],
        "expected_target_derived_exclusions": sorted(expected),
        "actual_target_derived_exclusions": sorted(actual),
    }


def validation_accuracy(metrics):
    values = metrics.get("val_accs")
    if values:
        return sum(values) / len(values)
    if "val_acc" in metrics:
        return float(metrics["val_acc"])
    raise ValueError("Validation scores are required for a current report card; do not select from test scores.")


def protocol_profile(matrix: dict):
    richest = matrix["modalities"][-1]
    rows = {}
    for protocol in matrix.get("protocols", PROTOCOLS):
        cell = matrix["results"][f"{protocol}/{richest}"]
        models = cell["models"]
        best_name, best = max(models.items(), key=lambda item: validation_accuracy(item[1]))
        best_gnn_name, best_gnn = max(
            ((name, models[name]) for name in GNNS), key=lambda item: validation_accuracy(item[1]))
        mlp = models["mlp"]
        rows[protocol] = {
            "modality": richest,
            "best_model": best_name,
            "selection": "mean validation accuracy; test reporting only",
            "selection_val_acc": validation_accuracy(best),
            "best_accuracy": best["acc"],
            "best_macro_f1": best["f1"],
            "best_gnn": best_gnn_name,
            "best_gnn_accuracy": best_gnn["acc"],
            "mlp_accuracy": mlp["acc"],
            "gnn_minus_mlp": best_gnn["acc"] - mlp["acc"],
            "directed_edges": cell["directed_edges"],
            "split_sizes": cell["split_sizes"],
        }
    scores = [row["best_accuracy"] for row in rows.values()]
    return {
        "richest_modality": richest,
        "views": rows,
        "best_accuracy_range": [min(scores), max(scores)],
        "protocol_sensitivity": max(scores) - min(scores),
    }


def modality_profile(domain: str, modality_rows: list):
    rows = [row for row in modality_rows if row["domain"] == domain]
    deltas = [row["full_minus_best_single_acc"] for row in rows]
    winners = {}
    for row in rows:
        name = row["best_single_modality"]
        winners[name] = winners.get(name, 0) + 1
    return {
        "protocols_evaluated": len(rows),
        "mean_full_minus_best_single_accuracy": sum(deltas) / len(deltas),
        "positive_protocols": sum(delta > 0 for delta in deltas),
        "nonnegative_protocols": sum(delta >= 0 for delta in deltas),
        "best_single_modality_counts": winners,
        "interpretation": "fusion utility is empirical and need not dominate the best single modality",
    }


def topology_profile(domain: str, topology: dict):
    cell = topology[domain]
    rewires = cell["degree_preserving_rewire_null"]["swap_runs"]
    return {
        "interpretation": cell["interpretation"],
        "observed_accuracy_all_test": cell["observed"]["accuracy_all_test"],
        "majority_accuracy": cell["observed"]["majority_accuracy"],
        "label_permutation_mean": cell["training_label_permutation_null"]["mean"],
        "label_permutation_runs": cell["training_label_permutation_null"]["runs"],
        "degree_rewire_mean": cell["degree_preserving_rewire_null"]["mean"],
        "degree_rewire_runs": cell["degree_preserving_rewire_null"]["runs"],
        "all_rewires_degree_verified": all(
            run.get("degree_sequence_verified") is True for run in rewires),
    }


def build_card(domain: str, audit_all: dict, topology: dict, modality_rows: list,
               six_task: dict):
    matrix = load(RESULTS / "newdomain_nc" / "protocol_matrix" /
                  f"nc_{domain}_protocol_matrix.json") or {}
    manifest = matrix.get("protocol_manifest") or load(
        RESULTS / "protocol_views" / f"{domain}_protocol_manifest.json") or {}
    correct = correctness(domain, audit_all.get(domain, {}), manifest)
    protocol = protocol_profile(matrix)
    modality = modality_profile(domain, modality_rows)
    top = topology_profile(domain, topology)
    evidence_complete = (
        len(matrix.get("results", {})) == len(matrix.get("modalities", [])) * 4
        and all(len(cell.get("models", {})) == 6 for cell in matrix.get("results", {}).values())
        and all(len(model.get("accs", [])) == 5
                for cell in matrix.get("results", {}).values()
                for model in cell.get("models", {}).values())
        and top["label_permutation_runs"] == 100
        and top["degree_rewire_runs"] == 10
        and top["all_rewires_degree_verified"]
        and six_task.get("audit", {}).get("status") == "PASS"
        and domain in six_task.get("domains", {})
    )
    status = "PASS" if correct["pass"] and evidence_complete else "FAIL"
    return {
        "domain": domain,
        "identity": graph_stats(domain),
        "integrity": correct,
        "evidence_complete": evidence_complete,
        "protocol_benchmark": protocol,
        "temporal_label_support": manifest.get("temporal_label_support"),
        "modality_benchmark": modality,
        "topology_diagnostic": top,
        "six_task_benchmark": six_task.get("domains", {}).get(domain),
        "status": status,
        "claim_boundary": (
            "invariants certify construction/evaluation correctness; benchmark scores support "
            "utility and sensitivity, not correctness by themselves"),
    }


def render(cards: dict):
    lines = [
        "# MAG Benchmark Integrity Cards (2026-09-10)", "",
        "> 当前四协议六任务证据。正确性由 provenance 与切分不变量证明；性能只说明效用和协议敏感性。", "",
        "| Domain | Integrity | NC cells | Six-task | T / SD / Temp / Group | Protocol range | GNN−MLP range | Fusion positive |",
        "|---|---|---:|---|---|---:|---:|---:|",
    ]
    for domain, card in cards.items():
        views = card["protocol_benchmark"]["views"]
        acc = " / ".join(f"{views[p]['best_accuracy']:.4f}" for p in PROTOCOLS)
        gaps = [views[p]["gnn_minus_mlp"] for p in PROTOCOLS]
        modality_count = 7 if domain == "arxiv" else 3
        lines.append(
            f"| {domain} | {card['status']} | {4 * modality_count} | "
            f"{'complete' if card.get('six_task_benchmark') else 'missing'} | {acc} | "
            f"{card['protocol_benchmark']['protocol_sensitivity']:.4f} | "
            f"[{min(gaps):+.4f}, {max(gaps):+.4f}] | "
            f"{card['modality_benchmark']['positive_protocols']}/4 |")
    for domain, card in cards.items():
        lines += ["", f"## {domain}", "",
                  f"- Integrity: **{card['status']}**; failed checks: "
                  f"{card['integrity']['failed_checks'] or 'none'}.",
                  f"- Target-derived exclusions: {card['integrity']['actual_target_derived_exclusions']}.",
                  f"- Topology diagnostic: observed {card['topology_diagnostic']['observed_accuracy_all_test']:.4f}, "
                  f"label-permutation mean {card['topology_diagnostic']['label_permutation_mean']:.4f}, "
                  f"degree-rewire mean {card['topology_diagnostic']['degree_rewire_mean']:.4f}; "
                  "this is diagnostic evidence, not a leakage verdict.",
                  f"- Modality fusion minus best single modality (mean across protocols): "
                  f"{card['modality_benchmark']['mean_full_minus_best_single_accuracy']:+.4f}."]
        support = card.get("temporal_label_support") or {}
        if support:
            lines.append(
                f"- Temporal label support: coverable test mass "
                f"{support['coverable_test_mass']:.4f}; unseen test classes "
                f"{support['unseen_test_classes']}; seen-class prior TVD "
                f"{support['seen_class_prior_tvd']:.4f}.")
    return "\n".join(lines) + "\n"


def main():
    audit = load(RESULTS / "data_quality_audit.json") or {}
    topology = load(RESULTS / "topology_nulls.json") or {}
    modality = load(RESULTS / "newdomain_nc" / "protocol_matrix" /
                    "protocol_modality_summary.json") or {"rows": []}
    six_task = load(RESULTS / "four_protocol_tasks" / "six_task_summary.json") or {}
    cards = {domain: build_card(domain, audit, topology, modality["rows"], six_task)
             for domain in DOMAINS}
    (RESULTS / "report_cards.json").write_text(
        json.dumps(cards, indent=2, ensure_ascii=False), encoding="utf-8")
    (RESULTS / "report_cards.md").write_text(render(cards), encoding="utf-8")
    print(json.dumps({domain: card["status"] for domain, card in cards.items()}))


if __name__ == "__main__":
    main()
