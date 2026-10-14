#!/usr/bin/env python3
"""Audit the same production YAML contract across mature and held-out domains."""
from __future__ import annotations

import json
from pathlib import Path

from agents.orchestrator_agent import OrchestratorAgent, register_domain_config
from agents.schema_discovery import validate_schema
from domains import load_domain_yaml, yaml_to_domain_config, yaml_to_task_config

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("steam", "tmdb", "wikiart", "spotify")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit_domain(domain: str):
    yaml_path = ROOT / "domains" / domain / "domain.yaml"
    yc = load_domain_yaml(str(yaml_path))
    task = yaml_to_task_config(yc)
    raw_path = (ROOT / task["data_file"]).resolve()
    items = load(raw_path)
    schema = validate_schema(items, task["schema_hint"], task["min_class_size"])

    cfg = yaml_to_domain_config(yc)
    register_domain_config(domain, cfg)
    agents = {name: object() for name in cfg.agents_desc}
    planner = OrchestratorAgent(client=None, agents=agents)
    plan = planner._default_plan(task)
    plan_names = [step.agent_name for step in plan]
    collector = yc.agents.collector_agent_name or f"{domain}_collector"
    builder = yc.agents.builder_agent_name or f"{domain}_graph_builder"
    expected_plan = [collector, "schema_discovery", builder,
                     "dataset_formatter", "graph_verifier"]

    graph_path = ROOT / "data" / f"{domain}_datasets" / f"{domain}_multimodal_graph_agent.json"
    graph = load(graph_path)
    node_ids = [str(node["id"]) for node in graph.get("nodes", [])]
    node_set = set(node_ids)
    declared_relations = {relation.name for relation in yc.relations}
    actual_relations = {edge.get("attributes", {}).get("relation")
                        for edge in graph.get("edges", [])}
    missing_endpoints = sum(
        str(edge.get("source")) not in node_set or str(edge.get("target")) not in node_set
        for edge in graph.get("edges", []))
    checks = {
        "raw_file_exists": raw_path.exists(),
        "schema_deterministically_valid": schema["valid"],
        "mandatory_five_agent_plan": plan_names == expected_plan,
        "unique_graph_node_ids": len(node_ids) == len(node_set),
        "graph_has_nodes_and_edges": bool(node_ids) and bool(graph.get("edges")),
        "all_edge_endpoints_exist": missing_endpoints == 0,
        "edge_relations_declared_in_yaml": actual_relations <= declared_relations,
        "target_lineage_matches_config": set(schema["excluded_from_benchmark"]) == {
            relation.name for relation in yc.relations
            if relation.group_by.split("[", 1)[0].split(".", 1)[0] ==
               yc.data.label_field.split("[", 1)[0].split(".", 1)[0]},
        "protocol_provenance_complete": all(
            protocol not in task["protocol"]["protocols"] or
            bool(task["protocol"]["time_field"] if protocol == "temporal"
                 else task["protocol"]["group_field"])
            for protocol in ("temporal", "group_inductive")),
    }
    return {
        "domain": domain,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "raw_items": len(items),
        "graph_nodes": len(node_ids),
        "graph_edges": len(graph.get("edges", [])),
        "declared_relations": sorted(declared_relations),
        "actual_relations": sorted(actual_relations),
        "target_derived_relations": schema["excluded_from_benchmark"],
        "applicable_protocols": task["protocol"]["protocols"],
        "plan": plan_names,
    }


def main():
    rows = {domain: audit_domain(domain) for domain in DOMAINS}
    result = {
        "status": "PASS" if all(row["status"] == "PASS" for row in rows.values()) else "FAIL",
        "scope": "production YAML -> schema audit -> mandatory plan -> existing graph contract",
        "domains": rows,
        "claim_boundary": (
            "Spotify demonstrates a held-out YAML domain contract. Its source has no trustworthy "
            "timestamp, so temporal evaluation is inapplicable rather than fabricated."),
    }
    out = ROOT / "benchmark" / "results" / "reusability_audit.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Production Reusability Audit", "", f"Status: **{result['status']}**", "",
             "| Domain | Raw | Graph nodes | Graph edges | Protocols | Five-agent plan |",
             "|---|---:|---:|---:|---|---|"]
    for domain, row in rows.items():
        lines.append(f"| {domain} | {row['raw_items']} | {row['graph_nodes']} | "
                     f"{row['graph_edges']} | {', '.join(row['applicable_protocols'])} | "
                     f"{row['status']} |")
    lines += ["", result["claim_boundary"]]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"],
                      "domains": {d: r["status"] for d, r in rows.items()}}))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
