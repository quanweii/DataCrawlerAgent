import json

import torch

from agents.graph_verifier import GraphVerifierAgent
from agents.messages import Blackboard, MessageBus
from agents.orchestrator_agent import DomainConfig, OrchestratorAgent, PlanStep, register_domain_config
from agents.schema_discovery import validate_schema
from benchmark.data.protocols import PROTOCOLS, generate_protocol_views


def test_generic_protocol_metadata_and_all_views(tmp_path):
    rows = [{"id": i, "date": f"20{20 + i // 10:02d}-01-01",
             "owner": f"g{i // 2}"} for i in range(30)]
    path = tmp_path / "rows.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    y = torch.tensor([i % 3 for i in range(30)])
    edges = torch.tensor([
        [i for i in range(29)] + [i + 1 for i in range(29)],
        [i + 1 for i in range(29)] + [i for i in range(29)],
    ])
    suite = generate_protocol_views(
        "toy", [str(i) for i in range(30)], y, edges,
        metadata_config={"data_file": str(path), "id_field": "id",
                         "time_field": "date", "group_field": "owner"})
    assert set(suite["views"]) == set(PROTOCOLS)
    assert suite["manifest"]["checks"]["group_overlap"] == 0
    assert suite["manifest"]["checks"]["inductive_views_have_no_cross_split_edges"]
    support = suite["manifest"]["temporal_label_support"]
    assert support["coverable_test_nodes"] <= support["test_nodes"]
    assert abs(support["coverable_test_mass"] + support["unseen_test_mass"] - 1) < 1e-12


def test_protocol_subset_does_not_require_metadata():
    y = torch.tensor([i % 2 for i in range(20)])
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    suite = generate_protocol_views(
        "unknown", [str(i) for i in range(20)], y, edge_index,
        protocols=("transductive", "split_disjoint"))
    assert list(suite["views"]) == ["transductive", "split_disjoint"]


def test_schema_validation_records_target_lineage():
    items = [{"id": i, "label": f"c{i % 2}", "owner": f"g{i % 4}"}
             for i in range(20)]
    audit = validate_schema(items, {
        "id_field": "id", "label_field": "label", "min_class": 2,
        "relations": [
            {"name": "same_label", "group_by": "label", "cap": 10},
            {"name": "same_owner", "group_by": "owner", "cap": 10},
        ],
    }, min_class_size=2)
    assert audit["valid"]
    assert audit["excluded_from_benchmark"] == ["same_label"]
    assert not audit["relation_lineage"]["same_owner"]["target_derived"]


def test_llm_plan_cannot_omit_required_quality_gates():
    register_domain_config("toy_plan", DomainConfig(
        domain="toy_plan",
        agents_desc={"toy_collector": "c", "schema_discovery": "s",
                     "toy_graph_builder": "b", "dataset_formatter": "f",
                     "graph_verifier": "v"},
        default_plan_factory="_default_plan_co_attribute",
        critical_agents={"toy_collector", "schema_discovery"},
        task_builders={}, namespace_map={}, graph_mode="homogeneous"))
    agents = {name: object() for name in (
        "toy_collector", "schema_discovery", "toy_graph_builder",
        "dataset_formatter", "graph_verifier")}
    orchestrator = OrchestratorAgent(client=None, agents=agents)
    plan = orchestrator._ensure_required_stages([
        PlanStep("toy_collector", "collect"),
    ], {"domain": "toy_plan"})
    names = [step.agent_name for step in plan]
    assert names == ["toy_collector", "schema_discovery", "toy_graph_builder",
                     "dataset_formatter", "graph_verifier"]
    builder = next(step for step in plan if step.agent_name == "toy_graph_builder")
    assert "schema_discovery" in builder.depends_on


def test_verifier_excludes_target_derived_relation_only_from_view():
    blackboard, bus = Blackboard(), MessageBus()
    nodes = [{"id": str(i), "attributes": {"label_index": i % 2,
                                             "label": f"c{i % 2}"}}
             for i in range(20)]
    edges = []
    for i in range(0, 18, 2):
        edges.append({"source": str(i), "target": str(i + 1),
                      "attributes": {"relation": "same_owner"}})
        edges.append({"source": str(i), "target": str(i + 2),
                      "attributes": {"relation": "same_label"}})
    blackboard.write("toy_graph", "nodes", nodes)
    blackboard.write("toy_graph", "edges", edges)
    agent = GraphVerifierAgent(None, "audit", agent_name="verifier",
                               blackboard=blackboard, message_bus=bus)
    agent._current_task = {
        "domain": "toy", "protocol": {"protocols": ["transductive", "split_disjoint"]},
        "integrity": {"relation_lineage": {
            "same_label": {"source_fields": ["label"], "target_derived": True},
            "same_owner": {"source_fields": ["owner"], "target_derived": False},
        }},
    }
    agent.action_load_graph({})
    agent.action_check_duplicates({})
    agent.action_check_leakage({})
    agent.action_check_homophily({})
    agent.action_check_coverage({})
    raw_edges = len(agent._accumulated_results["edges"])
    agent.action_report_issues({})
    assert len(agent._accumulated_results["edges"]) == raw_edges
    assert agent._accumulated_results["exclude_candidates"] == ["same_label"]
    assert agent._accumulated_results["protocol_manifest"][
        "provenance_excluded_relations"] == ["same_label"]
