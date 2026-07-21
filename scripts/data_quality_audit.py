#!/usr/bin/env python3
"""
Data Quality Audit for the NeurIPS D&B multimodal-attribute-graph benchmark.

Audits three reviewer-facing properties across the 5 released graphs
(arxiv, steam, tmdb, wikiart, spotify):

  (A) DUPLICATE NODES      -- same real-world entity appearing under multiple ids,
                               plus exact-id duplicates and same-name/different-id cases.
  (B) LP TRAIN/TEST NODE   -- transductive overlap of node sets between train_pos and
       OVERLAP                test_pos, plus fraction of test edges whose endpoints are
                               wholly unseen in training (inductive-leakage check) and
                               exact edge overlap (edge leakage).
  (C) MULTI-EDGE FOLDING   -- duplicate undirected {u,v} pairs (multi-relation folding)
                               vs. exact (src,tgt,relation) duplicates (true defects),
                               and self-loops.

Reproducible: plain python stdlib only (json, csv, collections).
Outputs:
    benchmark/results/data_quality_audit.json
    docs/data_quality_audit_2026-07-12.md  (written by a separate step)

Usage:
    python3 scripts/data_quality_audit.py
"""
import json
import csv
import os
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT_JSON = os.path.join(ROOT, "benchmark", "results", "data_quality_audit.json")

DOMAINS = ["arxiv", "steam", "tmdb", "wikiart", "spotify"]

# Real-world entity key inside node['attributes'], and the human-readable name key.
# arxiv graph nodes carry only label/label_index -- no title -- so name_key is None.
ENTITY_KEY = {
    "arxiv": ("arxiv_id", None),
    "steam": ("appid", "name"),
    "tmdb": ("movie_id", "title"),
    "wikiart": ("painting_id", "title"),
    "spotify": ("track_id", "name"),
}


def graph_path(dom):
    return os.path.join(DATA, f"{dom}_datasets", f"{dom}_multimodal_graph_agent.json")


def lp_path(dom, fname):
    return os.path.join(DATA, f"{dom}_lp", fname)


# --------------------------------------------------------------------------- #
# (A) Duplicate / repeated nodes
# --------------------------------------------------------------------------- #
def audit_nodes(nodes, dom):
    node_ids = [n["id"] for n in nodes]
    id_counter = Counter(node_ids)
    n_nodes = len(node_ids)
    n_unique_ids = len(id_counter)

    # ids that appear more than once (exact duplicate id)
    dup_id_keys = {k: v for k, v in id_counter.items() if v > 1}
    n_dup_id_keys = len(dup_id_keys)
    # redundant copies = total - unique  (also = sum(v-1) for dups)
    n_dup_extra = n_nodes - n_unique_ids
    dup_rate = (n_dup_extra / n_nodes * 100.0) if n_nodes else 0.0

    # entity-key check: does the canonical entity id (appid/movie_id/...) agree with
    # the node id, and does the same entity appear under multiple node ids?
    ent_key, name_key = ENTITY_KEY[dom]
    ent_to_node_ids = defaultdict(set)
    ent_mismatch = 0
    for n in nodes:
        attr = n.get("attributes", {})
        if ent_key and ent_key in attr:
            ent_val = attr[ent_key]
            ent_to_node_ids[str(ent_val)].add(n["id"])
            if str(ent_val) != str(n["id"]):
                ent_mismatch += 1
    # same entity (appid/...) appearing under >1 node id = real duplicate entity
    dup_entity_keys = {k: list(sorted(v)) for k, v in ent_to_node_ids.items() if len(v) > 1}

    # same human-readable name under different ids (e.g. a game/movie re-listed)
    name_to_ids = defaultdict(set)
    if name_key:
        for n in nodes:
            nm = n.get("attributes", {}).get(name_key)
            if isinstance(nm, str) and nm.strip():
                name_to_ids[nm].add(n["id"])
    name_collisions = {k: list(sorted(v)) for k, v in name_to_ids.items() if len(v) > 1}

    return {
        "n_nodes": n_nodes,
        "n_unique_ids": n_unique_ids,
        "n_dup_id_keys": n_dup_id_keys,
        "n_dup_extra_copies": n_dup_extra,
        "dup_rate_pct": round(dup_rate, 4),
        "dup_id_examples": dict(list(dup_id_keys.items())[:10]),
        "entity_key": ent_key,
        "entity_id_mismatch_count": ent_mismatch,
        "n_dup_entity_keys": len(dup_entity_keys),
        "dup_entity_examples": dict(list(dup_entity_keys.items())[:5]),
        "name_key": name_key,
        "n_name_collisions": len(name_collisions),
        "name_collision_examples": dict(list(name_collisions.items())[:5]),
    }


# --------------------------------------------------------------------------- #
# (B) LP train/test node overlap
# --------------------------------------------------------------------------- #
def load_edge_csv(path):
    """Return list of (src,dst) tuples; tolerant to missing file."""
    edges = []
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        # tolerant: find source/target columns regardless of exact header
        if reader.fieldnames is None:
            return []
        src_col = "source" if "source" in reader.fieldnames else reader.fieldnames[0]
        dst_col = "target" if "target" in reader.fieldnames else reader.fieldnames[1]
        for row in reader:
            s = row.get(src_col, "").strip()
            t = row.get(dst_col, "").strip()
            if s and t:
                edges.append((s, t))
    return edges


def audit_lp_overlap(dom):
    train = load_edge_csv(lp_path(dom, "train_pos.csv"))
    test = load_edge_csv(lp_path(dom, "test_pos.csv"))
    result = {
        "train_pos_exists": train is not None,
        "test_pos_exists": test is not None,
        "n_train_edges": len(train) if train else 0,
        "n_test_edges": len(test) if test else 0,
    }
    if not train or not test:
        result.update({
            "node_overlap_pct": None,
            "test_edges_fully_unseen_pct": None,
            "exact_edge_overlap_count": None,
            "exact_edge_overlap_pct": None,
        })
        return result

    train_nodes = set()
    for s, t in train:
        train_nodes.add(s)
        train_nodes.add(t)
    test_nodes = set()
    for s, t in test:
        test_nodes.add(s)
        test_nodes.add(t)

    overlap = train_nodes & test_nodes
    overlap_pct = (len(overlap) / len(test_nodes) * 100.0) if test_nodes else 0.0

    # fraction of test edges whose BOTH endpoints are unseen in train (inductive)
    unseen = test_nodes - train_nodes
    fully_unseen_edges = sum(1 for s, t in test if s in unseen and t in unseen)
    fully_unseen_pct = (fully_unseen_edges / len(test) * 100.0) if test else 0.0

    # fraction of test edges with at least one unseen endpoint
    partial_unseen_edges = sum(1 for s, t in test if s in unseen or t in unseen)
    partial_unseen_pct = (partial_unseen_edges / len(test) * 100.0) if test else 0.0

    # exact edge leakage: test (directed) edges that also appear in train
    train_set_dir = set(train)
    train_set_undir = set(tuple(sorted(e)) for e in train)
    exact_dir_overlap = sum(1 for e in test if e in train_set_dir)
    exact_undir_overlap = sum(1 for e in test if tuple(sorted(e)) in train_set_undir)

    result.update({
        "n_train_nodes": len(train_nodes),
        "n_test_nodes": len(test_nodes),
        "n_overlap_nodes": len(overlap),
        "node_overlap_pct": round(overlap_pct, 4),
        "test_nodes_unseen_in_train": len(unseen),
        "test_edges_fully_unseen_pct": round(fully_unseen_pct, 4),
        "test_edges_any_unseen_pct": round(partial_unseen_pct, 4),
        "exact_edge_overlap_directed": exact_dir_overlap,
        "exact_edge_overlap_undirected": exact_undir_overlap,
        "exact_edge_overlap_pct": round(exact_undir_overlap / len(test) * 100.0, 4) if test else 0.0,
    })
    return result


# --------------------------------------------------------------------------- #
# (C) Multi-edge folding / self-loops
# --------------------------------------------------------------------------- #
def audit_edges(edges):
    n_edges = len(edges)
    directed_pairs = Counter()
    undirected_pairs = Counter()
    rel_pairs = Counter()       # (src, tgt, relation) -- exact
    rel_pairs_undir = Counter() # (frozenset{src,tgt}, relation)
    self_loops = 0
    relations = Counter()

    for e in edges:
        s = e.get("source")
        t = e.get("target")
        rel = e.get("attributes", {}).get("relation") if isinstance(e.get("attributes"), dict) else None
        relations[rel] += 1
        if s == t:
            self_loops += 1
        directed_pairs[(s, t)] += 1
        undirected_pairs[tuple(sorted((s, t)))] += 1
        rel_pairs[(s, t, rel)] += 1
        rel_pairs_undir[(tuple(sorted((s, t))), rel)] += 1

    n_unique_undirected = len(undirected_pairs)
    n_unique_directed = len(directed_pairs)
    # extra undirected copies = edges beyond the first per {u,v}
    n_dup_undirected = n_edges - n_unique_undirected
    n_dup_directed = n_edges - n_unique_directed
    dup_undirected_rate = (n_dup_undirected / n_edges * 100.0) if n_edges else 0.0

    # TRUE defect: exact (src,tgt,relation) duplicate (same directed edge, same relation)
    n_unique_rel = len(rel_pairs)
    n_dup_exact = n_edges - n_unique_rel
    dup_exact_rate = (n_dup_exact / n_edges * 100.0) if n_edges else 0.0

    # multi-relation folding: same {u,v} pair appears with >1 distinct relation
    pair_relations = defaultdict(set)
    for (pair, rel), c in rel_pairs_undir.items():
        if pair[0] != pair[1]:  # skip self loops
            pair_relations[pair].add(rel)
    multi_relation_pairs = sum(1 for rels in pair_relations.values() if len(rels) > 1)

    return {
        "n_edges": n_edges,
        "relations": dict(relations),
        "n_unique_undirected_edges": n_unique_undirected,
        "n_unique_directed_edges": n_unique_directed,
        "n_dup_undirected_edges": n_dup_undirected,
        "n_dup_directed_edges": n_dup_directed,
        "dup_undirected_rate_pct": round(dup_undirected_rate, 4),
        "n_dup_exact_srctgtrel": n_dup_exact,
        "dup_exact_rate_pct": round(dup_exact_rate, 4),
        "n_self_loops": self_loops,
        "n_multi_relation_pairs": multi_relation_pairs,
    }


# --------------------------------------------------------------------------- #
# edge-endpoint integrity: edges referencing nodes not in the node set
# --------------------------------------------------------------------------- #
def audit_edge_endpoint_integrity(edges, node_id_set):
    missing = set()
    for e in edges:
        for ep in (e.get("source"), e.get("target")):
            if ep not in node_id_set:
                missing.add(ep)
    return {"n_edges_with_missing_endpoint_node": len(missing), "examples": list(missing)[:5]}


# --------------------------------------------------------------------------- #
def main():
    report = {}
    for dom in DOMAINS:
        with open(graph_path(dom)) as f:
            g = json.load(f)
        nodes, edges = g["nodes"], g["edges"]
        node_ids = {n["id"] for n in nodes}

        node_audit = audit_nodes(nodes, dom)
        edge_audit = audit_edges(edges)
        lp_audit = audit_lp_overlap(dom)
        integrity = audit_edge_endpoint_integrity(edges, node_ids)

        report[dom] = {
            "graph_file": os.path.relpath(graph_path(dom), ROOT),
            "A_duplicate_nodes": node_audit,
            "B_lp_train_test_overlap": lp_audit,
            "C_multiedge_folding": edge_audit,
            "edge_endpoint_integrity": integrity,
        }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {OUT_JSON}")

    # console summary
    for dom in DOMAINS:
        r = report[dom]
        na, ea, lp = r["A_duplicate_nodes"], r["C_multiedge_folding"], r["B_lp_train_test_overlap"]
        print(f"\n=== {dom} ===")
        print(f"  nodes={na['n_nodes']} unique={na['n_unique_ids']} dup_id_keys={na['n_dup_id_keys']} "
              f"dup_rate={na['dup_rate_pct']}%  (entity_dup_keys={na['n_dup_entity_keys']}, "
              f"name_collisions={na['n_name_collisions']})")
        print(f"  edges={ea['n_edges']} unique_undir={ea['n_unique_undirected_edges']} "
              f"dup_undir={ea['n_dup_undirected_edges']} ({ea['dup_undirected_rate_pct']}%) "
              f"exact_dup={ea['n_dup_exact_srctgtrel']} ({ea['dup_exact_rate_pct']}%) "
              f"self_loops={ea['n_self_loops']} multi_rel_pairs={ea['n_multi_relation_pairs']}")
        if lp["train_pos_exists"] and lp["test_pos_exists"]:
            print(f"  LP: train_edges={lp['n_train_edges']} test_edges={lp['n_test_edges']} "
                  f"node_overlap={lp['node_overlap_pct']}% fully_unseen={lp['test_edges_fully_unseen_pct']}% "
                  f"exact_edge_overlap={lp['exact_edge_overlap_undirected']} ({lp['exact_edge_overlap_pct']}%)")
        else:
            print(f"  LP: splits MISSING (train_exists={lp['train_pos_exists']} test_exists={lp['test_pos_exists']})")
        ie = r["edge_endpoint_integrity"]
        print(f"  integrity: edges_with_missing_node={ie['n_edges_with_missing_endpoint_node']}")


if __name__ == "__main__":
    main()
