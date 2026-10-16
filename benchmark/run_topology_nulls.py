#!/usr/bin/env python3
"""Topology-only predictability with label-permutation and degree-preserving nulls.

This is a diagnostic experiment, never an automatic leakage verdict.  The
degree-preserving edge-swap null retains every node degree while disrupting
which nodes are connected.  The label-permutation null retains the graph and
class counts while disrupting the assignment of training labels.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.data.domain_loader import load_domain_graph
from benchmark.data.protocols import protocol_excluded_relations
from benchmark.data.splits import stratified_split


def neighbor_vote(y, edge_index, train_mask, test_mask, seed_labels=None):
    """One-hop vote using labels on training neighbours only."""
    labels = y if seed_labels is None else seed_labels
    n, c = y.numel(), int(y.max().item()) + 1
    src, dst = edge_index
    use = train_mask[src] & test_mask[dst]
    code = dst[use] * c + labels[src[use]]
    votes = torch.bincount(code, minlength=n * c).reshape(n, c)
    covered = test_mask & votes.sum(1).gt(0)
    pred = votes.argmax(1)
    acc_all = float((pred[test_mask] == y[test_mask]).float().mean())
    acc_covered = float((pred[covered] == y[covered]).float().mean()) if covered.any() else None
    truth = y[test_mask]
    majority = float((truth == torch.bincount(truth).argmax()).float().mean())
    return {
        "accuracy_all_test": acc_all,
        "accuracy_covered_test": acc_covered,
        "test_coverage": float(covered.sum() / max(int(test_mask.sum()), 1)),
        "majority_accuracy": majority,
    }


def undirected_edges(edge_index):
    return {(min(int(a), int(b)), max(int(a), int(b)))
            for a, b in zip(edge_index[0].tolist(), edge_index[1].tolist()) if a != b}


def degree_preserving_swap(edge_index, seed, swaps_per_edge=0.25):
    """Simple double-edge-swap Markov chain; node degrees are exact invariants."""
    rng = random.Random(seed)
    edges = list(undirected_edges(edge_index))
    edge_set = set(edges)
    target = max(1, int(len(edges) * swaps_per_edge))
    accepted = attempts = 0
    max_attempts = target * 20
    while accepted < target and attempts < max_attempts:
        attempts += 1
        i, j = rng.sample(range(len(edges)), 2)
        a, b = edges[i]; c, d = edges[j]
        if len({a, b, c, d}) < 4:
            continue
        if rng.random() < 0.5:
            e1, e2 = tuple(sorted((a, d))), tuple(sorted((c, b)))
        else:
            e1, e2 = tuple(sorted((a, c))), tuple(sorted((b, d)))
        if e1[0] == e1[1] or e2[0] == e2[1] or e1 in edge_set or e2 in edge_set:
            continue
        old1, old2 = edges[i], edges[j]
        edge_set.remove(old1); edge_set.remove(old2)
        edge_set.add(e1); edge_set.add(e2)
        edges[i], edges[j] = e1, e2
        accepted += 1
    src, dst = [], []
    for a, b in edges:
        src.extend((a, b)); dst.extend((b, a))
    return torch.tensor([src, dst], dtype=torch.long), {
        "requested_swaps": target, "accepted_swaps": accepted, "attempts": attempts,
        "degree_sequence_preserved": True,
    }


def summarize(values):
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std()),
            "q025": float(np.quantile(arr, 0.025)), "q975": float(np.quantile(arr, 0.975))}


def run_domain(domain, permutation_runs, rewire_runs, seed):
    graph = load_domain_graph(domain, exclude_relations=protocol_excluded_relations(domain))
    y, edge_index = graph["y"], graph["edge_index"]
    train, _, test = stratified_split(y, seed=seed)
    observed = neighbor_vote(y, edge_index, train, test)
    rng = np.random.RandomState(seed + 1000)
    train_idx = torch.where(train)[0]
    perm = []
    for _ in range(permutation_runs):
        seed_y = y.clone()
        order = torch.as_tensor(rng.permutation(len(train_idx)), dtype=torch.long)
        seed_y[train_idx] = y[train_idx[order]]
        perm.append(neighbor_vote(y, edge_index, train, test, seed_y)["accuracy_all_test"])
    rewired, swap_meta = [], []
    original_degree = torch.bincount(edge_index[0], minlength=y.numel())
    for i in range(rewire_runs):
        null_edges, meta = degree_preserving_swap(edge_index, seed + 2000 + i)
        meta["degree_sequence_verified"] = bool(torch.equal(
            original_degree, torch.bincount(null_edges[0], minlength=y.numel())))
        swap_meta.append(meta)
        rewired.append(neighbor_vote(y, null_edges, train, test)["accuracy_all_test"])
    return {
        "domain": domain,
        "interpretation": "topology_predictability_diagnostic_not_leakage_verdict",
        "excluded_target_derived_relations": list(protocol_excluded_relations(domain)),
        "observed": observed,
        "training_label_permutation_null": {"runs": permutation_runs, **summarize(perm)},
        "degree_preserving_rewire_null": {"runs": rewire_runs, **summarize(rewired),
                                           "swap_runs": swap_meta},
        "observed_minus_permutation_mean": observed["accuracy_all_test"] - float(np.mean(perm)),
        "observed_minus_rewire_mean": observed["accuracy_all_test"] - float(np.mean(rewired)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="arxiv,steam,tmdb,wikiart")
    ap.add_argument("--permutation-runs", type=int, default=100)
    ap.add_argument("--rewire-runs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    results = {d: run_domain(d, args.permutation_runs, args.rewire_runs, args.seed)
               for d in args.domains.split(",") if d}
    out = ROOT / "benchmark" / "results" / "topology_nulls.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# Topology-only predictability nulls", "",
          "> Diagnostic only. High topology predictability is not, by itself, leakage.", "",
          "| Domain | Observed | Label-permutation null | Degree-preserving null | Obs−perm | Obs−rewire | Test coverage |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for d, r in results.items():
        md.append(f"| {d} | {r['observed']['accuracy_all_test']:.4f} | "
                  f"{r['training_label_permutation_null']['mean']:.4f} | "
                  f"{r['degree_preserving_rewire_null']['mean']:.4f} | "
                  f"{r['observed_minus_permutation_mean']:+.4f} | "
                  f"{r['observed_minus_rewire_mean']:+.4f} | "
                  f"{r['observed']['test_coverage']:.4f} |")
    (out.with_suffix(".md")).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
