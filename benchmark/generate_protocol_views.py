#!/usr/bin/env python3
"""Materialize the four protocol views for one or all gold datasets."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.data.domain_loader import load_domain_graph
from benchmark.data.protocols import (generate_protocol_views,
                                      protocol_excluded_relations,
                                      save_protocol_suite)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="arxiv,steam,tmdb,wikiart")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--output-dir", default="benchmark/results/protocol_views")
    args = ap.parse_args()
    output = ROOT / args.output_dir
    manifests = {}
    for domain in [x.strip() for x in args.domains.split(",") if x.strip()]:
        excluded = protocol_excluded_relations(domain)
        graph = load_domain_graph(domain, exclude_relations=excluded)
        suite = generate_protocol_views(
            domain, graph["node_ids"], graph["y"], graph["edge_index"],
            seed=args.seed, excluded_relations=excluded)
        tensor_path, manifest_path = save_protocol_suite(domain, suite, output)
        manifests[domain] = suite["manifest"]
        print(f"[{domain}] {tensor_path} | {manifest_path}")
    (output / "protocol_manifest_all.json").write_text(
        json.dumps(manifests, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
