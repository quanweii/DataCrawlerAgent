#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extend the held-out contract-reuse audit to Amazon and Reddit (2026-09-19).

Pipeline per domain (deterministic, no LLM, no training):
  1. create minimal cs_lo stubs (title-only markdown, empty images/) if absent
  2. load domains/<d>/domain.yaml through the production loader
  3. build the co-attribute graph with crawlers.config_driven.build_graph_from_config
  4. serialize in the production graph format (nodes/edges/metadata)
  5. run the SAME audit_domain() used for Steam/TMDB/WikiArt/Spotify

Writes benchmark/results/heldout_contract_20260919.json. Never touches the
mature-domain graphs or the production protocol_views/ manifests.
"""
import json
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from domains import load_domain_yaml  # noqa: E402
from domains.domain_schema import yaml_to_graph_config  # noqa: E402
from crawlers.config_driven import build_graph_from_config  # noqa: E402

DOMAINS = ["amazon", "reddit"]
ID_FIELD = {"amazon": "asin", "reddit": "post_id"}
TEXT_FIELD = {"amazon": "title", "reddit": "title"}
OUT_JSON = PROJECT_ROOT / "benchmark" / "results" / "heldout_contract_20260919.json"


def stub_cslo(domain, items, cslo_dir):
    if cslo_dir.exists() and any(cslo_dir.iterdir()):
        print(f"[{domain}] cslo stubs already present, skipping")
        return
    cslo_dir.mkdir(parents=True, exist_ok=True)
    for it in items:
        iid = str(it[ID_FIELD[domain]])
        d = cslo_dir / iid
        d.mkdir(exist_ok=True)
        (d / f"{iid}.md").write_text(
            f"# {str(it.get(TEXT_FIELD[domain], iid))[:200]}\n", encoding="utf-8")
        (d / "images").mkdir(exist_ok=True)
    print(f"[{domain}] created {len(items)} cslo stubs at {cslo_dir}")


def build_domain(domain):
    yaml_path = PROJECT_ROOT / "domains" / domain / "domain.yaml"
    yc = load_domain_yaml(str(yaml_path))
    cfg = yaml_to_graph_config(yc)
    items = json.loads((PROJECT_ROOT / yc.data.data_file).read_text(encoding="utf-8"))
    cslo_dir = PROJECT_ROOT / yc.output_dirs.items
    stub_cslo(domain, items, cslo_dir)

    t0 = time.time()
    nodes, edges, stats = build_graph_from_config(items, str(cslo_dir), cfg)
    build_s = time.time() - t0

    out_path = PROJECT_ROOT / "data" / f"{domain}_datasets" / \
        f"{domain}_multimodal_graph_agent.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # keep any pre-existing legacy heterogeneous graph intact under a backup name
    if out_path.exists():
        backup = out_path.with_name(out_path.stem + "_legacy_heterogeneous.json")
        if not backup.exists():
            shutil.move(str(out_path), str(backup))
            print(f"[{domain}] moved legacy graph to {backup.name}")
    graph = {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "domain": domain,
            "built_by": "run_heldout_contract.py (production config_driven path)",
            "config": str(yaml_path),
            "build_seconds": round(build_s, 2),
            "stats": {k: (v if isinstance(v, (int, float, str)) else str(v))
                      for k, v in stats.items()},
        },
    }
    out_path.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
    print(f"[{domain}] nodes={len(nodes)} edges={len(edges)} "
          f"({build_s:.1f}s) -> {out_path.name}")
    return {"nodes": len(nodes), "edges": len(edges), "stats": stats,
            "build_seconds": round(build_s, 2)}


def audit(domain):
    sys.path.insert(0, str(PROJECT_ROOT / "benchmark"))
    from run_reusability_audit import audit_domain
    return audit_domain(domain)


def main():
    report = {"date": "2026-09-19", "purpose": "held-out contract reuse n=1 -> n=3",
              "domains": {}}
    for domain in DOMAINS:
        build = build_domain(domain)
        checks = audit(domain)
        report["domains"][domain] = {"build": build, "audit": checks}
        status = checks.get("status")
        print(f"[{domain}] audit status = {status}")
        if status != "pass":
            failed = {k: v for k, v in checks.items() if v is False}
            print(f"[{domain}] FAILED CHECKS: {failed}")
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
