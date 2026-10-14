#!/usr/bin/env python3
"""Synchronize regenerated protocol manifests into existing result bundles.

This updates audit metadata only. Model metrics and per-seed predictions are
left untouched because protocol masks and edge views are deterministic and
unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark" / "results"
DOMAINS = ("arxiv", "steam", "tmdb", "wikiart")


def main():
    updated = []
    for domain in DOMAINS:
        manifest_path = RESULTS / "protocol_views" / f"{domain}_protocol_manifest.json"
        matrix_path = (RESULTS / "newdomain_nc" / "protocol_matrix" /
                       f"nc_{domain}_protocol_matrix.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))

        # Refuse to attach metadata to a result produced from different views.
        for protocol, stats in manifest["views"].items():
            for modality in matrix["modalities"]:
                cell = matrix["results"][f"{protocol}/{modality}"]
                if cell["directed_edges"] != stats["directed_edges"]:
                    raise ValueError(
                        f"{domain}/{protocol}: manifest edges changed "
                        f"({stats['directed_edges']} != {cell['directed_edges']})")
                if list(cell["split_sizes"]) != list(stats["nodes"]):
                    raise ValueError(f"{domain}/{protocol}: split sizes changed")
        matrix["protocol_manifest"] = manifest
        matrix_path.write_text(
            json.dumps(matrix, indent=2, ensure_ascii=False), encoding="utf-8")
        updated.append(domain)
    print(json.dumps({"updated": updated, "metrics_unchanged": True}))


if __name__ == "__main__":
    main()
