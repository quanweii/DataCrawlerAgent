#!/usr/bin/env python3
"""Summarize protocol-wise modality ablations from per-run JSON artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


DOMAINS = ("arxiv", "steam", "tmdb", "wikiart")
PROTOCOLS = ("transductive", "split_disjoint", "temporal", "group_inductive")
EXPECTED = {
    "arxiv": ("text", "image", "formula", "text+image", "text+formula", "image+formula", "text+image+formula"),
    "steam": ("text", "image", "text+image"),
    "tmdb": ("text", "image", "text+image"),
    "wikiart": ("text", "image", "text+image"),
}


def slug(modality: str) -> str:
    return modality.replace("+", "_")


def best_model(payload: dict) -> tuple[str, dict]:
    for metrics in payload["models"].values():
        if not metrics.get("val_accs"):
            raise ValueError("Missing validation scores: replay NC before aggregation; test scores cannot select models.")
    return max(payload["models"].items(), key=lambda item: statistics.mean(item[1]["val_accs"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="benchmark/results/newdomain_nc/protocol_matrix",
    )
    parser.add_argument(
        "--output-prefix",
        default="benchmark/results/newdomain_nc/protocol_matrix/protocol_modality_summary",
    )
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    rows: list[dict] = []
    missing: list[str] = []
    payloads: dict[str, dict[str, dict]] = {}

    for domain in DOMAINS:
        singles = tuple(m for m in EXPECTED[domain] if "+" not in m)
        full = EXPECTED[domain][-1]
        for protocol in PROTOCOLS:
            entries: dict[str, dict] = {}
            for modality in EXPECTED[domain]:
                path = input_dir / f"nc_{domain}_{protocol}_{slug(modality)}.json"
                if not path.exists():
                    missing.append(str(path))
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                payloads.setdefault(domain, {})[f"{protocol}/{modality}"] = payload
                model, metrics = best_model(payload)
                entries[modality] = {
                    "model": model,
                    "val_acc": statistics.mean(metrics["val_accs"]),
                    "acc": metrics["acc"],
                    "acc_std": metrics["acc_std"],
                    "f1": metrics["f1"],
                    "f1_std": metrics["f1_std"],
                }
            if full not in entries or any(modality not in entries for modality in singles):
                continue
            best_single = max(singles, key=lambda modality: entries[modality]["val_acc"])
            rows.append(
                {
                    "domain": domain,
                    "protocol": protocol,
                    "best_single_modality": best_single,
                    "best_single": entries[best_single],
                    "full_modality": full,
                    "full": entries[full],
                    "full_minus_best_single_acc": entries[full]["acc"] - entries[best_single]["acc"],
                    "all_modalities": entries,
                }
            )

    if missing:
        raise SystemExit("Missing result files:\n" + "\n".join(missing))

    # Concurrent modality batches may each write a partial domain matrix. Rebuild
    # the authoritative matrix from the complete per-run artifacts.
    for domain in DOMAINS:
        matrix_path = input_dir / f"nc_{domain}_protocol_matrix.json"
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        matrix["protocols"] = list(PROTOCOLS)
        matrix["modalities"] = list(EXPECTED[domain])
        matrix["results"] = {
            f"{protocol}/{modality}": payloads[domain][f"{protocol}/{modality}"]
            for modality in EXPECTED[domain]
            for protocol in PROTOCOLS
        }
        matrix_path.write_text(json.dumps(matrix, indent=2), encoding="utf-8")

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps({"selection": "model and single modality by mean validation accuracy; test reporting only", "rows": rows}, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Protocol-wise modality summary",
        "",
        "> Each cell selects its architecture by mean validation accuracy; the single modality is also selected by validation. A negative gap means full fusion underperforms the best single modality.",
        "",
        "| Domain | Protocol | Best single (model) | Single acc | Full fusion (model) | Full acc | Full−single |",
        "|---|---|---|---:|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['domain']} | {row['protocol']} | "
            f"{row['best_single_modality']} ({row['best_single']['model']}) | "
            f"{row['best_single']['acc']:.4f}±{row['best_single']['acc_std']:.4f} | "
            f"{row['full_modality']} ({row['full']['model']}) | "
            f"{row['full']['acc']:.4f}±{row['full']['acc_std']:.4f} | "
            f"{row['full_minus_best_single_acc']:+.4f} |"
        )
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(prefix.with_suffix(".json"))
    print(prefix.with_suffix(".md"))


if __name__ == "__main__":
    main()
