"""Generate four auditable evaluation protocols for the four gold MAGs.

The raw MAG is immutable.  A protocol view consists of node masks and a view
edge_index; frozen per-node embeddings are reused because they do not depend on
the split.  Temporal and group-inductive views retain only within-split edges,
so validation/test nodes are evaluated as unseen periods/groups rather than by
message passing across the training boundary.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from .splits import make_split_disjoint_view, stratified_split

PROTOCOLS = ("transductive", "split_disjoint", "temporal", "group_inductive")
ROOT = Path(__file__).resolve().parents[2]

# These exclusions are based on field lineage, never on a homophily threshold.
TARGET_DERIVED_RELATIONS = {
    "arxiv": (),
    "steam": ("shared_genre",),
    "tmdb": ("shared_genre",),
    "wikiart": ("shared_style",),
}


def protocol_excluded_relations(domain: str) -> Tuple[str, ...]:
    return tuple(TARGET_DERIVED_RELATIONS.get(domain, ()))

DOMAIN_META = {
    "steam": {
        "raw": "data/steam_raw/steam_games_10k.json", "id": "appid",
        "time": "release_date", "group": "developers", "group_semantics": "primary developer",
    },
    "tmdb": {
        "raw": "data/tmdb_raw/tmdb_movies_raw.json", "id": "movie_id",
        "time": "release_date", "group": "directors", "group_semantics": "primary director",
    },
    "wikiart": {
        "raw": "data/wikiart_raw/wikiart_v2_enriched.json", "id": "painting_id",
        "time": "year", "group": "artist", "group_semantics": "artist",
    },
    "arxiv": {
        "raw": None, "id": None, "time": "arxiv_id_month",
        "group": "first_author_from_extracted_paper", "group_semantics": "first author",
    },
}


def _stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def _parse_time(domain: str, value, node_id: str) -> float | None:
    if domain == "arxiv":
        m = re.match(r"^(\d{2})(\d{2})\.", node_id)
        if not m:
            return None
        yy, month = int(m.group(1)), int(m.group(2))
        year = 2000 + yy if yy <= 90 else 1900 + yy
        return float(year * 12 + month)
    if value in (None, ""):
        return None
    if domain == "wikiart":
        m = re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", str(value))
        return float(m.group(1)) if m else None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d %b, %Y", "%b %d, %Y", "%Y"):
        try:
            dt = datetime.strptime(text, fmt)
            return float(dt.year * 12 + dt.month)
        except ValueError:
            pass
    return None


def _clean_primary(value) -> str | None:
    if isinstance(value, list):
        value = value[0] if value else None
    if value in (None, ""):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip().casefold()
    return text or None


def _arxiv_first_author(node_id: str) -> str | None:
    for sub in ("cs_si", "cs_lo", "math_lo", "physics_socph"):
        path = ROOT / "data" / f"{sub}_papers" / node_id / f"{node_id}.md"
        if not path.exists():
            continue
        try:
            lines = [x.strip() for x in path.read_text(encoding="utf-8", errors="ignore")[:8192].splitlines() if x.strip()]
        except OSError:
            return None
        # First non-heading line after the title is normally the author line.
        for line in lines[1:8]:
            if line.startswith("##") or line.lower().startswith("abstract"):
                continue
            first = re.split(r"[,;]", line, maxsplit=1)[0]
            first = re.sub(r"[*†‡\d]+$", "", first).strip()
            if 2 <= len(first) <= 100:
                return first.casefold()
        return None
    return None


def _resolve_path(row: Mapping, field_path: str):
    """Resolve ``a.b`` and ``field[0]`` references used by domain YAML."""
    if not field_path:
        return None
    current = row
    for token in re.split(r"\.(?![^\[]*\])", field_path):
        match = re.fullmatch(r"([^\[]+)(?:\[(\d+)\])?", token)
        if not match or not isinstance(current, Mapping):
            return None
        current = current.get(match.group(1))
        if match.group(2) is not None:
            if not isinstance(current, list):
                return None
            index = int(match.group(2))
            current = current[index] if index < len(current) else None
    return current


def load_protocol_metadata(domain: str, node_ids: Sequence[str],
                           metadata_config: Mapping[str, object] | None = None) -> Dict[str, object]:
    """Align timestamp and primary semantic group to canonical node order."""
    frozen = ROOT / "data" / "protocol_metadata" / (domain + ".json")
    if metadata_config is None and frozen.exists():
        cached = json.loads(frozen.read_text(encoding="utf-8"))
        positions = {str(x): i for i, x in enumerate(cached.pop("node_ids"))}
        indices = [positions[str(x)] for x in node_ids]
        cached["times"] = [cached["times"][i] for i in indices]
        cached["groups"] = [cached["groups"][i] for i in indices]
        return cached
    if metadata_config:
        cfg = {
            "raw": metadata_config.get("data_file"),
            "id": metadata_config.get("id_field"),
            "time": metadata_config.get("time_field"),
            "group": metadata_config.get("group_field"),
            "group_semantics": metadata_config.get("group_semantics", "primary entity group"),
            "id_sanitize": metadata_config.get("id_sanitize"),
        }
    else:
        if domain not in DOMAIN_META:
            raise ValueError(f"No protocol metadata configuration for {domain}")
        cfg = DOMAIN_META[domain]
    raw_by_id: Mapping[str, Mapping] = {}
    if cfg["raw"]:
        raw_path = Path(str(cfg["raw"]))
        if not raw_path.is_absolute():
            raw_path = ROOT / raw_path
        rows = json.loads(raw_path.read_text(encoding="utf-8"))
        sanitize = cfg.get("id_sanitize")
        raw_by_id = {}
        for row in rows:
            row_id = _resolve_path(row, str(cfg["id"]))
            if row_id is None:
                continue
            key = str(row_id)
            if sanitize:
                key = re.sub(str(sanitize), "_", key)
            raw_by_id[key] = row

    times: List[float | None] = []
    groups: List[str] = []
    missing_time = missing_group = 0
    for node_id in map(str, node_ids):
        row = raw_by_id.get(node_id, {})
        time_value = node_id if domain == "arxiv" and not metadata_config else _resolve_path(row, str(cfg["time"]))
        timestamp = _parse_time(domain, time_value, node_id)
        group = (_arxiv_first_author(node_id) if domain == "arxiv" and not metadata_config
                 else _clean_primary(_resolve_path(row, str(cfg["group"]))))
        if timestamp is None:
            missing_time += 1
        if group is None:
            missing_group += 1
            group = f"__unknown_singleton__:{node_id}"
        times.append(timestamp)
        groups.append(group)
    return {
        "times": times, "groups": groups,
        "time_field": cfg["time"], "group_field": cfg["group"],
        "group_semantics": cfg["group_semantics"],
        "time_coverage": 1.0 - missing_time / max(len(node_ids), 1),
        "group_coverage": 1.0 - missing_group / max(len(node_ids), 1),
    }


def _masks_from_assignment(assignment: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a = torch.as_tensor(assignment, dtype=torch.long)
    return tuple(a.eq(i) for i in range(3))  # type: ignore[return-value]


def temporal_split(times: Sequence[float | None], ratios=(0.6, 0.2, 0.2)):
    """Chronological split with equal timestamps kept in the same partition."""
    valid = [float(t) for t in times if t is not None]
    if len(valid) < 3:
        raise ValueError("Temporal protocol requires at least three timestamped nodes")
    unique = sorted(set(valid))
    if len(unique) < 3:
        raise ValueError("Temporal protocol requires at least three distinct timestamps")
    n = len(valid)
    c = Counter(float(t) for t in times if t is not None)
    target1, target2 = ratios[0] * n, (ratios[0] + ratios[1]) * n
    running, cut1, cut2 = 0, unique[0], unique[-1]
    for value in unique:
        running += c[value]
        if running <= target1:
            cut1 = value
        if running <= target2:
            cut2 = value
    assignment = []
    for t in times:
        if t is None:
            # Missing chronology is excluded from this derived view, not
            # silently treated as old/training data. It remains in the raw MAG.
            assignment.append(-1)
        elif float(t) <= cut1:
            assignment.append(0)
        elif float(t) <= cut2:
            assignment.append(1)
        else:
            assignment.append(2)
    return _masks_from_assignment(assignment), (cut1, cut2)


def group_inductive_split(groups: Sequence[str], y: torch.Tensor, ratios=(0.6, 0.2, 0.2), seed=1):
    """Deterministic greedy assignment of whole primary groups to splits."""
    members: Dict[str, List[int]] = defaultdict(list)
    for i, group in enumerate(groups):
        members[str(group)].append(i)
    rng = np.random.RandomState(seed)
    keys = list(members)
    rng.shuffle(keys)
    keys.sort(key=lambda k: len(members[k]), reverse=True)
    n, n_classes = len(groups), int(y.max().item()) + 1
    target_n = np.asarray(ratios, dtype=float) * n
    global_cls = np.bincount(y.cpu().numpy(), minlength=n_classes).astype(float)
    target_cls = np.outer(np.asarray(ratios, dtype=float), global_cls)
    split_n = np.zeros(3, dtype=float)
    split_cls = np.zeros((3, n_classes), dtype=float)
    group_to_split = {}
    y_np = y.cpu().numpy()
    for key in keys:
        idx = members[key]
        counts = np.bincount(y_np[idx], minlength=n_classes).astype(float)
        scores = []
        for s in range(3):
            size_need = (target_n[s] - split_n[s]) / max(target_n[s], 1.0)
            present = counts > 0
            class_need = float(np.mean(
                (target_cls[s, present] - split_cls[s, present]) /
                np.maximum(target_cls[s, present], 1.0))) if present.any() else 0.0
            overflow = max(0.0, split_n[s] + len(idx) - target_n[s] * 1.05) / max(target_n[s], 1.0)
            scores.append(size_need + 0.25 * class_need - 8.0 * overflow)
        chosen = int(np.argmax(scores))
        group_to_split[key] = chosen
        split_n[chosen] += len(idx)
        split_cls[chosen] += counts
    assignment = [group_to_split[str(g)] for g in groups]
    return _masks_from_assignment(assignment), group_to_split


def _view_stats(edge_index: torch.Tensor, masks, y: torch.Tensor) -> Dict[str, object]:
    sid = torch.full((masks[0].numel(),), -1, dtype=torch.long)
    for i, mask in enumerate(masks):
        sid[mask] = i
    src, dst = edge_index
    cross = int((sid[src] != sid[dst]).sum().item()) if edge_index.numel() else 0
    node_counts = [int(m.sum()) for m in masks]
    class_counts = [int(torch.unique(y[m]).numel()) for m in masks]
    covered = set(src.tolist()) | set(dst.tolist()) if edge_index.numel() else set()
    assigned = sum(node_counts)
    return {
        "nodes": node_counts,
        "unassigned_nodes": int(sid.eq(-1).sum()),
        "classes": class_counts,
        "train_missing_classes": int(torch.unique(y).numel() - class_counts[0]),
        "directed_edges": int(edge_index.size(1)),
        "cross_split_directed_edges": cross,
        "isolated_fraction_among_assigned": float((assigned - len(covered)) / max(assigned, 1)),
    }


def _temporal_label_support(y: torch.Tensor, masks) -> Dict[str, object]:
    """Quantify label emergence separately from within-support distribution shift."""
    train_mask, _, test_mask = masks
    train_y, test_y = y[train_mask], y[test_mask]
    global_classes = sorted(set(y.tolist()))
    train_classes = set(train_y.tolist())
    test_classes = set(test_y.tolist())
    visible_test = torch.tensor(
        [int(label) in train_classes for label in test_y.tolist()], dtype=torch.bool)
    coverable_nodes = int(visible_test.sum())
    test_nodes = int(test_y.numel())
    coverable_mass = coverable_nodes / max(test_nodes, 1)

    keys = sorted(train_classes)
    p = np.asarray([int(train_y.eq(label).sum()) for label in keys], dtype=float)
    q = np.asarray([int(test_y.eq(label).sum()) for label in keys], dtype=float)
    p = p / max(p.sum(), 1.0)
    q = q / max(q.sum(), 1.0)
    midpoint = 0.5 * (p + q)

    def kl(left, right):
        valid = left > 0
        return float(np.sum(left[valid] * np.log(left[valid] / right[valid])))

    visible_counts = [int(test_y.eq(label).sum()) for label in keys]
    return {
        "global_classes": len(global_classes),
        "train_visible_classes": len(train_classes),
        "train_missing_global_classes": len(set(global_classes) - train_classes),
        "test_classes": len(test_classes),
        "unseen_test_classes": len(test_classes - train_classes),
        "coverable_test_nodes": coverable_nodes,
        "test_nodes": test_nodes,
        "coverable_test_mass": coverable_mass,
        "unseen_test_mass": 1.0 - coverable_mass,
        "train_support_oracle_ceiling": coverable_mass,
        "largest_train_visible_class_test_mass": (
            max(visible_counts, default=0) / max(test_nodes, 1)),
        "seen_class_prior_tvd": float(0.5 * np.abs(p - q).sum()),
        "seen_class_prior_js_nats": float(0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)),
        "interpretation": (
            "train-support coverage bounds supervised closed-world recognition; "
            "prior divergence diagnoses shift within train-visible labels and is not a leakage verdict"),
    }


def generate_protocol_views(domain: str, node_ids: Sequence[str], y: torch.Tensor,
                            edge_index: torch.Tensor, seed: int = 1,
                            ratios=(0.6, 0.2, 0.2),
                            excluded_relations: Sequence[str] = (),
                            metadata_config: Mapping[str, object] | None = None,
                            protocols: Sequence[str] = PROTOCOLS) -> Dict[str, object]:
    """Create all four views and a JSON-friendly audit manifest."""
    requested = tuple(protocols)
    unknown = sorted(set(requested) - set(PROTOCOLS))
    if unknown:
        raise ValueError(f"Unknown protocols: {unknown}")
    needs_metadata = any(p in requested for p in ("temporal", "group_inductive"))
    meta = (load_protocol_metadata(domain, node_ids, metadata_config)
            if needs_metadata else {
                "times": [None] * len(node_ids), "groups": [],
                "time_field": "", "group_field": "", "group_semantics": "",
                "time_coverage": 0.0, "group_coverage": 0.0,
            })
    stratified = stratified_split(y, ratios=ratios, seed=seed)
    views = {}
    temporal_masks = temporal_cuts = None
    group_masks = group_assignment = None
    if "transductive" in requested:
        views["transductive"] = {"masks": stratified, "edge_index": edge_index.clone()}
    if "split_disjoint" in requested:
        views["split_disjoint"] = {"masks": stratified,
            "edge_index": make_split_disjoint_view(edge_index, stratified)}
    if "temporal" in requested:
        temporal_masks, temporal_cuts = temporal_split(meta["times"], ratios=ratios)
        views["temporal"] = {"masks": temporal_masks,
            "edge_index": make_split_disjoint_view(edge_index, temporal_masks)}
    if "group_inductive" in requested:
        group_masks, group_assignment = group_inductive_split(meta["groups"], y, ratios=ratios, seed=seed)
        views["group_inductive"] = {"masks": group_masks,
            "edge_index": make_split_disjoint_view(edge_index, group_masks)}
    manifest = {
        "domain": domain, "seed": seed, "ratios": list(ratios),
        "raw_graph_immutable": True,
        "sanitized_base_graph": True,
        "provenance_excluded_relations": list(excluded_relations),
        "base_view": "provenance_sanitized_graph",
        "excluded_target_derived_relations": list(
            protocol_excluded_relations(domain) or tuple(excluded_relations)),
        "representation_policy": "frozen per-node embeddings are encoded once, sanitized before encoding, and reused across protocol views; fusion is fitted without labels",
        "metadata": {k: v for k, v in meta.items() if k not in ("times", "groups")},
        "requested_protocols": list(requested),
        "available_protocols": list(views),
        "temporal_cutoffs": list(temporal_cuts) if temporal_cuts else None,
        "group_counts": ([sum(1 for s in group_assignment.values() if s == i) for i in range(3)]
                         if group_assignment is not None else None),
        "temporal_label_support": (
            _temporal_label_support(y, temporal_masks)
            if temporal_masks is not None else None),
        "views": {name: _view_stats(v["edge_index"], v["masks"], y) for name, v in views.items()},
    }
    # Explicit invariants: group overlap and temporal order.
    group_sets = [set() for _ in range(3)]
    for group, split in (group_assignment or {}).items():
        group_sets[split].add(group)
    manifest["checks"] = {
        "non_temporal_views_assign_all_nodes": all(
            sum(int(views[p]["masks"][i].sum()) for i in range(3)) == len(node_ids)
            for p in ("transductive", "split_disjoint", "group_inductive") if p in views),
        "temporal_assigned_nodes_match_coverage": (
            True if temporal_masks is None else
            sum(int(temporal_masks[i].sum()) for i in range(3)) ==
            sum(t is not None for t in meta["times"])),
        "group_overlap": (None if group_assignment is None else len(
            (group_sets[0] & group_sets[1]) | (group_sets[0] & group_sets[2]) |
            (group_sets[1] & group_sets[2]))),
        "inductive_views_have_no_cross_split_edges": all(
            manifest["views"][p]["cross_split_directed_edges"] == 0
            for p in ("split_disjoint", "temporal", "group_inductive") if p in views),
    }
    return {"views": views, "manifest": manifest}


def save_protocol_suite(domain: str, suite: Mapping[str, object], output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / f"{domain}_protocol_views.pt"
    manifest_path = output_dir / f"{domain}_protocol_manifest.json"
    torch.save(suite["views"], tensor_path)
    manifest_path.write_text(json.dumps(suite["manifest"], ensure_ascii=False, indent=2), encoding="utf-8")
    return tensor_path, manifest_path
