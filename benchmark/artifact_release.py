"""Opt-in, fail-closed publisher for benchmark bundles (added 2026-09-18).

Not the historical production publisher. This adds bundle-level checks to
existing protocol primitives. Rejects instead of silently repairing artifacts.
The trusted contract must be retained separately from a candidate bundle.
"""
from pathlib import Path
import hashlib
import json
import torch
from benchmark.data.splits import stratified_split


def digest_tensor(t):
    return hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()


def audit_bundle(b, contract):
    errors = []
    n = len(b['ids'])
    if len(set(b['ids'])) != n:
        errors.append('duplicate_entity')
    e = b['edges']
    in_range = not e.numel() or (int(e.min()) >= 0 and int(e.max()) < n)
    if not in_range:
        errors.append('dangling_endpoint')
    if e.numel() and bool(e[0].eq(e[1]).any()):
        errors.append('self_loop')
    active = set(b['edge_types'])
    if any(b['relation_lineage'].get(r, {}).get('target_derived') for r in active):
        errors.append('target_relation')
    if any(f.get('target_derived') for f in b['feature_lineage']):
        errors.append('target_feature')
    if b['feature_ids'] != b['ids'] or digest_tensor(b['x']) != contract['feature_sha256']:
        errors.append('feature_version_mismatch')
    tr, va, te = b['masks']
    if bool((tr.int()+va.int()+te.int()).gt(1).any()):
        errors.append('node_partition_overlap')
    assigned = tr | va | te
    if b['protocol'] == 'temporal':
        missing = torch.tensor([t is None for t in contract['times']])
        if bool((assigned & missing).any()):
            errors.append('missing_time_admitted')
        ranges = [[contract['times'][i] for i in m.nonzero().flatten().tolist()
                   if contract['times'][i] is not None] for m in b['masks']]
        if any(not a for a in ranges) or any(max(ranges[i]) >= min(ranges[i+1])
                                             for i in range(2) if ranges[i] and ranges[i+1]):
            errors.append('temporal_order')
    if b['protocol'] == 'group_inductive':
        gs = [{contract['groups'][i] for i in m.nonzero().flatten().tolist()} for m in b['masks']]
        if gs[0] & gs[1] or gs[0] & gs[2] or gs[1] & gs[2]:
            errors.append('group_overlap')
    if b['protocol'] in ('transductive', 'split_disjoint') and contract['stratified_required']:
        expected = stratified_split(b['y'], seed=contract['split_seed'])
        # Class counts, not exact shuffled membership: equivalent valid splits pass.
        for left, right in zip(b['masks'], expected):
            if not torch.equal(torch.bincount(b['y'][left], minlength=contract['classes']),
                               torch.bincount(b['y'][right], minlength=contract['classes'])):
                errors.append('declared_stratification_mismatch')
                break
    if in_range and b['protocol'] != 'transductive' and e.numel():
        split = torch.full((n,), -1)
        for i, m in enumerate(b['masks']):
            split[m] = i
        if bool(((split[e[0]] != split[e[1]]) | (split[e[0]] < 0)).any()):
            errors.append('cross_split_edge')
    allowed = set(va.nonzero().flatten().tolist())
    if not set(b['selection_nodes']).issubset(allowed):
        errors.append('test_label_checkpoint_access')
    return sorted(set(errors))


def publish_bundle(bundle, contract, destination, audited=True):
    """Write only accepted bundles; existing paths are never overwritten."""
    path = Path(destination)
    if path.exists():
        raise FileExistsError(path)
    errors = audit_bundle(bundle, contract) if audited else []
    record = {'published': not errors, 'errors': errors, 'audited': audited}
    path.parent.mkdir(parents=True, exist_ok=True)
    if not errors:
        torch.save(bundle, path)
        record['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix('.receipt.json').write_text(json.dumps(record, indent=2))
    return record
