"""Opt-in, source-backed record checks. This is not a semantic correspondence model.

Trusted entries must be retained separately from candidate exports. Warnings
describe optional coverage and never constitute a record-rejection decision.
"""
from collections import Counter
from pathlib import Path
import hashlib
from PIL import Image


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_records(records, trusted):
    counts = Counter(str(r.get('id')) for r in records if r.get('id') not in (None, ''))
    rows = []
    for index, record in enumerate(records):
        errors, warnings = [], []
        identity = str(record.get('id', ''))
        source = trusted.get(identity)
        for key in ['id', 'type', 'image_url', 'image_path', 'text_path']:
            if record.get(key) in (None, ''):
                errors.append('missing_required:' + key)
        if counts[identity] > 1:
            errors.append('duplicate_entity_id')
        if source is None:
            errors.append('unverifiable_source_id')
        else:
            for key in ['type', 'image_url']:
                if record.get(key) != source.get(key):
                    errors.append('source_field_mismatch:' + key)
            for modality in ['image', 'text']:
                path = record.get(modality + '_path')
                try:
                    actual = digest(path)
                    if actual != source.get(modality + '_sha256'):
                        errors.append('source_bytes_mismatch:' + modality)
                    if record.get(modality + '_sha256') != actual:
                        errors.append('declared_hash_mismatch:' + modality)
                    if modality == 'image':
                        with Image.open(path) as image:
                            image.load()
                            if min(image.size) < 32:
                                errors.append('image_below_minimum_32px')
                except (OSError, ValueError, TypeError):
                    errors.append('unreadable:' + modality)
            if record.get('text_source_fields') != source.get('text_source_fields'):
                errors.append('source_text_field_declaration_mismatch')
            if not source.get('has_description'):
                warnings.append('optional_description_absent')
            if not source.get('has_creator'):
                warnings.append('optional_creator_absent')
            if source.get('omitted_current_did_you_know'):
                warnings.append('legacy_fun_fact_alias_omits_current_field')
        rows.append(dict(index=index, id=identity, rejected=bool(errors),
                         errors=sorted(set(errors)), warnings=sorted(set(warnings))))
    return dict(rows=rows, records=len(rows), rejected=sum(r['rejected'] for r in rows),
                errors=dict(Counter(v for r in rows for v in r['errors'])),
                warnings=dict(Counter(v for r in rows for v in r['warnings'])))
