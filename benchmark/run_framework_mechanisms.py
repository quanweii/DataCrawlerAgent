"""Paired schema-role comparison and production-validator fault injection.

This is a schema-stage mechanism study, NOT an end-to-end multi-agent baseline.
Raw datasets and benchmark graphs are read-only. Every network response is saved.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'benchmark/path_b'))
from agents.schema_discovery import validate_schema, profile_fields, _base_field
from domains import load_domain_yaml, yaml_to_task_config
from probe_p1_schema import _load_env

DOMAINS = ('steam', 'tmdb', 'wikiart', 'spotify')
SYSTEM = ('Design a homogeneous multimodal graph schema from the supplied profile. '
          'Choose a unique id_field, one meaningful categorical label_field, and relations '
          '[{name,group_by,cap}]. Fields can use [0] for a primary value of a list. '
          'The label must retain 2..100 classes with at least 20 examples each. '
          'Relations must use existing fields; do not re-encode the target field. '
          'Return only JSON. No benchmark performance scores are available.')


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def score(items, proposal, hint):
    if not isinstance(proposal, dict):
        return {'parsed': False, 'valid': False, 'raw_target_relation': None,
                'publishable': False, 'label_agreement': False, 'id_agreement': False}
    try:
        audit = validate_schema(items, proposal, 20)
        survivors = [r for r in audit['confirmed_schema'].get('relations', [])
                     if r['name'] not in audit['excluded_from_benchmark']
                     and audit['relation_lineage'][r['name']]['shared_fraction'] >= .05]
        return {'parsed': True, 'valid': audit['valid'],
                'raw_target_relation': bool(audit['excluded_from_benchmark']),
                'publishable': bool(audit['valid'] and survivors),
                'label_agreement': _base_field(proposal.get('label_field','')) == _base_field(hint['label_field']),
                'id_agreement': proposal.get('id_field') == hint['id_field'],
                'errors': audit['errors'], 'warnings': audit['warnings'],
                'excluded': audit['excluded_from_benchmark'], 'label_stats': audit['label_stats']}
    except Exception as exc:
        return {'parsed': False, 'valid': False, 'raw_target_relation': None,
                'publishable': False, 'label_agreement': False, 'id_agreement': False,
                'validator_error': type(exc).__name__}


def call(client, model, messages, seed):
    start = time.monotonic()
    try:
        r = client.chat.completions.create(model=model, messages=messages, temperature=.2,
                                           max_tokens=1200, seed=seed)
        raw = r.choices[0].message.content or ''
        match = re.search(r'\{[\s\S]*\}', raw)
        try:
            proposal = json.loads(match.group(0)) if match else None
        except ValueError:
            proposal = None
        return {'raw': raw, 'proposal': proposal, 'model_returned': r.model,
                'seconds': time.monotonic()-start, 'finish_reason': r.choices[0].finish_reason,
                'usage': r.usage.model_dump() if r.usage else None, 'messages': messages}
    except Exception as exc:
        return {'proposal': None, 'raw': '', 'seconds': time.monotonic()-start,
                'error_type': type(exc).__name__, 'status_code': getattr(exc,'status_code',None)}


def faults(domain, items, hint):
    # Ground truth is the known injected operation, independent of validator verdict.
    clean = copy.deepcopy(hint)
    clean['relations'] = [r for r in clean['relations']
                          if _base_field(r['group_by']) != _base_field(clean['label_field'])]
    cases = []
    for name in ('clean', 'renamed_relation', 'duplicate_id', 'missing_id',
                 'nonexistent_target', 'nonexistent_relation', 'target_relation'):
        proposal = copy.deepcopy(clean)
        rows = items
        expected_fault = name not in ('clean', 'renamed_relation')
        if name == 'renamed_relation':
            proposal['relations'][0]['name'] = 'independently_sourced_relation'
        elif name == 'duplicate_id':
            rows = [*items, copy.deepcopy(items[0])]
        elif name == 'missing_id':
            row = copy.deepcopy(items[0]); row.pop(_base_field(proposal['id_field']), None)
            rows = [row, *items[1:]]
        elif name == 'nonexistent_target':
            proposal['label_field'] = '__fault_missing_target__'
        elif name == 'nonexistent_relation':
            proposal['relations'].append({'name':'__fault_relation__','group_by':'__fault_missing_field__','cap':30})
        elif name == 'target_relation':
            proposal['relations'].append({'name':'__target_derived__','group_by':proposal['label_field'],'cap':30})
        audit = validate_schema(rows, proposal, 20)
        for variant in ('full', 'without_schema_gate', 'without_lineage_exclusion'):
            rejected = not audit['valid'] and variant != 'without_schema_gate'
            retained = {r['name'] for r in proposal['relations']}
            if variant != 'without_lineage_exclusion':
                retained -= set(audit['excluded_from_benchmark'])
            prevented = rejected or (name == 'target_relation' and '__target_derived__' not in retained)
            cases.append({'domain': domain, 'case': name, 'variant': variant,
                          'injected_fault': expected_fault, 'release_rejected': rejected,
                          'fault_prevented': prevented if expected_fault else None,
                          'false_rejection': rejected if not expected_fault else None,
                          'errors': audit['errors'], 'excluded': audit['excluded_from_benchmark']})
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    out = ROOT / 'benchmark/results/framework_mechanisms_20260915'
    out.mkdir(parents=True, exist_ok=True)
    result = {'design': 'paired_shared_proposal_schema_role_ablation',
              'scope': 'Schema-stage mechanism only; not production end-to-end superiority.',
              'max_output_tokens_per_call': 1200, 'temperature': .2,
              'cases': [], 'faults': [], 'sources': {}}
    client = None
    if not args.offline:
        from openai import OpenAI
        _load_env()
        client = OpenAI(api_key=os.getenv('API_KEY'), base_url=os.getenv('MODEL_BASE_URL'),
                        timeout=35, max_retries=0)
        model = os.getenv('MODEL_NAME', 'qwen-plus')
        result['configured_model'] = model
    for domain in DOMAINS:
        task = yaml_to_task_config(load_domain_yaml(str(ROOT/'domains'/domain/'domain.yaml')))
        rawpath = ROOT / task['data_file']
        raw = rawpath.read_bytes(); items = json.loads(raw)
        hint = task['schema_hint']
        result['sources'][domain] = {'data_sha256': hashlib.sha256(raw).hexdigest(),
                                     'count': len(items), 'id_reference': hint['id_field'],
                                     'label_reference': hint['label_field']}
        result['faults'].extend(faults(domain, items, hint))
        sample = random.Random(20260915).sample(items, min(300,len(items)))
        profile = json.dumps(profile_fields(sample), ensure_ascii=False)
        prompt = 'Profile of a 300-item sample; full-data validator available after proposal.\n'+profile
        if client:
            for seed in range(args.seeds):
                casepath = out / f'{domain}_seed{seed}.json'
                if casepath.exists():
                    case = json.loads(casepath.read_text())
                    result['cases'].append(case)
                    continue
                first_messages = [{'role':'system','content':SYSTEM},{'role':'user','content':prompt}]
                first = call(client, model, first_messages, seed)
                first['score'] = score(items, first['proposal'], hint)
                feedback = json.dumps({k:v for k,v in first['score'].items()
                                       if k not in ('label_agreement','id_agreement')})
                revision = 'Revise the proposal using this deterministic feedback. Output JSON only.\n'+feedback
                arms = {
                    'single_agent_revision': first_messages + [{'role':'assistant','content':first['raw']},
                                                               {'role':'user','content':revision}],
                    'independent_schema_reviewer': [
                        {'role':'system','content':SYSTEM+' You are an independent schema reviewer. Inspect and revise the supplied proposal.'},
                        {'role':'user','content':prompt+'\nInitial proposal:\n'+first['raw']+'\n'+revision}]
                }
                # Counterbalance request order; both receive identical substantive evidence.
                names = list(arms)
                if seed % 2: names.reverse()
                case = {'domain':domain, 'seed':seed, 'initial': first, 'arms': {}}
                for name in names:
                    response = call(client, model, arms[name], seed)
                    response['score'] = score(items,response['proposal'],hint)
                    case['arms'][name] = response
                save(casepath, case)
                result['cases'].append(case)
                print(domain, seed, {n:r['score']['publishable'] for n,r in case['arms'].items()}, flush=True)
                save(out/'results.json', result)
        save(out/'results.json', result)
    summary = {'fault_injection': {}, 'role_comparison': {},
               'claim_boundary': result['scope'], 'fresh_training_runs': 0}
    for variant in ('full','without_schema_gate','without_lineage_exclusion'):
        rows = [r for r in result['faults'] if r['variant']==variant]
        bad = [r for r in rows if r['injected_fault']]
        good = [r for r in rows if not r['injected_fault']]
        summary['fault_injection'][variant] = {'prevented':sum(r['fault_prevented'] for r in bad),
            'faults':len(bad),'false_rejections':sum(r['false_rejection'] for r in good),'controls':len(good)}
    for arm in ('single_agent_revision','independent_schema_reviewer'):
        responses = [r['arms'][arm] for r in result['cases']]
        summary['role_comparison'][arm] = {'n':len(responses), **{k:sum(bool(r['score'][k]) for r in responses)
            for k in ('parsed','valid','publishable','label_agreement','id_agreement')},
            'api_errors':sum('error_type' in r for r in responses),
            'revision_tokens':sum((r.get('usage') or {}).get('total_tokens',0) for r in responses)}
    pairs = [(r['arms']['single_agent_revision']['score']['publishable'],
              r['arms']['independent_schema_reviewer']['score']['publishable']) for r in result['cases']]
    summary['paired_publishability'] = {'reviewer_only':sum(b and not a for a,b in pairs),
                                       'single_only':sum(a and not b for a,b in pairs),
                                       'both':sum(a and b for a,b in pairs),
                                       'neither':sum(not a and not b for a,b in pairs)}
    summary['actual_api_calls'] = sum(1+len(r['arms']) for r in result['cases'])
    summary['total_api_tokens'] = sum((x.get('usage') or {}).get('total_tokens',0)
        for c in result['cases'] for x in [c['initial'], *c['arms'].values()])
    save(out/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__': main()
