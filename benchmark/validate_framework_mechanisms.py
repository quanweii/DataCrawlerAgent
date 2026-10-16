"""Validate saved paired traces, inputs, costs, and injected-fault outcomes offline."""
import hashlib
import json
from pathlib import Path
import sys
root = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from domains import load_domain_yaml, yaml_to_task_config
out = root/'benchmark/results/framework_mechanisms_20260915'
r = json.loads((out/'results.json').read_text())
s = json.loads((out/'summary.json').read_text())
checks=[]
def check(name, condition):
    checks.append({'name':name,'pass':bool(condition)})
check('12 distinct domain-seed cases',len({(c['domain'],c['seed']) for c in r['cases']})==12)
check('84 injected/control variant rows',len(r['faults'])==84)
for domain,source in r['sources'].items():
    task=yaml_to_task_config(load_domain_yaml(str(root/'domains'/domain/'domain.yaml')))
    check(domain+' source unchanged',hashlib.sha256((root/task['data_file']).read_bytes()).hexdigest()==source['data_sha256'])
for c in r['cases']:
    prefix=f"{c['domain']}/{c['seed']}"
    check(prefix+' saved trace equals aggregate',json.loads((out/f"{c['domain']}_seed{c['seed']}.json").read_text())==c)
    for name,a in c['arms'].items():
        check(prefix+'/'+name+' received initial proposal',c['initial']['raw'] in '\n'.join(m['content'] for m in a['messages']))
        check(prefix+'/'+name+' no gold feedback',not any('label_agreement' in m['content'] or 'id_agreement' in m['content'] for m in a['messages']))
        check(prefix+'/'+name+' complete response',not a.get('error_type') and a.get('finish_reason')=='stop')
for variant,summary in s['fault_injection'].items():
    rows=[x for x in r['faults'] if x['variant']==variant]
    check(variant+' detection recount',sum(bool(x['fault_prevented']) for x in rows)==summary['prevented'])
    check(variant+' expected injections',sum(x['injected_fault'] for x in rows)==20)
    check(variant+' controls',sum(not x['injected_fault'] for x in rows)==8)
    check(variant+' rejection recount',sum(bool(x['false_rejection']) for x in rows)==summary['false_rejections'])
calls=[a for c in r['cases'] for a in [c['initial'],*c['arms'].values()]]
check('36 actual calls',len(calls)==s['actual_api_calls']==36)
check('token accounting',sum(a['usage']['total_tokens'] for a in calls)==s['total_api_tokens'])
for name,summary in s['role_comparison'].items():
    for key in ('parsed','valid','publishable','label_agreement','id_agreement'):
        check(name+'/'+key,sum(c['arms'][name]['score'][key] for c in r['cases'])==summary[key])
result={'passed':sum(x['pass'] for x in checks),'total':len(checks),'checks':checks}
(out/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':result['passed'],'total':len(checks),'failed':[x['name'] for x in checks if not x['pass']]}))
print(json.dumps({'label_choices':[(c['domain'],c['seed'], {n:a['proposal'].get('label_field') for n,a in c['arms'].items()}) for c in r['cases']], 'models':sorted({a['model_returned'] for a in calls})}))
sys.exit(0 if all(x['pass'] for x in checks) else 1)
