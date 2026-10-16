"""Readback validation and descriptive aggregation, separate from publisher."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmark.run_artifact_release_study import FAULTS,CONTROLS,residue


def main():
    base=ROOT/'benchmark/results/artifact_release_20260918'
    rows=[];impact=[];checks=0;errors=[];training={'nc':0,'lp':0}
    def check(ok,desc):
        nonlocal checks
        checks+=1
        if not ok:errors.append(desc)
    for domain in ('arxiv','steam','tmdb','wikiart'):
        result=json.loads((base/f'{domain}_results.json').read_text())
        check(result['completed'],domain+' completion')
        check(len(result['rows'])==117,domain+' 3 seeds x 13 cases x 3 policies')
        check(len(result['impact'])==30,domain+' impacts')
        prov=result['source']
        check(hashlib.sha256((ROOT/prov['graph']).read_bytes()).hexdigest()==prov['graph_sha256'],domain+' immutable graph')
        for m in ('text','image'):
            f=ROOT/'benchmark/processed'/domain/f'{m}_embeddings.pt'
            check(hashlib.sha256(f.read_bytes()).hexdigest()==prov[m+'_sha256'],domain+' immutable '+m)
        for row in result['rows']:
            folder=base/domain/f"seed{row['seed']}"/row['fault']
            path=folder/f"{row['arm']}.pt"
            check(path.exists()==row['published'],str(path)+' release existence')
            c=json.loads((folder/'contract.json').read_text());w=json.loads((folder/'witness.json').read_text())
            clean=torch.load(folder/'clean_comparator.pt',weights_only=False)
            if path.exists():
                check(hashlib.sha256(path.read_bytes()).hexdigest()==row['output_sha256'],str(path)+' checksum')
                reread=torch.load(path,weights_only=False)
                check(residue(row['fault'],reread,clean,c,w)==row['escaped'],str(path)+' independent residue')
            else:check(not row['escaped'],str(path)+' no phantom artifact')
            rows.append(row)
        for r in result['impact']:
            for name in ('clean','faulty'):
                v=r[name]
                if v['nc'] is not None:
                    a=np.asarray(v['nc_curve']); idx=int(a[:,0].argmax())
                    check(abs(v['nc']-a[idx,1])<1e-10,f'{domain}/{r["fault"]}/{name} NC selection replay')
                if v['lp'] is not None:
                    a=np.asarray(v['lp_curve']);idx=int(a[:,0].argmax())
                    check(abs(v['lp']-a[idx,1])<1e-10,f'{domain}/{r["fault"]}/{name} LP selection replay')
            for metric in ('nc','lp'):
                a,b=r['clean'][metric],r['faulty'][metric]
                if a is not None and b is not None:
                    check(abs(r[metric+'_delta_pp']-100*(b-a))<1e-8,'paired delta')
                training[metric]+=r['faulty'].get(metric+'_trainings',0)
                if r['fault'] in ('target_relation','missing_time_zero','group_overlap'):
                    training[metric]+=r['clean'].get(metric+'_trainings',0)
            impact.append(r)
    matrix={}
    for arm in ('homophily','unchecked','audited_release'):
        ar=[r for r in rows if r['arm']==arm]
        matrix[arm]={'by_fault':{f:{'escaped':sum(r['escaped'] for r in ar if r['fault']==f),
            'published':sum(r['published'] for r in ar if r['fault']==f),'n':sum(r['fault']==f for r in ar)} for f in FAULTS},
            'controls_published':sum(r['published'] for r in ar if r['control']),
            'controls':sum(r['control'] for r in ar),'faulty_published':sum(r['published'] for r in ar if not r['control'])}
    effect={}
    for f in FAULTS:
        effect[f]={}
        for d in ('arxiv','steam','tmdb','wikiart','pooled'):
            rr=[r for r in impact if r['fault']==f and (d=='pooled' or r['domain']==d)]
            effect[f][d]={}
            for metric in ('nc','lp'):
                vals=[r[metric+'_delta_pp'] for r in rr if r[metric+'_delta_pp'] is not None]
                effect[f][d][metric]={'n':len(vals),'mean_pp':float(np.mean(vals)) if vals else None,
                    'std_pp':float(np.std(vals)) if vals else None,
                    'min_pp':min(vals) if vals else None,'max_pp':max(vals) if vals else None}
    summary={'validation':{'checks':checks,'errors':errors,'passed':not errors},'matrix':matrix,
             'effects':effect,'new_stress_test_trainings':training,'old_benchmark_trainings_unchanged':2520,
             'scope':'New audited release extension; not historical full framework or human superiority. Outputs rejected, not repaired.'}
    (base/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Serialized artifact release study','',summary['scope'],'',
           f'Validation: {checks-len(errors)}/{checks}. New stress-test trainings: {training}.','',
           '| Policy | '+' | '.join(FAULTS)+' | Clean acceptance |',
           '|---|'+'---:|'*(len(FAULTS)+1)]
    for a,v in matrix.items():lines.append('| '+a+' | '+' | '.join(f"{v['by_fault'][f]['escaped']}/{v['by_fault'][f]['n']}" for f in FAULTS)+f" | {v['controls_published']}/{v['controls']} |")
    lines+=['','## Paired unchecked-minus-clean score differences (percentage points)',
            'Descriptive pooled means, not all positive; cohort-changing faults are not isolated inflation estimates.','',
            '| Fault | NC mean | NC range | LP mean | LP range |','|---|---:|---|---:|---|']
    for f in FAULTS:
        v=effect[f]['pooled']; fields=[]
        for m in ('nc','lp'):
            q=v[m];fields.extend(['N/A','N/A'] if not q['n'] else [f"{q['mean_pp']:+.3f} (n={q['n']})",f"[{q['min_pp']:+.3f}, {q['max_pp']:+.3f}]"])
        lines.append('| '+f+' | '+' | '.join(fields)+' |')
    lines+=['','Source hashes, individual curves, receipts, candidates, contracts and published bundles remain in this directory.']
    (base/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
