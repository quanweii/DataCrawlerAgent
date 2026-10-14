#!/usr/bin/env python3
"""Validation-only selection manifest, then test reporting, for the replayed NC grid.

No score on the test set participates in selecting either architecture or
single-modality baseline. Ties follow the fixed model/modality order below.
"""
import argparse, copy, hashlib, json, math, statistics
from pathlib import Path

DOMAINS = ['arxiv', 'steam', 'tmdb', 'wikiart']
PROTOCOLS = ['transductive', 'split_disjoint', 'temporal', 'group_inductive']
MODELS = ['mlp', 'gcn', 'gat', 'graphsage', 'gin', 'gcnii']
MODALITIES = {'arxiv':['text','image','formula','text+image','text+formula','image+formula','text+image+formula'],
             'steam':['text','image','text+image'], 'tmdb':['text','image','text+image'],
             'wikiart':['text','image','text+image']}

def choose(models, candidates):
    scores = {m: statistics.mean(models[m]['val_accs']) for m in candidates}
    if not all(math.isfinite(v) for v in scores.values()):
        raise ValueError('Missing or non-finite validation scores; never fall back to test scores')
    return max(candidates, key=lambda m: scores[m])

def selection_manifest(domains):
    chosen, fused = {}, []
    for d in DOMAINS:
        matrix = domains[d]
        for p in PROTOCOLS:
            for mod in MODALITIES[d]:
                cell = matrix['results'][p+'/'+mod]
                selected = choose(cell['models'], MODELS)
                chosen[d+'/'+p+'/'+mod] = dict(selected_model=selected,
                    selection_val_acc=statistics.mean(cell['models'][selected]['val_accs']))
            full = MODALITIES[d][-1]
            singles = [m for m in MODALITIES[d] if '+' not in m]
            single = max(singles, key=lambda m:chosen[d+'/'+p+'/'+m]['selection_val_acc'])
            gnn = choose(matrix['results'][p+'/'+full]['models'], MODELS[1:])
            fused.append(dict(domain=d, protocol=p, full_modality=full,
                selected_model=chosen[d+'/'+p+'/'+full]['selected_model'],
                selected_gnn=gnn, selected_single_modality=single,
                selected_single_model=chosen[d+'/'+p+'/'+single]['selected_model']))
    return dict(policy='mean validation accuracy across seeds 0-4',
                ties='first in fixed model and modality orders',
                model_order=MODELS, modality_order=MODALITIES, cells=chosen, fused=fused)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-dir',default='benchmark/results/validation_selected_nc_20261014')
    args=ap.parse_args();base=Path(args.input_dir)
    domains={};checks=[];changes=[]
    def check(ok, message):
        checks.append(message)
        if not ok: raise AssertionError(message)
    for d in DOMAINS:
        obj=dict(domain=d, protocols=PROTOCOLS, modalities=MODALITIES[d], models=MODELS,
                 split_seed=1, init_seeds=5, results={})
        for mod in MODALITIES[d]:
            for p in PROTOCOLS:
                path=base/f'nc_{d}_{p}_{mod.replace("+","_")}.json'
                c=json.loads(path.read_text())
                check(c.get('complete') is True,f'{d}/{p}/{mod} complete')
                check(c['partition_matches_saved'],f'{d}/{p}/{mod} saved partition')
                check(list(c['models'])==MODELS,f'{d}/{p}/{mod} fixed order')
                for m in MODELS:
                    r=c['models'][m]
                    check([x['seed'] for x in r['runs']]==list(range(5)),f'{d}/{p}/{mod}/{m} seeds')
                    for metric,raw in [('acc','test_acc'),('f1','test_f1'),('val_acc','val_acc')]:
                        values=[x[raw] for x in r['runs']]
                        check(all(math.isfinite(v) and 0<=v<=1 for v in values),f'{d}/{p}/{mod}/{m}/{metric} finite')
                        check(all(abs(a-b)<1e-12 for a,b in zip(values,r[metric+'s'])),f'{d}/{p}/{mod}/{m}/{metric} values')
                        check(abs(statistics.mean(values)-r[metric])<1e-12,f'{d}/{p}/{mod}/{m}/{metric} mean')
                        check(abs(statistics.pstdev(values)-r[metric+'_std'])<1e-12,f'{d}/{p}/{mod}/{m}/{metric} std')
                obj['results'][p+'/'+mod]=c
        domains[d]=obj
        (base/f'nc_{d}_protocol_matrix.json').write_text(json.dumps(obj,indent=2))
    manifest=selection_manifest(domains)
    # Metamorphic fairness check: deliberately reverse/replace every test
    # outcome and verify that the selected architectures/modalities do not move.
    mutated=copy.deepcopy(domains)
    for matrix in mutated.values():
        for cell in matrix['results'].values():
            for i,m in enumerate(MODELS):
                r=cell['models'][m]
                r.update(acc=i/10,f1=1-i/10,accs=[i/10]*5,f1s=[1-i/10]*5)
                for run in r['runs']:run.update(test_acc=i/10,test_f1=1-i/10)
    check(selection_manifest(mutated)==manifest,'Selection invariant to all changed test outcomes')
    ties={m:dict(val_accs=[.5]*5,acc=i/10) for i,m in enumerate(MODELS)}
    check(choose(ties,MODELS)=='mlp','Validation ties use fixed order without test tie-break')
    (base/'validation_selection_manifest.json').write_text(json.dumps(manifest,indent=2))
    # Reporting pass reads the already written selection manifest.
    manifest=json.loads((base/'validation_selection_manifest.json').read_text())
    rows=[];fusion=[]
    for selected in manifest['fused']:
        d,p=selected['domain'],selected['protocol'];matrix=domains[d]
        full=matrix['results'][p+'/'+selected['full_modality']]
        name=selected['selected_model'];r=full['models'][name]
        gnn=full['models'][selected['selected_gnn']]
        single=matrix['results'][p+'/'+selected['selected_single_modality']]['models'][selected['selected_single_model']]
        diff=[100*(a-b) for a,b in zip(r['accs'],single['accs'])]
        rows.append(dict(**selected, selection_val_acc=r['val_acc'],acc=r['acc'],acc_std=r['acc_std'],
            f1=r['f1'],f1_std=r['f1_std'],selected_gnn_minus_mlp=gnn['acc']-full['models']['mlp']['acc']))
        fusion.append(dict(**selected,full_val_acc=r['val_acc'],single_val_acc=single['val_acc'],
            full_acc=r['acc'],single_acc=single['acc'],paired_gain_pp=diff,
            gain_mean_pp=statistics.mean(diff),gain_std_pp=statistics.pstdev(diff)))
    result=dict(schema_version='validation-selected-NC-v1',selection=manifest['policy'],
                training_runs=sum(len(c['models'][m]['runs']) for x in domains.values() for c in x['results'].values() for m in MODELS),
                rows=rows,fusion=fusion,domains=domains)
    (base/'validation_selected_summary.json').write_text(json.dumps(result,indent=2))
    audit=dict(status='PASS',checks=len(checks),training_runs=result['training_runs'],
               cells=64, model_cells=384,manifest_sha256=hashlib.sha256((base/'validation_selection_manifest.json').read_bytes()).hexdigest())
    (base/'validation_selection_audit.json').write_text(json.dumps(audit,indent=2))
    lines=['# Validation-selected primary NC results','',
           'Architecture and single-modality baseline: mean validation accuracy over seeds 0-4. Ties use fixed order. All outcomes below are test results.', '',
           '| Domain | View | Selected model | Validation acc | Test acc mean ± std | Test macro-F1 | Selected GNN−MLP |',
           '|---|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["domain"]} | {r["protocol"]} | {r["selected_model"]} | {r["selection_val_acc"]:.4f} | {r["acc"]:.4f} ± {r["acc_std"]:.4f} | {r["f1"]:.4f} | {r["selected_gnn_minus_mlp"]:+.4f} |')
    lines += ['','## Validation-selected modality comparison','',
              '| Domain | View | Full model | Single modality | Single model | Full−single paired mean ± std (pp) |',
              '|---|---|---|---|---|---:|']
    for r in fusion:
        lines.append(f'| {r["domain"]} | {r["protocol"]} | {r["selected_model"]} | {r["selected_single_modality"]} | {r["selected_single_model"]} | {r["gain_mean_pp"]:+.2f} ± {r["gain_std_pp"]:.2f} |')
    lines += ['','## Complete fused-model test results','',
              '| Domain | View | Model | Validation acc | Test acc ± std | Test macro-F1 ± std |',
              '|---|---|---|---:|---:|---:|']
    for d in DOMAINS:
        for p in PROTOCOLS:
            c=domains[d]['results'][p+'/'+MODALITIES[d][-1]]
            for m in MODELS:
                r=c['models'][m]
                lines.append(f'| {d} | {p} | {m} | {r["val_acc"]:.4f} | {r["acc"]:.4f} ± {r["acc_std"]:.4f} | {r["f1"]:.4f} ± {r["f1_std"]:.4f} |')
    (base/'validation_selected_summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
