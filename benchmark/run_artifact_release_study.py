"""Release-boundary fault campaign on immutable real-graph subsets.

Three policies: train-label homophily selection; unchecked export; NEW audited
export. This is not an end-to-end or human-participant comparison.
All outputs go to a fresh study directory; canonical data remain read-only.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark.data.domain_loader import load_domain_graph, build_features, DOMAIN_GRAPH
from benchmark.data.protocols import load_protocol_metadata, temporal_split, group_inductive_split, protocol_excluded_relations
from benchmark.data.splits import stratified_split, make_split_disjoint_view
from benchmark.artifact_release import publish_bundle, digest_tensor
from benchmark.models.gnn import GraphSAGEModel
from benchmark.run_four_protocol_graph_tasks import prepare_splits, binary_metrics, scores

FAULTS = ['target_relation', 'target_feature', 'duplicate_entity', 'missing_time_zero',
          'group_overlap', 'self_loop', 'dangling_endpoint', 'non_stratified',
          'test_label_selection', 'feature_version']
CONTROLS = ['clean', 'renamed_relation', 'reordered_edges']


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False))


def reedge(b):
    if b['protocol'] == 'transductive':
        return
    e = b['edges']
    sid = torch.full((len(b['ids']),), -1)
    for i, m in enumerate(b['masks']):
        sid[m] = i
    keep = (sid[e[0]] == sid[e[1]]) & (sid[e[0]] >= 0)
    b['edges'] = e[:, keep]
    b['edge_types'] = [r for r, k in zip(b['edge_types'], keep.tolist()) if k]


def set_masks(b, masks):
    b['masks'] = tuple(m.clone() for m in masks)
    b['selection_nodes'] = masks[1].nonzero().flatten().tolist()
    reedge(b)


def base_bundle(domain, cap):
    g = load_domain_graph(domain, exclude_relations=protocol_excluded_relations(domain))
    y = g['y']
    gen = torch.Generator().manual_seed(20260918)
    chosen = []
    for c in sorted(y.unique().tolist()):
        ids = y.eq(c).nonzero().flatten()
        chosen += ids[torch.randperm(len(ids), generator=gen)[:max(12, cap//len(y.unique()))]].tolist()
    chosen = sorted(chosen)
    ids = [g['node_ids'][i] for i in chosen]
    yi = y[chosen]
    k = int(y.max()) + 1
    x = build_features(domain, g, 'text+image')[chosen].float()
    # Reserved zero block isolates explicit target-field serialization at fixed dimension.
    x = torch.cat([x, torch.zeros((len(ids), k))], 1)
    raw_path = ROOT / DOMAIN_GRAPH[domain]
    raw = json.loads(raw_path.read_text())
    idx = {v: i for i, v in enumerate(ids)}
    edges, types, seen = [], [], set()
    excluded = set(protocol_excluded_relations(domain))
    for e in raw['edges']:
        r = e.get('attributes', {}).get('relation', '?')
        u, v = idx.get(e['source']), idx.get(e['target'])
        if r in excluded or u is None or v is None or u == v:
            continue
        for a, z in ((u,v),(v,u)):
            if (a,z,r) not in seen:
                seen.add((a,z,r)); edges.append((a,z)); types.append(r)
    meta = load_protocol_metadata(domain, ids)
    b = {'domain':domain, 'ids':ids, 'feature_ids':ids.copy(), 'x':x, 'y':yi,
         'edges':torch.tensor(edges, dtype=torch.long).t().contiguous(), 'edge_types':types,
         'relation_lineage':{r:{'target_derived':False} for r in set(types)},
         'feature_lineage':[], 'protocol':'transductive'}
    contract = {'times':meta['times'], 'groups':meta['groups'], 'classes':k,
                'feature_sha256':digest_tensor(x), 'stratified_required':True, 'split_seed':1}
    provenance = {'graph':str(raw_path.relative_to(ROOT)),
                  'graph_sha256':hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                  'nodes_full':len(y), 'nodes_subset':len(ids), 'subset_indices':chosen,
                  'edges_directed_subset':len(types), 'feature_sha256':digest_tensor(x)}
    for m in ('text','image'):
        p=ROOT/'benchmark/processed'/domain/f'{m}_embeddings.pt'
        provenance[m+'_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    return b, contract, provenance


def case_pair(base, contract, name, seed):
    clean, c = copy.deepcopy(base), copy.deepcopy(contract)
    c['split_seed'] = seed + 1
    set_masks(clean, stratified_split(clean['y'], seed=seed+1))
    rng = np.random.RandomState(seed+20260918)
    n = len(clean['ids'])
    if name == 'missing_time_zero':
        clean['protocol']='temporal'
        # Explicitly missing metadata, including naturally missing rows.
        hidden = rng.choice(n, max(2,n//20), replace=False).tolist()
        for i in hidden: c['times'][i]=None
        set_masks(clean, temporal_split(c['times'])[0])
    elif name == 'group_overlap':
        clean['protocol']='group_inductive'
        set_masks(clean, group_inductive_split(c['groups'], clean['y'], seed=seed+1)[0])
    b=copy.deepcopy(clean)
    witness={}
    if name=='target_relation':
        added=[]
        for cls in b['y'].unique().tolist():
            ii=b['y'].eq(cls).nonzero().flatten().tolist()
            for j,u in enumerate(ii):
                for step in (1,2):
                    v=ii[(j+step)%len(ii)]
                    if u!=v: added.extend([(u,v),(v,u)])
        b['edges']=torch.cat([b['edges'],torch.tensor(added).t()],1)
        b['edge_types']+=['injected_target_relation']*len(added)
        b['relation_lineage']['injected_target_relation']={'target_derived':True}
    elif name=='target_feature':
        # Explicit target-field feature, NOT a claim about CLIP re-encoding text.
        b['x'][:,-c['classes']:]=F.one_hot(b['y'],c['classes']).float()
        b['feature_lineage']=[{'name':'serialized_target_field','target_derived':True}]
    elif name=='duplicate_entity':
        tr,va,te=b['masks']; u=int(tr.nonzero()[0]); v=int(te.nonzero()[0])
        b['ids'][v]=b['ids'][u]; b['feature_ids'][v]=b['feature_ids'][u]
        b['x'][v]=b['x'][u]; b['y'][v]=b['y'][u]
        witness={'indices':[u,v]}
    elif name=='missing_time_zero':
        wrong=[0.0 if t is None else t for t in c['times']]
        b['edges']=base['edges'].clone(); b['edge_types']=base['edge_types'].copy()
        set_masks(b,temporal_split(wrong)[0])
    elif name=='group_overlap':
        by={}
        for i,g in enumerate(c['groups']): by.setdefault(g,[]).append(i)
        candidates=[v for v in by.values() if len(v)>=2]
        if not candidates: raise RuntimeError('No repeated group in subset')
        members=candidates[rng.randint(len(candidates))]; u,v=members[:2]
        s=next(i for i,m in enumerate(b['masks']) if m[u])
        for m in b['masks']:m[v]=False
        b['masks'][(s+1)%3][v]=True
        b['selection_nodes']=b['masks'][1].nonzero().flatten().tolist()
        b['edges']=base['edges'].clone();b['edge_types']=base['edge_types'].copy();reedge(b)
        witness={'indices':[u,v]}
    elif name in ('self_loop','dangling_endpoint'):
        u=int(rng.randint(n));v=u if name=='self_loop' else n+seed+5
        r=b['edge_types'][0]
        b['edges']=torch.cat([b['edges'],torch.tensor([[u],[v]])],1);b['edge_types'].append(r)
        witness={'edge':[u,v]}
    elif name=='non_stratified':
        # Label-sorted contiguous split falsely declared stratified, not arbitrary random splitting.
        idx=torch.argsort(b['y']);masks=[]
        for v in (idx[:int(.6*n)],idx[int(.6*n):int(.8*n)],idx[int(.8*n):]):
            m=torch.zeros(n,dtype=torch.bool);m[v]=True;masks.append(m)
        set_masks(b,tuple(masks))
    elif name=='test_label_selection':
        b['selection_nodes']=b['masks'][2].nonzero().flatten().tolist()
    elif name=='feature_version':
        perm=torch.randperm(n,generator=torch.Generator().manual_seed(seed+50))
        b['x']=b['x'][perm];b['feature_ids']=[b['feature_ids'][i] for i in perm]
    elif name=='renamed_relation':
        mapping={r:'renamed_'+r for r in b['relation_lineage']}
        b['edge_types']=[mapping[r] for r in b['edge_types']]
        b['relation_lineage']={mapping[r]:v for r,v in b['relation_lineage'].items()}
    elif name=='reordered_edges':
        p=torch.arange(b['edges'].size(1)-1,-1,-1)
        b['edges']=b['edges'][:,p];b['edge_types']=[b['edge_types'][i] for i in p]
    return clean,b,c,witness


def choose_homophily(b):
    b=copy.deepcopy(b); tr=b['masks'][0]; e=b['edges']; n=len(b['ids'])
    stats={}
    for j,r in enumerate(b['edge_types']):
        u,v=e[:,j].tolist()
        if 0<=u<n and 0<=v<n and tr[u] and tr[v]:
            a,z=stats.get(r,(0,0));stats[r]=(a+int(b['y'][u]==b['y'][v]),z+1)
    if stats:
        chosen=sorted(stats,key=lambda r:(-stats[r][0]/stats[r][1],r))[0]
        keep=torch.tensor([r==chosen for r in b['edge_types']])
        b['edges']=e[:,keep];b['edge_types']=[chosen]*int(keep.sum())
    return b


def residue(name,b,clean,c,w):
    # Independent post-deserialization witnesses; does NOT call audit_bundle.
    n=len(b['ids'])
    if name=='target_relation':return 'injected_target_relation' in b['edge_types']
    if name=='target_feature':return bool(b['x'][:,-c['classes']:].abs().sum()>0)
    if name=='duplicate_entity':return len(set(b['ids']))<n
    if name=='missing_time_zero':
        return any(t is None and any(bool(m[i]) for m in b['masks']) for i,t in enumerate(c['times']))
    if name=='group_overlap':
        a,z=w['indices'];return any(bool(m[a])!=bool(m[z]) for m in b['masks'])
    if name in ('self_loop','dangling_endpoint'):
        u,v=w['edge'];return bool(((b['edges'][0]==u)&(b['edges'][1]==v)).any())
    if name=='non_stratified':
        return any(not torch.equal(torch.bincount(b['y'][a],minlength=c['classes']),
                                   torch.bincount(clean['y'][z],minlength=c['classes']))
                   for a,z in zip(b['masks'],clean['masks']))
    if name=='test_label_selection':return bool(set(b['selection_nodes'])&set(b['masks'][2].nonzero().flatten().tolist()))
    if name=='feature_version':return not torch.equal(b['x'],clean['x'])
    return False


def evaluate(b,seed,epochs,device):
    n=len(b['ids']);e=b['edges'];x=b['x'].to(device);y=b['y'].to(device)
    if e.numel() and (e.min()<0 or e.max()>=n):return {'nc':None,'lp':None,'status':'invalid_endpoint_no_training'}
    masks=[m.to(device) for m in b['masks']];select=torch.tensor(b['selection_nodes'],device=device)
    if any(int(m.sum())==0 for m in masks) or not len(select):return {'nc':None,'lp':None,'status':'empty_cohort'}
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    model=GraphSAGEModel(x.size(1),64,int(y.max())+1,2,.2).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=.005,weight_decay=5e-4)
    edge=e.to(device);history=[];best=-1;nc=None
    for epoch in range(epochs):
        model.train();opt.zero_grad();out=model(x,edge)
        loss=F.cross_entropy(out[masks[0]],y[masks[0]]);loss.backward();opt.step()
        model.eval()
        with torch.no_grad():
            pred=model(x,edge).argmax(1)
            score=float((pred[select]==y[select]).float().mean());test=float((pred[masks[2]]==y[masks[2]]).float().mean())
        history.append([score,test])
        if score>best:best=score;nc=test
    result={'nc':nc,'nc_curve':history,'nc_selected_by':'test' if set(b['selection_nodes'])&set(b['masks'][2].nonzero().flatten().tolist()) else 'validation','lp':None,'status':'ok','nc_trainings':1,'lp_trainings':0}
    try:
        pos,neg,msg=prepare_splits(b['protocol'],e,b['masks'],n,seed+1)
        pos=[v.to(device) for v in pos];neg=[v.to(device) for v in neg];msg=msg.to(device)
        torch.manual_seed(seed)
        model=GraphSAGEModel(x.size(1),64,32,2,.2).to(device)
        opt=torch.optim.Adam(model.parameters(),lr=.005,weight_decay=5e-4)
        best=-1;curve=[]
        selection=2 if result['nc_selected_by']=='test' else 1
        for epoch in range(epochs):
            model.train();opt.zero_grad();z=model(x,msg)
            s=torch.cat([scores(z,pos[0]),scores(z,neg[0])]);target=torch.cat([torch.ones(pos[0].size(1),device=device),torch.zeros(neg[0].size(1),device=device)])
            F.binary_cross_entropy_with_logits(s,target).backward();opt.step();model.eval()
            with torch.no_grad():
                z=model(x,msg)
                val=binary_metrics(scores(z,pos[selection]).cpu().numpy(),scores(z,neg[selection]).cpu().numpy())
                test=binary_metrics(scores(z,pos[2]).cpu().numpy(),scores(z,neg[2]).cpu().numpy())
            curve.append([val['ap'],test['auc']])
            if val['ap']>best:best=val['ap'];result['lp']=test['auc']
        result['lp_curve']=curve;result['lp_trainings']=1
    except ValueError as exc:result['lp_status']=str(exc)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--domain',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--epochs',type=int,default=60);ap.add_argument('--cap',type=int,default=2048);ap.add_argument('--seeds',type=int,default=3)
    ap.add_argument('--device',default='cuda:0');args=ap.parse_args()
    torch.set_num_threads(2)
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    result_path=out/f'{args.domain}_results.json'
    if result_path.exists():raise FileExistsError('Use fresh output directory')
    base,c,prov=base_bundle(args.domain,args.cap)
    save(out/f'{args.domain}_source.json',prov)
    rows=[];impact=[];cache={}
    for seed in range(args.seeds):
        for name in CONTROLS+FAULTS:
            clean,b,contract,w=case_pair(base,c,name,seed)
            assert name in CONTROLS or residue(name,b,clean,contract,w),name
            folder=out/args.domain/f'seed{seed}'/name
            folder.mkdir(parents=True,exist_ok=True)
            torch.save(clean,folder/'clean_comparator.pt');torch.save(b,folder/'candidate.pt')
            save(folder/'contract.json',contract);save(folder/'witness.json',w)
            for arm in ('homophily','unchecked','audited_release'):
                cand=choose_homophily(b) if arm=='homophily' else b
                target=folder/f'{arm}.pt'
                receipt=publish_bundle(cand,contract,target,audited=arm=='audited_release')
                reread=torch.load(target,weights_only=False) if target.exists() else None
                escaped=False if reread is None else residue(name,reread,clean,contract,w)
                rows.append({'domain':args.domain,'seed':seed,'fault':name,'arm':arm,'control':name in CONTROLS,
                             'published':receipt['published'],'escaped':escaped,'errors':receipt['errors'],
                             'output_sha256':receipt.get('sha256')})
            if name in FAULTS:
                # Paired fault effect on unchecked exports; no fake scores for rejected releases.
                key=(seed,clean['protocol'],tuple(contract['times']) if name=='missing_time_zero' else ())
                if key not in cache:
                    cache[key]=evaluate(clean,seed,args.epochs,args.device)
                good=cache[key];bad=evaluate(torch.load(folder/'unchecked.pt',weights_only=False),seed,args.epochs,args.device)
                impact.append({'domain':args.domain,'seed':seed,'fault':name,'protocol':b['protocol'],
                               'clean':good,'faulty':bad,
                               'nc_delta_pp':None if good['nc'] is None or bad['nc'] is None else 100*(bad['nc']-good['nc']),
                               'lp_delta_pp':None if good['lp'] is None or bad['lp'] is None else 100*(bad['lp']-good['lp'])})
            save(result_path,{'source':prov,'settings':vars(args),'rows':rows,'impact':impact,
                             'completed':False,'distinct_clean_fits':len(cache)})
            print(args.domain,seed,name,[(r['arm'],r['published'],r['escaped']) for r in rows[-3:]],flush=True)
    save(result_path,{'source':prov,'settings':vars(args),'rows':rows,'impact':impact,
                     'completed':True,'distinct_clean_fits':len(cache),
                     'claim_boundary':'New opt-in audited publisher; simulated policies; constructed faults; subset stress tests; not human superiority or legacy 0-defect claim.'})


if __name__=='__main__':main()
