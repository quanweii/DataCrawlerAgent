#!/usr/bin/env python3
"""A new-source case using production agents, CLIP, protocols and six tasks.

The source adapter and YAML are explicit setup; no graph/task algorithms are
reimplemented here. Old four-domain data and results are left intact.
"""
import os,argparse,json,time,hashlib,sys,statistics,csv
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,default=2);ap.add_argument('--phase',choices=['construct','encode','evaluate','all'],default='all')
ap.add_argument('--output-dir',default='benchmark/results/cleveland_e2e_20261014',help='Use a fresh directory for a separate replay')
args=ap.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
OUT=ROOT/args.output_dir;OUT.mkdir(exist_ok=True,parents=True)
if (OUT/'case_summary.json').exists():
    raise FileExistsError('Completed case artifacts are immutable; use a fresh --output-dir for a new execution')
from dotenv import load_dotenv
load_dotenv(ROOT/'.env')
import torch,numpy as np
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from benchmark.data import domain_loader as loader
loader.DOMAIN_GRAPH['cleveland']='data/cleveland_datasets/cleveland_multimodal_graph_agent.json'
loader.DOMAIN_RAW['cleveland']='data/cleveland_cslo';loader.DOMAIN_MODALITIES['cleveland']=['text','image']
from domains import load_domain_yaml,yaml_to_task_config
from benchmark.data.protocols import generate_protocol_views,save_protocol_suite,load_protocol_metadata
YAML=ROOT/'domains/cleveland/domain.yaml';yc=load_domain_yaml(str(YAML))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def json_value(o):
    if isinstance(o,torch.Tensor):
        return dict(type='tensor',shape=list(o.shape),dtype=str(o.dtype),sha256=hashlib.sha256(o.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
    if isinstance(o,np.generic):return o.item()
    if isinstance(o,Path):return str(o)
    if isinstance(o,set):return sorted(o)
    raise TypeError(type(o).__name__)
def save(path,obj):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=json_value),encoding='utf-8');tmp.replace(path)
def note(phase,started,**kw):
    q=OUT/'timings.json';o=json.loads(q.read_text()) if q.exists() else {}
    o[phase]=dict(seconds=time.monotonic()-started,**kw);save(q,o)

def construct():
    from orchestrator import load_orchestrator_from_yaml
    assert json.loads((OUT/'researcher_confirmation.json').read_text())['author_confirmed']
    task=yaml_to_task_config(yc);task['domain_config']=str(YAML)
    task['user_request']='Construct a genuinely multimodal Cleveland CC0 benchmark: classify object type from text/images with creator/culture relations; export auditable views and validation-selected model results.'
    task['schema_mode']='confirm';save(OUT/'task.json',task)
    started=time.monotonic();orch=load_orchestrator_from_yaml(str(YAML));calls=[]
    original=orch.client.chat.completions.create
    def tracked(*a,**k):
        t=time.monotonic();resp=original(*a,**k)
        calls.append(dict(model=k.get('model'),input_messages=k.get('messages'),
                          response=resp.choices[0].message.content,seconds=time.monotonic()-t,
                          usage=resp.usage.model_dump() if resp.usage else None))
        save(OUT/'llm_calls.json',calls);return resp
    orch.client.chat.completions.create=tracked
    result=orch.execute_pipeline(task);save(OUT/'construction_result.json',result)
    assert result['status'] in ['success','partial'],result.get('errors')
    graph=ROOT/loader.DOMAIN_GRAPH['cleveland'];assert graph.exists()
    g=json.loads(graph.read_text());assert len(g['nodes'])==900 and len(g['edges'])>0
    # Fixed tools, rather than an LLM success message, verify completion.
    stages=result.get('stages',{})
    assert 'schema_discovery' in stages and stages['schema_discovery'].get('schema_audit',{}).get('valid',
        stages['schema_discovery'].get('results',{}).get('schema_audit',{}).get('valid',False)),str(stages['schema_discovery'])[:500]
    accepted={**task['schema_hint']};save(OUT/'accepted_schema.json',accepted)
    note('construction',started,llm_calls=len(calls),status=result['status'],graph_sha256=sha(graph))
    print('CONSTRUCTION COMPLETE',len(g['nodes']),len(g['edges']),flush=True)

def encode():
    from benchmark.encoders.text_encoder import TextEncoder
    from benchmark.encoders.image_encoder import ImageEncoder
    started=time.monotonic();g=json.loads((ROOT/loader.DOMAIN_GRAPH['cleveland']).read_text());ids=[n['id'] for n in g['nodes']]
    raw=ROOT/loader.DOMAIN_RAW['cleveland'];processed=ROOT/'benchmark/processed/cleveland';processed.mkdir(exist_ok=True,parents=True)
    for mod,Cls in [('text',TextEncoder),('image',ImageEncoder)]:
        path=processed/(mod+'_embeddings.pt')
        if not path.exists():
            enc=Cls(model_name='openai/clip-vit-large-patch14',device='cuda')
            if mod=='text':enc.encode_batch([str(raw/i/(i+'.md')) for i in ids],str(processed),ids=ids)
            else:enc.encode_batch({i:str(raw/i/'images') for i in ids},str(processed))
            del enc;torch.cuda.empty_cache()
        o=torch.load(path,map_location='cpu',weights_only=False)
        assert o['ids']==ids and tuple(o['embeddings'].shape)==(900,768)
        assert torch.isfinite(o['embeddings']).all() and o['embeddings'].norm(dim=1).gt(0).all()
    note('encoding',started,nodes=900,modalities=['text','image'],dimension=768)
    print('ENCODING COMPLETE',flush=True)

def evaluate():
    from torch_geometric.data import Data
    from benchmark.models.gnn import get_model,model_lr
    from benchmark.tasks.node_classification import train_and_eval
    from benchmark.run_four_protocol_graph_tasks import prepare_splits,train_one,mean_std
    from benchmark.run_four_protocol_modality_tasks import load_aligned,retrieval,alignment,train_match,agg
    from benchmark.artifact_release import publish_bundle,digest_tensor
    from types import SimpleNamespace
    started=time.monotonic();g=loader.load_domain_graph('cleveland');node_ids=g['node_ids'];y=g['y']
    pconfig=dict(data_file=yc.data.data_file,id_field='id',time_field='',group_field='creator_ids',group_semantics=yc.protocol.group_semantics)
    suite=generate_protocol_views('cleveland',node_ids,y,g['edge_index'],seed=1,metadata_config=pconfig,protocols=yc.protocol.protocols)
    save_protocol_suite('cleveland',suite,ROOT/'benchmark/results/protocol_views')
    save(OUT/'protocol_manifest.json',suite['manifest'])
    models=['mlp','gcn','gat','graphsage','gin','gcnii'];nc={};selected_nc={};nc_runs=0
    for mod in ['text','image','text+image']:
        ds=loader.SingleDomainDataset('cleveland',modality=mod)
        for protocol,v in suite['views'].items():
            cell={};states={}
            for model in models:
                runs=[];states[model]=[]
                for seed in range(5):
                    torch.manual_seed(seed);m=get_model(model,768,256,6,num_layers=2,dropout=.5,proj_dim=768)
                    data=Data(x=ds.data.x,edge_index=v['edge_index'],y=y);data.num_nodes=900
                    r=train_and_eval(m,data,*v['masks'],num_epochs=300,lr=model_lr(model),device='cuda')
                    runs.append(dict(seed=seed,**r));nc_runs+=1
                    states[model].append({k:t.detach().cpu().clone() for k,t in m.state_dict().items()})
                    print('NC',protocol,mod,model,seed,f'val={r["val_acc"]:.4f} test={r["test_acc"]:.4f}',flush=True)
                cell[model]=dict(runs=runs,val_mean=statistics.mean(r['val_acc'] for r in runs),
                    acc=statistics.mean(r['test_acc'] for r in runs),acc_std=statistics.pstdev(r['test_acc'] for r in runs),
                    f1=statistics.mean(r['test_f1'] for r in runs))
            winner=max(models,key=lambda m:cell[m]['val_mean']);key=protocol+'/'+mod
            nc[key]=cell;selected_nc[key]=dict(model=winner,**{k:v for k,v in cell[winner].items() if k!='runs'})
            save(OUT/'nc_all_models.json',nc);save(OUT/'nc_validation_selection.json',selected_nc)
            if mod=='text+image':
                for seed,state in enumerate(states[winner]):
                    path=OUT/'checkpoints'/f'{protocol}_{winner}_seed{seed}.pt';path.parent.mkdir(exist_ok=True)
                    torch.save(dict(state_dict=state,model=winner,input_dim=768,hidden=256,classes=6,layers=2,dropout=.5,
                        proj_dim=768,seed=seed,protocol=protocol,label_names=g['label_names'],feature_sha256=digest_tensor(ds.data.x),
                        selection='mean validation accuracy; fixed-order ties',validation_acc=cell[winner]['runs'][seed]['val_acc']),path)
                contract=dict(feature_sha256=digest_tensor(ds.data.x),times=[None]*900,
                    groups=load_protocol_metadata('cleveland',node_ids,pconfig)['groups'],
                    stratified_required=True,split_seed=1,classes=6)
                bundle=dict(ids=node_ids,feature_ids=node_ids,x=ds.data.x,y=y,label_names=g['label_names'],edges=v['edge_index'],masks=v['masks'],
                    protocol=protocol,edge_types=['same_creator','shared_culture'],
                    relation_lineage={'same_creator':{'source_fields':['creator_ids'],'target_derived':False},
                                      'shared_culture':{'source_fields':['culture'],'target_derived':False}},
                    feature_lineage=[{'source_fields':['title','description','fun_fact'],'target_derived':False},
                                     {'source_fields':['images.web'],'target_derived':False}],
                    selection_nodes=v['masks'][1].nonzero().flatten().tolist())
                save(OUT/'contracts'/f'{protocol}.json',{k:v for k,v in contract.items() if k!='groups'})
                receipt=publish_bundle(bundle,contract,OUT/'published'/f'{protocol}_benchmark.pt')
                assert receipt['published'];save(OUT/'published'/f'{protocol}.audit.json',receipt)
    ds=loader.SingleDomainDataset('cleveland',modality='text+image');graph={};selected_graph={}
    ga=SimpleNamespace(hidden=128,layers=2,dropout=.5,epochs=200,lr=.001,patience=30,eval_every=5)
    graph_runs=0
    for protocol,v in suite['views'].items():
        pos,neg,msg=prepare_splits(protocol,v['edge_index'],v['masks'],900,1)
        for model in models:
            runs=[train_one(ds.data.x,v['edge_index'],msg,pos,neg,v['masks'],y,model,seed,ga,'cuda') for seed in range(5)]
            graph_runs+=5;graph[protocol+'/'+model]=dict(runs=runs,val_mean=statistics.mean(r['val_ap'] for r in runs),
                **{k:mean_std(runs,k) for k in ['test_auc','test_ap','nmi','ari']})
            print('GRAPH',protocol,model,flush=True);save(OUT/'graph_all_models.json',graph)
        winner=max(models,key=lambda m:graph[protocol+'/'+m]['val_mean'])
        selected_graph[protocol]=dict(model=winner,**{k:v for k,v in graph[protocol+'/'+winner].items() if k!='runs'})
    save(OUT/'graph_validation_selection.json',selected_graph)
    a,fa,_=load_aligned('cleveland','text',node_ids);b,fb,_=load_aligned('cleveland','image',node_ids);assert (fa&fb).all()
    ma=SimpleNamespace(hidden=256,dropout=.3,lr=.001,epochs=50,patience=8,batch_size=1024,sampler_seed=1)
    mods={};matching_runs=0
    for protocol,v in suite['views'].items():
        tr,va,te=v['masks'];runs=[train_match(a,b,y,v['masks'],seed,ma,'cuda') for seed in range(5)];matching_runs+=5
        mods[protocol]=dict(retrieval_ti=retrieval(a[te],b[te],'cuda'),retrieval_it=retrieval(b[te],a[te],'cuda'),
            alignment=alignment(a[te],b[te]),matching=dict(runs=runs,**{k:agg(runs,k) for k in ['auc','ap','accuracy','f1']}),
            cohort=[int(m.sum()) for m in v['masks']])
        print('MODALITY',protocol,flush=True)
    save(OUT/'modality_tasks.json',mods)
    summary=dict(source='Cleveland Museum of Art CC0',nodes=900,classes=6,protocols=list(suite['views']),
        temporal_unavailable='creation-year bounds do not establish observation/collection chronology',
        nc=selected_nc,graph=selected_graph,modality=mods,
        training_runs=dict(nc=nc_runs,lp_clustering=graph_runs,matching=matching_runs,total=nc_runs+graph_runs+matching_runs),
        publication='three audited serialized benchmark bundles and fifteen validation-selected NC checkpoints',
        settings=dict(nc=dict(epochs=300,hidden=256,layers=2,dropout=.5,proj_dim=768),graph=vars(ga),matching=vars(ma)),
        source_hashes={str(q.relative_to(ROOT)):sha(q) for q in [Path(__file__),YAML,ROOT/'tools/cleveland_tools.py',
            ROOT/'benchmark/tasks/node_classification.py',ROOT/'benchmark/run_four_protocol_graph_tasks.py',ROOT/'benchmark/run_four_protocol_modality_tasks.py',
            ROOT/'agents/dataset_formatter.py',ROOT/loader.DOMAIN_GRAPH['cleveland']]})
    save(OUT/'case_summary.json',summary);note('evaluation',started,training_runs=summary['training_runs'])
    print('CASE COMPLETE',json.dumps(summary['training_runs']),flush=True)

if args.phase in ['construct','all']:construct()
if args.phase in ['encode','all']:encode()
if args.phase in ['evaluate','all']:evaluate()
