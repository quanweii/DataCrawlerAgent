"""Independent saved-artifact witnesses, including checkpoint-level predictions."""
from pathlib import Path
import os,json,hashlib,sys,csv,statistics
os.environ['CUDA_VISIBLE_DEVICES']='2';os.environ['OMP_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import torch
from PIL import Image
from sklearn.metrics import accuracy_score,f1_score
from benchmark.data import domain_loader as loader
from benchmark.models.gnn import get_model
from benchmark.artifact_release import digest_tensor
loader.DOMAIN_GRAPH['cleveland']='data/cleveland_datasets/cleveland_multimodal_graph_agent.json'
loader.DOMAIN_RAW['cleveland']='data/cleveland_cslo';loader.DOMAIN_MODALITIES['cleveland']=['text','image']
OUT=ROOT/'benchmark/results/cleveland_e2e_20261014';s=json.loads((OUT/'case_summary.json').read_text());checks=[]
def check(ok,name):
 checks.append(name)
 if not ok:raise AssertionError(name)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
items=json.loads((ROOT/'data/cleveland_raw/cleveland_items.json').read_text());g=json.loads((ROOT/loader.DOMAIN_GRAPH['cleveland']).read_text())
check(len(items)==len(g['nodes'])==900,'real cohort count')
check({a['id'] for a in items}=={n['id'] for n in g['nodes']},'raw-to-graph identities')
image_hashes=set()
for a in items:
 d=ROOT/'data/cleveland_cslo'/a['id'];p=d/'images/primary.jpg';t=d/(a['id']+'.md')
 check(a['license']=='CC0','CC0 '+a['id'])
 im=Image.open(p);im.load();check(min(im.size)>31,'decode '+a['id'])
 check(sha(p)==a['image_sha256'],'image hash '+a['id']);image_hashes.add(sha(p))
 check(sha(t)==a['text_sha256'],'text hash '+a['id'])
 check(not any(l.lower().startswith(('type:','culture:','creator:','department:','tombstone:')) for l in t.read_text().splitlines()),'no structured target/context header '+a['id'])
check(len(image_hashes)==900,'unique exact image fingerprints')
for mod in ['text','image']:
 p=ROOT/f'benchmark/processed/cleveland/{mod}_embeddings.pt';o=torch.load(p,map_location='cpu',weights_only=False)
 check(o['ids']==[n['id'] for n in g['nodes']],'encoder ID order '+mod)
 check(tuple(o['embeddings'].shape)==(900,768) and torch.isfinite(o['embeddings']).all() and o['embeddings'].norm(dim=1).gt(0).all(),'encoder coverage '+mod)
nc=json.loads((OUT/'nc_all_models.json').read_text());selectors=json.loads((OUT/'nc_validation_selection.json').read_text())
models=['mlp','gcn','gat','graphsage','gin','gcnii'];runs=0
for key,c in nc.items():
 check(list(c)==models,'NC fixed search order '+key)
 for m,r in c.items():
  check([x['seed'] for x in r['runs']]==list(range(5)),'NC seeds '+key+'/'+m);runs+=5
  check(abs(statistics.mean(x['val_acc'] for x in r['runs'])-r['val_mean'])<1e-12,'NC validation mean '+key+'/'+m)
 winner=max(models,key=lambda m:c[m]['val_mean']);check(winner==selectors[key]['model'],'NC validation selection '+key)
 check(abs(selectors[key]['acc']-statistics.mean(x['test_acc'] for x in c[winner]['runs']))<1e-12,'NC test report '+key)
check(runs==270,'270 NC executions')
graph=json.loads((OUT/'graph_all_models.json').read_text())
for p in s['protocols']:
 chosen=max(models,key=lambda m:graph[p+'/'+m]['val_mean']);check(chosen==s['graph'][p]['model'],'LP/clustering validation AP selection '+p)
ds=loader.SingleDomainDataset('cleveland',modality='text+image');checkpoint_witnesses=[]
for protocol in s['protocols']:
 b=torch.load(OUT/'published'/f'{protocol}_benchmark.pt',map_location='cpu',weights_only=False)
 check(b['ids']==ds.graph['node_ids'] and digest_tensor(b['x'])==digest_tensor(ds.data.x),'published bundle feature alignment '+protocol)
 masks=b['masks'];check(not any((masks[i]&masks[j]).any() for i in range(3) for j in range(i+1,3)),'bundle disjoint masks '+protocol)
 if protocol!='transductive':
  assignment=torch.full((900,),-1)
  for i,m in enumerate(masks):assignment[m]=i
  check(torch.equal(assignment[b['edges'][0]],assignment[b['edges'][1]]),'bundle split-edge isolation '+protocol)
 for seed in range(5):
  name=s['nc'][protocol+'/text+image']['model'];cp=OUT/'checkpoints'/f'{protocol}_{name}_seed{seed}.pt'
  payload=torch.load(cp,map_location='cpu',weights_only=False)
  model=get_model(name,768,256,6,num_layers=2,dropout=.5,proj_dim=768);model.load_state_dict(payload['state_dict']);model.to('cuda').eval()
  with torch.no_grad():pred=model(b['x'].to('cuda'),b['edges'].to('cuda')).argmax(1).cpu()
  val=accuracy_score(b['y'][masks[1]],pred[masks[1]]);acc=accuracy_score(b['y'][masks[2]],pred[masks[2]])
  f1=f1_score(b['y'][masks[2]],pred[masks[2]],average='macro')
  r=nc[protocol+'/text+image'][name]['runs'][seed]
  check(abs(val-r['val_acc'])<1e-12 and abs(acc-r['test_acc'])<1e-12 and abs(f1-r['test_f1'])<1e-12,'independent checkpoint predictions '+protocol+'/'+str(seed))
  path=OUT/'predictions'/f'{protocol}_seed{seed}.csv';path.parent.mkdir(exist_ok=True)
  with path.open('w',newline='') as f:
   writer=csv.writer(f);writer.writerow(['entity_id','test_label','prediction'])
   for i in masks[2].nonzero().flatten().tolist():writer.writerow([b['ids'][i],int(b['y'][i]),int(pred[i])])
  checkpoint_witnesses.append(dict(protocol=protocol,seed=seed,model=name,checkpoint_sha256=sha(cp),test_acc=acc,val_acc=val,test_f1=f1))
 for field in ['retrieval_ti','retrieval_it']:check(s['modality'][protocol][field]['n']==int(masks[2].sum()),'full-gallery endpoint '+protocol+'/'+field)
check(s['training_runs']==dict(nc=270,lp_clustering=90,matching=15,total=375),'complete training accounting')
for path,value in s['source_hashes'].items():
 frozen=OUT/'execution_snapshot'/path
 check(sha(frozen if frozen.exists() else ROOT/path)==value,'source reproducibility '+path)
audit=dict(status='PASS',checks=len(checks),real_images=900,text_embeddings=900,image_embeddings=900,
           all_six_tasks=True,training_runs=s['training_runs'],checkpoint_prediction_witnesses=checkpoint_witnesses)
(OUT/'case_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps({k:v for k,v in audit.items() if k!='checkpoint_prediction_witnesses'},indent=2))
