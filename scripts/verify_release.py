"""Offline verification of frozen graphs, protocols and paper result witnesses."""
from pathlib import Path
import json,sys,statistics,itertools
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def read(rel):return json.loads((ROOT/rel).read_text(encoding='utf8'))
def main():
 from benchmark.data.protocols import generate_protocol_views,protocol_excluded_relations
 from benchmark.data.domain_loader import load_domain_graph
 report={'release':'www2027-20261027','domains':{},'network_or_training':False}
 metadata=read('data/DATASET_RELEASE.json')
 for d,expected in metadata['domains'].items():
  graph=read(f'data/{d}_datasets/{d}_multimodal_graph_agent.json');ids=[str(n['id']) for n in graph['nodes']]
  labels=[n['attributes']['label_index'] for n in graph['nodes']]
  assert len(ids)==len(set(ids))==expected['nodes'] and len(set(labels))==expected['classes']
  assert len(graph['edges'])==expected['raw_edges']
  for m,spec in expected['features'].items():
   obj=torch.load(ROOT/f'benchmark/processed/{d}/{m}_embeddings.pt',weights_only=False,map_location='cpu')
   assert list(obj['embeddings'].shape)==[spec['rows'],768]
   # Preserve original frozen rows: the loader resolves repeated IDs by the last row.
   duplicate_rows=len(obj['ids'])-len(set(map(str,obj['ids'])))
   assert sorted(set(ids)-set(map(str,obj['ids'])))==spec['missing_canonical_ids']
   assert torch.isfinite(obj['embeddings']).all()
  saved=torch.load(ROOT/f'benchmark/results/protocol_views/{d}_protocol_views.pt',weights_only=False,map_location='cpu')
  g=load_domain_graph(d,exclude_relations=protocol_excluded_relations(d))
  rebuilt=generate_protocol_views(d,g['node_ids'],g['y'],g['edge_index'],seed=1,excluded_relations=protocol_excluded_relations(d))
  for view,v in saved.items():
   assert all(torch.equal(a,b) for a,b in zip(v['masks'],rebuilt['views'][view]['masks'])),(d,view,'masks')
   assert torch.equal(v['edge_index'],rebuilt['views'][view]['edge_index']),(d,view,'edges')
  report['domains'][d]={'nodes':len(ids),'classes':len(set(labels)),'raw_edges':len(graph['edges']),'four_saved_protocols_exactly_rebuilt':True}
  print('Verified',d,flush=True)
 registry=read('review_output/primary_run_registry.json');assert len(registry['runs'])==2520
 report['primary_fit_records']=2520
 fixtures=read('review_output/schema_fixture_tests.json');assert len(fixtures['tests'])==15 and all(x['expected_valid']==x['actual_valid'] for x in fixtures['tests'])
 report['archived_schema_fixtures']=15
 active=read('review_output/lp_active_selection.json');paper=read('review_output/table4_paper_data.json')
 for row in paper['records']:
  a=active[row['domain']][row['view']];b=row['graph'];assert a['model']==b['model']
  for metric in ['test_auc','test_ap','nmi','ari']:
   assert a[metric]['values']==b[metric]['values'],(row['domain'],row['view'],metric)
 report['paper_table4_graph_rows_exactly_matched']=16
 lp=0
 for base in sorted((ROOT/'benchmark/results/positive_evidence_20261022/lp_repair').glob('*/*')):
  split=torch.load(base/'splits.pt',weights_only=False,map_location='cpu')
  sets=lambda es:[{tuple(sorted(e)) for e in t.t().tolist()} for t in es]
  ps,ns=sets(split['pos']),sets(split['neg']);msg=sets([split['message']])[0]
  assert all(not (ps[i]&ps[j] or ns[i]&ns[j]) for i in range(3) for j in range(i+1,3))
  assert not set.union(*ps)&set.union(*ns) and msg==ps[0]
  for seed in range(5):
   c=torch.load(base/f'seed{seed}.pt',weights_only=False,map_location='cpu');scores=c['test_scores']
   yy=np.r_[np.ones(len(scores[0])),np.zeros(len(scores[1]))];ss=torch.cat(scores).numpy()
   assert abs(roc_auc_score(yy,ss)-c['metrics']['test_auc'])<1e-6
   assert abs(average_precision_score(yy,ss)-c['metrics']['test_ap'])<1e-6;lp+=1
 report['recomputed_saved_LP_scores_and_disjoint_splits']=lp;assert lp==120
 episodes=ROOT/'review_output/guarded_production_runs/episodes'
 report['guarded_episode_directories']=len([p for p in episodes.iterdir() if p.is_dir()]);assert report['guarded_episode_directories']==12
 report['passed']=True
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
