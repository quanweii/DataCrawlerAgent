"""Inspect or trial-build a file-backed MAG declaration without an LLM/API call."""
from pathlib import Path
import argparse,json,sys,tempfile,re
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def main():
 ap=argparse.ArgumentParser(description=__doc__);sp=ap.add_subparsers(dest='mode',required=True)
 inspect=sp.add_parser('inspect');inspect.add_argument('--source',required=True)
 trial=sp.add_parser('trial');trial.add_argument('--config',required=True);trial.add_argument('--output',required=True)
 args=ap.parse_args()
 if args.mode=='inspect':
  from agents.schema_discovery import profile_fields
  rows=json.loads(Path(args.source).read_text(encoding='utf8'))
  if not isinstance(rows,list) or not all(isinstance(x,dict) for x in rows):raise ValueError('Source must be a JSON array of objects.')
  print(json.dumps({'records':len(rows),'profile':profile_fields(rows)},ensure_ascii=False,indent=2));return
 from domains.domain_schema import load_domain_yaml,yaml_to_graph_config
 from agents.schema_contract import inspect_schema
 from crawlers.config_driven import build_graph_from_config
 yc=load_domain_yaml(args.config);rows=json.loads(Path(yc.data.data_file).read_text(encoding='utf8'))[:yc.data.max_items]
 proposal={'id_field':yc.data.id_field,'label_field':yc.data.label_field,'relations':[{'name':r.name,'group_by':r.group_by} for r in yc.relations]}
 audit=inspect_schema(rows,proposal,proposal,yc.data.min_class_size)
 if not audit['valid']:raise ValueError(json.dumps({k:v for k,v in audit.items() if k!='normalized_rows'},ensure_ascii=False))
 dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
 # Structural trial placeholders contain no target text and are never encoded.
 with tempfile.TemporaryDirectory(prefix='mag-forge-trial-') as tmp:
  for row in audit['normalized_rows']:
   iid=re.sub(yc.data.id_sanitize,'_',row['id']) if yc.data.id_sanitize else row['id']
   if Path(iid).name!=iid or iid in ['.','..'] or '/' in iid or '\\' in iid:raise ValueError('Use an ID sanitization rule for path-bearing IDs.')
   p=Path(tmp)/iid;p.mkdir(parents=True,exist_ok=True);(p/(iid+'.md')).write_text('# Entity '+iid+'\n',encoding='utf8');(p/'images').mkdir()
  nodes,edges,stats=build_graph_from_config(rows,tmp,yaml_to_graph_config(yc))
 graph={'nodes':nodes,'edges':edges,'metadata':{'statistics':stats,'scope':'structural trial only; placeholder text is not benchmark content'}}
 (dest/'trial_graph.json').write_text(json.dumps(graph,ensure_ascii=False),encoding='utf8')
 report={'domain':yc.domain,'nodes':len(nodes),'edges':len(edges),'classes':len(audit['vocabulary']),'contract_valid':True,'statistics':stats,'scope':'structural trial; no feature encoding, LLM calls or scientific scores'}
 (dest/'trial_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
