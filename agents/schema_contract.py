"""Versioned executable schema contract. A list target never implies its first item."""
from __future__ import annotations
import re,json,math,collections

class PathError(ValueError):pass

def tokens(path):
 if not isinstance(path,str) or not path:raise PathError('empty field path')
 out=[]
 for part in path.split('.'):
  m=re.fullmatch(r'([^\[\].]+)((?:\[\d+\])*)',part)
  if not m:raise PathError('malformed path: '+path)
  out.append(m[1]);out.extend(int(x) for x in re.findall(r'\[(\d+)\]',m[2]))
 return out

def resolve(row,path):
 x=row
 for token in tokens(path):
  if isinstance(token,int):
   if not isinstance(x,list):raise PathError('index requires a list: '+path)
   if token>=len(x):raise PathError('list index missing: '+path)
  elif not isinstance(x,dict) or token not in x:raise PathError('key missing: '+path)
  x=x[token]
 return x

def scalar(x):
 return isinstance(x,(str,int,float,bool)) and x!='' and not (isinstance(x,float) and not math.isfinite(x))
def key(x):return json.dumps([type(x).__name__,x],ensure_ascii=False,separators=(',',':'))

def inspect_schema(items,proposal,confirmed=None,min_class_size=20,missing_policy='error'):
 """Three levels: base-field existence; executable scalar/list contract; confirmed task.

 Missing targets/IDs are rejected unless the confirmed contract explicitly chooses
 'exclude'. Empty relation lists are valid absence, not an invented category.
 Vocabulary contains retained typed scalar labels in stable order, contiguous IDs.
 """
 if missing_policy not in ['error','exclude']:raise ValueError('missing_policy must be explicit')
 errors=[];missing=[];normalized=[];fields=[proposal.get('id_field',''),proposal.get('label_field','')]+[r.get('group_by','') for r in proposal.get('relations',[])]
 existence=all(any(isinstance(row,dict) and path.split('.')[0].split('[')[0] in row for row in items) for path in fields if path) and all(fields)
 for i,row in enumerate(items):
  try:
   idv=resolve(row,fields[0]);lab=resolve(row,fields[1])
   if not scalar(idv) or isinstance(idv,bool):raise PathError('ID must be one scalar string/number')
   if not scalar(lab):raise PathError('target must be one finite categorical scalar; no implicit indexing')
   rels={}
   for rel in proposal.get('relations',[]):
    vals=resolve(row,rel['group_by']);vals=vals if isinstance(vals,list) else ([] if vals in [None,''] else [vals])
    if any(not scalar(v) for v in vals):raise PathError('relation requires scalar or list of finite scalars')
    rels[rel['name']]=sorted(set(key(v) for v in vals))
   normalized.append({'id':str(idv),'target_key':key(lab),'target':lab,'relations':rels,'source_row':i})
  except (PathError,KeyError,TypeError) as e:
   missing.append({'row':i,'reason':str(e)})
 if missing_policy=='error' and missing:errors.append('path/type/missing violations: '+str(len(missing)))
 ids=[r['id'] for r in normalized]
 if len(ids)!=len(set(ids)):errors.append('duplicate scalar entity IDs')
 if len({type(r['target']).__name__ for r in normalized})>1:errors.append('mixed scalar target types require explicit normalization')
 counts=collections.Counter(r['target_key'] for r in normalized);vocab=sorted(k for k,c in counts.items() if c>=min_class_size);mapping={k:i for i,k in enumerate(vocab)}
 if not 2<=len(vocab)<=100:errors.append('retained categorical support outside 2..100')
 retained=[{**r,'label_index':mapping[r['target_key']]} for r in normalized if r['target_key'] in mapping];retained.sort(key=lambda r:r['id'])
 task_consistent=None
 if confirmed is not None:
  # Path/index and relation membership define the task; never strip [0].
  try:
   task_consistent=all(tokens(proposal[f])==tokens(confirmed[f]) for f in ['id_field','label_field']) and sorted((r['name'],tuple(tokens(r['group_by']))) for r in proposal.get('relations',[]))==sorted((r['name'],tuple(tokens(r['group_by']))) for r in confirmed.get('relations',[]))
  except (PathError,KeyError):task_consistent=False
 return {'version':'executable-schema-v2','field_exists':bool(existence),'path_type_executable':bool(existence and not errors),'task_consistent':task_consistent,'valid':bool(existence and not errors and task_consistent is True),'errors':errors,'invalid_rows':missing,'missing_policy':missing_policy,'normalized_rows':retained,'vocabulary':vocab,'vocabulary_rule':'typed scalar JSON keys sorted after min-support filtering; contiguous indices','retained':len(retained)}

def confirmed_task(proposal,confirmed):
 """Compare independently of relation display names; exact indexed target and group paths."""
 return tokens(proposal['id_field'])==tokens(confirmed['id_field']) and tokens(proposal['label_field'])==tokens(confirmed['label_field']) and sorted(tuple(tokens(r['group_by'])) for r in proposal.get('relations',[]))==sorted(tuple(tokens(r['group_by'])) for r in confirmed.get('relations',[]))
