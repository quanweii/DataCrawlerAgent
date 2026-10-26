import copy
import pytest
import torch
from agents.schema_discovery import validate_schema
from agents.schema_contract import inspect_schema
from benchmark.run_four_protocol_graph_tasks import prepare_splits,positive_keys

@pytest.fixture
def source():return [{'id':str(i),'label':'A' if i%2 else 'B','groups':['g'+str(i%4)]} for i in range(40)]

@pytest.fixture
def declaration():return {'id_field':'id','label_field':'label','relations':[{'name':'shared','group_by':'groups'}]}

@pytest.mark.parametrize('field,value',[('label',None),('label',[]),('label',{'x':1}),('label',float('nan')),('id',[0])])
def test_required_scalar_contract(source,declaration,field,value):
 source[0][field]=value
 assert not validate_schema(source,declaration,20)['valid']

def test_empty_relation_and_stable_vocabulary(source,declaration):
 source[0]['groups']=[]
 a=inspect_schema(source,declaration,declaration,20)
 b=inspect_schema(source[::-1],declaration,declaration,20)
 assert a['valid'] and b['valid'] and a['vocabulary']==b['vocabulary']

def test_negative_prediction_objects_are_disjoint():
 n=12;pairs=torch.tensor([list(range(n)),[(i+1)%n for i in range(n)]])
 edges=torch.cat([pairs,pairs.flip(0)],1)
 masks=tuple(torch.tensor([i//4==s for i in range(n)]) for s in range(3))
 for seed in range(5):
  pos,neg,msg=prepare_splits('transductive',edges,masks,n,seed)
  ns=[positive_keys(e,n) for e in neg]
  assert all(not ns[i]&ns[j] for i in range(3) for j in range(i+1,3))

def test_dense_graph_rejects_impossible_disjoint_negatives():
 pairs=torch.tensor([[0,0,0,1,1,2,3,4],[1,2,3,2,4,5,4,5]])
 masks=tuple(torch.tensor([i//2==s for i in range(6)]) for s in range(3))
 with pytest.raises(ValueError,match='negatives'):
  prepare_splits('transductive',torch.cat([pairs,pairs.flip(0)],1),masks,6,1)
