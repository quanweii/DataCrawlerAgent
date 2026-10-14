import json
from pathlib import Path
from agents.dataset_formatter import DatasetFormatterAgent
from agents.messages import Blackboard
from openai import OpenAI

def exercise(domain):
    bb=Blackboard();nodes=[{'id':'a','type':'item','attributes':{'label':'one','label_index':0,'image_count':1,'markdown_length':30}},
                         {'id':'b','type':'item','attributes':{'label':'two','label_index':1,'image_count':1,'markdown_length':40}}]
    edges=[{'source':'a','target':'b','type':'co_attribute','attributes':{'relation':'same_creator'}}]
    for key,value in [('nodes',nodes),('edges',edges),('stats',{'num_labels':2})]:bb.write(domain+'_graph',key,value)
    agent=DatasetFormatterAgent(OpenAI(api_key='test-local-no-network'),'test',blackboard=bb)
    agent._current_task={'domain':domain,'graph_mode':'homogeneous'}
    agent.action_collect_data(agent._current_task)
    n=agent.action_build_nodes();e=agent.action_build_edges();meta=agent._build_metadata('sample',n['nodes'],e['edges'])
    assert n['nodes']==nodes and e['edges']==edges,'Formatter must preserve IDs/endpoints/labels'
    assert meta['domain']==domain and meta['statistics']['total_images']==2
    if domain=='fresh_unregistered_domain':assert meta['schema']['edge_relations']==['same_creator']
    return {'domain':domain,'nodes':len(nodes),'edges':len(edges),'passed':True}
result={'status':'PASS','cases':[exercise(d) for d in ['fresh_unregistered_domain','steam','tmdb','wikiart']]}
Path('/home/dev_qw/www2027_paper/revisions/new_multimodal_source_20261014/formatter_test.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
