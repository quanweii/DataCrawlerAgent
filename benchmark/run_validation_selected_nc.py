#!/usr/bin/env python3
"""Replay the primary NC grid with validation logging; leave historical artifacts intact.

Each job is one domain/modality, all four protocols, six models, five seeds.
Architecture selection is performed separately by aggregate_validation_selected_nc.py.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--domain', required=True)
p.add_argument('--modality', required=True)
p.add_argument('--gpu', type=int, required=True)
args = p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark.data.domain_loader import SingleDomainDataset
from benchmark.data.protocols import PROTOCOLS, generate_protocol_views, protocol_excluded_relations
from benchmark.models.gnn import get_model, model_lr
from benchmark.tasks.node_classification import train_and_eval

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
MODELS = ['mlp', 'gcn', 'gat', 'graphsage', 'gin', 'gcnii']
OUT = ROOT/'benchmark/results/validation_selected_nc_20261014'
OUT.mkdir(exist_ok=True, parents=True)
HP = dict(epochs=300, hidden=256, layers=2, dropout=.5, proj_dim=768,
          weight_decay=.0005, patience=50, validation_interval=10,
          gradient_clip=1., split_seed=1, init_seeds=list(range(5)))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def save(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)

domain, modality = args.domain, args.modality
oldpath = ROOT/f'benchmark/results/newdomain_nc/protocol_matrix/nc_{domain}_protocol_matrix.json'
old = json.loads(oldpath.read_text())
assert old['split_seed'] == 1 and old['init_seeds'] == 5
excluded = protocol_excluded_relations(domain)
ds = SingleDomainDataset(domain, modality=modality, exclude_relations=excluded)
suite = generate_protocol_views(domain, ds.graph['node_ids'], ds.data.y,
                                ds.data.edge_index, seed=1, excluded_relations=excluded)
saved = torch.load(ROOT/f'benchmark/results/protocol_views/{domain}_protocol_views.pt',
                   map_location='cpu', weights_only=False)
print('saved keys', list(saved), flush=True)
for proto in PROTOCOLS:
    a, b = suite['views'][proto], saved[proto]
    assert torch.equal(a['edge_index'], b['edge_index']), (domain, proto, 'edges changed')
    assert all(torch.equal(x,y) for x,y in zip(a['masks'], b['masks'])), (domain, proto, 'masks changed')
sources = {str(q.relative_to(ROOT)):sha(q) for q in [Path(__file__),
    ROOT/'benchmark/tasks/node_classification.py', ROOT/'benchmark/models/gnn.py',
    ROOT/'benchmark/data/domain_loader.py', ROOT/'benchmark/data/protocols.py',
    oldpath, ROOT/f'benchmark/results/protocol_views/{domain}_protocol_views.pt']}
input_hashes = dict(x=tensor_sha(ds.data.x), y=tensor_sha(ds.data.y),
                    base_edges=tensor_sha(ds.data.edge_index),
                    node_ids=hashlib.sha256(json.dumps(ds.graph['node_ids']).encode()).hexdigest())
for proto in PROTOCOLS:
    path = OUT/f'nc_{domain}_{proto}_{modality.replace("+", "_")}.json'
    v = suite['views'][proto]
    tr, va, te = v['masks']
    view_hashes = dict(edges=tensor_sha(v['edge_index']), masks=[tensor_sha(t) for t in v['masks']])
    if path.exists():
        cell = json.loads(path.read_text())
        assert cell['hyperparameters'] == HP and cell['input_hashes'] == input_hashes
        assert cell['view_hashes'] == view_hashes and cell['source_hashes'] == sources
    else:
        cell = dict(domain=domain, protocol=proto, modality=modality,
                    hyperparameters=HP, source_hashes=sources, input_hashes=input_hashes,
                    view_hashes=view_hashes, partition_matches_saved=True,
                    split_sizes=[int(t.sum()) for t in v['masks']],
                    directed_edges=int(v['edge_index'].size(1)), models={})
    data = Data(x=ds.data.x, edge_index=v['edge_index'], y=ds.data.y)
    data.num_nodes = ds.data.num_nodes
    for model in MODELS:
        runs = cell['models'].get(model, {}).get('runs', [])
        for seed in range(len(runs), 5):
            torch.manual_seed(seed)
            m = get_model(model, data.x.size(1), 256, int(data.y.max())+1,
                          num_layers=2, dropout=.5, proj_dim=768)
            r = train_and_eval(m, data, tr, va, te, num_epochs=300,
                               lr=model_lr(model), device='cuda')
            runs.append(dict(seed=seed, **r))
            vals = [r['val_acc'] for r in runs]
            accs = [r['test_acc'] for r in runs]
            f1s = [r['test_f1'] for r in runs]
            cell['models'][model] = dict(runs=runs, val_accs=vals, accs=accs, f1s=f1s,
                val_acc=float(np.mean(vals)), val_acc_std=float(np.std(vals)),
                acc=float(np.mean(accs)), acc_std=float(np.std(accs)),
                f1=float(np.mean(f1s)), f1_std=float(np.std(f1s)))
            save(path, cell)
            print(f'{domain}/{modality}/{proto}/{model}/{seed}: '
                  f'val={r["val_acc"]:.5f} test={r["test_acc"]:.5f} '
                  f'time={r["training_time"]:.2f}s', flush=True)
        data = data.cpu()
    cell['complete'] = True
    save(path, cell)
print('JOB COMPLETE', domain, modality, flush=True)
