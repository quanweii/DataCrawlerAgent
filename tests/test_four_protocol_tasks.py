import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


graph = load("four_graph", "benchmark/run_four_protocol_graph_tasks.py")
modal = load("four_modal", "benchmark/run_four_protocol_modality_tasks.py")


def test_lp_split_hides_validation_and_test_edges():
    pairs = torch.tensor([[0,0,0,1,1,2,3,4],[1,2,3,2,4,5,4,5]])
    edges = torch.cat([pairs, pairs.flip(0)], dim=1)
    masks = (torch.tensor([1,1,1,0,0,0,0],dtype=torch.bool),
             torch.tensor([0,0,0,1,0,0,0],dtype=torch.bool),
             torch.tensor([0,0,0,0,1,1,1],dtype=torch.bool))
    pos, neg, msg = graph.prepare_splits("transductive", edges, masks, 7, 1)
    keys = lambda e: graph.positive_keys(graph.undirected_unique(e), 7)
    assert not keys(msg) & (keys(pos[1]) | keys(pos[2]))
    assert all(len(keys(pos[i]) & keys(neg[i])) == 0 for i in range(3))


def test_protocol_endpoint_membership():
    edges = torch.tensor([[0,1,2,3,4,5],[1,0,3,2,5,4]])
    masks = (torch.tensor([1,1,0,0,0,0],dtype=torch.bool),
             torch.tensor([0,0,1,1,0,0],dtype=torch.bool),
             torch.tensor([0,0,0,0,1,1],dtype=torch.bool))
    pos = graph.split_positive_edges("split_disjoint", edges, masks, 1)
    for e,m in zip(pos,masks):
        assert bool((m[e[0]] & m[e[1]]).all())


def test_matching_hard_negatives_have_no_identity_match():
    idx = torch.arange(8); labels = torch.tensor([0,0,0,1,1,1,2,2])
    neg = modal.hard_negative_targets(idx, labels, 1)
    assert bool((neg != idx).all())
    assert bool((labels[neg] == labels[idx]).all())
