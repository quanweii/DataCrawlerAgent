#!/usr/bin/env python3
"""Protocol-aware link prediction and label-free node clustering.

For every domain/protocol/model/seed this runner trains a graph autoencoder on
training positive edges only.  Validation/test positive edges are never in the
message graph.  The same learned node representation is evaluated for:

* link prediction: ROC-AUC/AP on fixed, filtered 1:1 negatives;
* node clustering: KMeans fitted on train-node embeddings and evaluated on
  test-node embeddings with NMI/ARI. Labels choose k and score clusters only.

The four node protocols are consumed from the canonical saved protocol views.
For transductive LP, graph edges are split 60/20/20.  For the other protocols,
within-train/within-val/within-test edges are the corresponding positives.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, average_precision_score,
                             normalized_mutual_info_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.data.domain_loader import SingleDomainDataset
from benchmark.data.protocols import PROTOCOLS, protocol_excluded_relations
from benchmark.models.gnn import get_model, model_lr


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def undirected_unique(edge_index):
    a = torch.minimum(edge_index[0], edge_index[1])
    b = torch.maximum(edge_index[0], edge_index[1])
    keep = a.ne(b)
    pairs = torch.stack([a[keep], b[keep]], 1)
    return torch.unique(pairs, dim=0).t().contiguous()


def bidirectional(pairs):
    return torch.cat([pairs, pairs.flip(0)], dim=1)


def split_positive_edges(protocol, edge_index, masks, seed):
    pairs = undirected_unique(edge_index)
    if protocol == "transductive":
        g = torch.Generator().manual_seed(seed)
        pairs = pairs[:, torch.randperm(pairs.size(1), generator=g)]
        n = pairs.size(1)
        ntr, nva = int(0.6 * n), int(0.2 * n)
        return pairs[:, :ntr], pairs[:, ntr:ntr+nva], pairs[:, ntr+nva:]
    out = []
    for mask in masks:
        keep = mask[pairs[0]] & mask[pairs[1]]
        out.append(pairs[:, keep])
    return tuple(out)


def positive_keys(pairs, n):
    return set((pairs[0] * n + pairs[1]).tolist())


def sample_negatives(nodes, count, forbidden, n, seed):
    """Fixed uniform negatives within the eligible node cohort."""
    nodes = nodes.cpu().numpy().astype(np.int64)
    if len(nodes) < 2:
        raise ValueError("negative sampling requires at least two eligible nodes")
    rng = np.random.RandomState(seed)
    chosen = set()
    src, dst = [], []
    node_set = set(nodes.tolist())
    max_possible = len(nodes) * (len(nodes) - 1) // 2 - sum(
        1 for key in forbidden
        if (key // n) in node_set and (key % n) in node_set)
    if count > max_possible:
        raise ValueError(f"requested {count} negatives but only {max_possible} exist")
    while len(src) < count:
        batch = max(1024, 3 * (count - len(src)))
        aa = rng.choice(nodes, batch)
        bb = rng.choice(nodes, batch)
        lo, hi = np.minimum(aa, bb), np.maximum(aa, bb)
        for u, v in zip(lo.tolist(), hi.tolist()):
            if u == v:
                continue
            key = u * n + v
            if key in forbidden or key in chosen:
                continue
            chosen.add(key); src.append(u); dst.append(v)
            if len(src) == count:
                break
    return torch.tensor([src, dst], dtype=torch.long)


def prepare_splits(protocol, edge_index, masks, n, seed):
    trp, vap, tep = split_positive_edges(protocol, edge_index, masks, seed)
    if min(trp.size(1), vap.size(1), tep.size(1)) == 0:
        raise ValueError(f"empty positive edge split: {[trp.size(1),vap.size(1),tep.size(1)]}")
    allp = undirected_unique(edge_index)
    forbidden = positive_keys(allp, n)
    if protocol == "transductive":
        eligible = [torch.arange(n)] * 3
    else:
        eligible = [m.nonzero(as_tuple=True)[0] for m in masks]
    neg = [sample_negatives(eligible[i], p.size(1), forbidden, n, seed + 101 + i)
           for i, p in enumerate((trp, vap, tep))]
    # Hard safety checks used again by the standalone validator.
    psets = [positive_keys(p, n) for p in (trp, vap, tep)]
    assert not (psets[0] & psets[1] or psets[0] & psets[2] or psets[1] & psets[2])
    for ne in neg:
        assert not (positive_keys(ne, n) & forbidden)
    msg = bidirectional(trp)
    assert not (positive_keys(undirected_unique(msg), n) & (psets[1] | psets[2]))
    return (trp, vap, tep), tuple(neg), msg


def scores(z, edges):
    return (z[edges[0]] * z[edges[1]]).sum(dim=1)


def binary_metrics(pos, neg):
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[pos, neg]
    return {"auc": float(roc_auc_score(y, s)),
            "ap": float(average_precision_score(y, s))}


@torch.no_grad()
def cluster_metrics(z, y, train_mask, test_mask, seed):
    tr = train_mask.cpu().numpy(); te = test_mask.cpu().numpy()
    z = z.cpu().numpy(); yy = y.cpu().numpy()
    k = int(torch.unique(y[train_mask]).numel())
    if k < 2:
        return {"nmi": None, "ari": None, "k": k, "status": "not_estimable"}
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    km.fit(z[tr])
    pred = km.predict(z[te])
    return {"nmi": float(normalized_mutual_info_score(yy[te], pred)),
            "ari": float(adjusted_rand_score(yy[te], pred)), "k": k,
            "status": "ok"}


def train_one(x, full_view_edges, msg_edges, pos, neg, masks, y, model_name,
              seed, args, device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    model = get_model(model_name, x.size(1), args.hidden, args.hidden,
                      num_layers=args.layers, dropout=args.dropout,
                      proj_dim=None).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=model_lr(model_name, default=args.lr),
                           weight_decay=5e-4)
    xd, md = x.to(device), msg_edges.to(device)
    trp, vap, tep = [e.to(device) for e in pos]
    trn, van, ten = [e.to(device) for e in neg]
    best, state, stale, best_epoch = -1.0, None, 0, 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); opt.zero_grad()
        z = model(xd, md)
        ps, ns = scores(z, trp), scores(z, trn)
        loss = -F.logsigmoid(ps).mean() - F.logsigmoid(-ns).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at epoch {epoch}")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                zz = model(xd, md)
                vm = binary_metrics(scores(zz, vap).cpu().numpy(), scores(zz, van).cpu().numpy())
            if vm["ap"] > best + 1e-8:
                best, stale, best_epoch = vm["ap"], 0, epoch
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += args.eval_every
                if stale >= args.patience: break
    if state is None:
        raise RuntimeError("no valid checkpoint")
    model.load_state_dict(state); model.eval()
    with torch.no_grad():
        z_lp = model(xd, md)
        lp = binary_metrics(scores(z_lp, tep).cpu().numpy(), scores(z_lp, ten).cpu().numpy())
        # Clustering may consume the protocol-allowed within-split topology at inference.
        z_cluster = model(xd, full_view_edges.to(device))
        cl = cluster_metrics(z_cluster, y, masks[0], masks[2], seed)
    return {"seed": seed, "best_epoch": best_epoch, "val_ap": best,
            "test_auc": lp["auc"], "test_ap": lp["ap"],
            "nmi": cl["nmi"], "ari": cl["ari"], "cluster_k": cl["k"],
            "seconds": time.time() - started}


def mean_std(runs, key):
    vals = [r[key] for r in runs if r[key] is not None]
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
            "values": [float(v) for v in vals]} if vals else {"mean": None, "std": None, "values": []}


def run(args):
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu is not None else (
        "cuda" if torch.cuda.is_available() else "cpu")
    domain = args.domain
    excluded = protocol_excluded_relations(domain)
    ds = SingleDomainDataset(domain, modality=args.modality, exclude_relations=excluded)
    view_path = ROOT / "benchmark/results/protocol_views" / f"{domain}_protocol_views.pt"
    manifest_path = ROOT / "benchmark/results/protocol_views" / f"{domain}_protocol_manifest.json"
    views = torch.load(view_path, map_location="cpu", weights_only=False)
    protocols = list(PROTOCOLS) if args.protocols == "all" else args.protocols.split(",")
    models = [x for x in args.models.split(",") if x]
    out_dir = ROOT / "benchmark/results/four_protocol_tasks/graph"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.out_tag}" if args.out_tag else ""
    out_path = out_dir / f"{domain}_graph_tasks{suffix}.json"
    if out_path.exists() and args.resume:
        result = json.loads(out_path.read_text())
    else:
        result = {"schema_version": "four-protocol-graph-tasks-v1", "domain": domain,
                  "modality": args.modality, "models": models,
                  "seeds": list(range(args.seeds)), "protocols": protocols,
                  "hyperparameters": {"split_seed": args.split_seed, "epochs_max": args.epochs,
                      "hidden_dim": args.hidden, "layers": args.layers, "dropout": args.dropout,
                      "default_lr": args.lr, "gin_lr": model_lr("gin", default=args.lr),
                      "weight_decay": 5e-4, "patience": args.patience,
                      "validation_interval": args.eval_every, "kmeans_n_init": 10},
                  "input_sha256": {"protocol_views": sha256(view_path),
                                     "protocol_manifest": sha256(manifest_path)},
                  "policies": {
                      "representation_training": "label-free graph autoencoder",
                      "lp_message_graph": "training positive edges only",
                      "lp_selection": "validation AP",
                      "clustering": "KMeans fit on train-node embeddings; NMI/ARI on test nodes",
                      "clustering_inference_topology": "full protocol-allowed view; no labels",
                      "target_relation_exclusions": list(excluded)}, "results": {}}
    for protocol in protocols:
        v = views[protocol]; masks = tuple(v["masks"]); ei = v["edge_index"]
        assert all(not bool((masks[i] & masks[j]).any()) for i in range(3) for j in range(i+1,3))
        pos, neg, msg = prepare_splits(protocol, ei, masks, ds.data.num_nodes, args.split_seed)
        split_meta = {"node_counts": [int(m.sum()) for m in masks],
                      "positive_edges": [int(e.size(1)) for e in pos],
                      "negative_edges": [int(e.size(1)) for e in neg],
                      "message_directed_edges": int(msg.size(1)),
                      "view_directed_edges": int(ei.size(1)),
                      "positive_overlap": 0, "negative_positive_overlap": 0,
                      "heldout_positive_in_message_graph": 0}
        for model_name in models:
            cell = f"{protocol}/{model_name}"
            old = result["results"].get(cell, {})
            runs = old.get("runs", []) if args.resume else []
            done = {int(x["seed"]) for x in runs}
            for seed in range(args.seeds):
                if seed in done: continue
                print(f"[{domain}/{protocol}/{model_name}/seed={seed}]", flush=True)
                rr = train_one(ds.data.x, ei, msg, pos, neg, masks, ds.data.y,
                               model_name, seed, args, device)
                runs.append(rr)
                result["results"][cell] = {"protocol": protocol, "model": model_name,
                    "split": split_meta, "runs": runs,
                    "link_prediction": {k: mean_std(runs, k) for k in ("test_auc","test_ap")},
                    "node_clustering": {k: mean_std(runs, k) for k in ("nmi","ari")}}
                out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                print(f"  AUC={rr['test_auc']:.4f} AP={rr['test_ap']:.4f} "
                      f"NMI={rr['nmi']:.4f} ARI={rr['ari']:.4f}", flush=True)
    result["completed"] = True
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True, choices=["arxiv","steam","tmdb","wikiart"])
    p.add_argument("--modality", default=None)
    p.add_argument("--protocols", default="all")
    p.add_argument("--models", default="mlp,gcn,gat,graphsage,gin,gcnii")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--split-seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out-tag", default="")
    a = p.parse_args()
    if a.modality is None:
        a.modality = "text+image+formula" if a.domain == "arxiv" else "text+image"
    run(a)
