#!/usr/bin/env python3
"""Four-protocol modality retrieval, alignment and matching.

All tasks consume the canonical node masks. Retrieval and alignment are
test-cohort diagnostics over frozen encoder outputs. Matching trains only on
train-node pairs, selects on validation-node pairs and evaluates on disjoint
test-node pairs. Same-label hard negatives suppress the class shortcut.
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark.data.domain_loader import load_domain_graph, PROCESSED_DIR
from benchmark.data.protocols import PROTOCOLS, protocol_excluded_relations


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def load_aligned(domain, modality, node_ids):
    path = PROCESSED_DIR / domain / f"{modality}_embeddings.pt"
    obj = torch.load(path, map_location="cpu", weights_only=False)
    rows = {str(x): i for i, x in enumerate(obj["ids"])}
    emb = obj["embeddings"].float()
    out = torch.zeros(len(node_ids), emb.size(1))
    found = torch.zeros(len(node_ids), dtype=torch.bool)
    for i, nid in enumerate(node_ids):
        j = rows.get(str(nid))
        if j is not None:
            out[i] = emb[j]; found[i] = bool(torch.isfinite(emb[j]).all() and emb[j].norm() > 1e-12)
    out = F.normalize(torch.nan_to_num(out), dim=1)
    return out, found, path


@torch.no_grad()
def retrieval(q, g, device, ks=(1,5,10), chunk=512):
    q, g = q.to(device), g.to(device)
    n = q.size(0); counts = {k: 0 for k in ks}; rr = 0.0
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        sim = q[s:e] @ g.t()
        true = sim[torch.arange(e-s, device=device), torch.arange(s,e,device=device)]
        # Exact rank; ties are conservatively ranked after all equal scores.
        rank = (sim >= true[:,None]).sum(1)
        for k in ks: counts[k] += int((rank <= k).sum())
        rr += float((1.0 / rank.float()).sum())
    return {**{f"r@{k}": counts[k]/n for k in ks}, "mrr": rr/n, "n": n}


def linear_cka(x, y):
    x = x.double(); y = y.double()
    x = x - x.mean(0); y = y - y.mean(0)
    cross = x.t() @ y
    denom = torch.linalg.norm(x.t() @ x) * torch.linalg.norm(y.t() @ y)
    return float((torch.linalg.norm(cross) ** 2 / denom.clamp_min(1e-30)).item())


def alignment(x, y, seed=1, max_n=10000):
    if x.size(0) > max_n:
        g = torch.Generator().manual_seed(seed)
        ix = torch.randperm(x.size(0), generator=g)[:max_n]
        x, y = x[ix], y[ix]
    cos_pos = (x*y).sum(1)
    perm = torch.roll(torch.arange(y.size(0)), 1)
    cos_neg = (x*y[perm]).sum(1)
    return {"linear_cka": linear_cka(x,y),
            "paired_cosine_mean": float(cos_pos.mean()),
            "shifted_negative_cosine_mean": float(cos_neg.mean()),
            "cosine_gap": float(cos_pos.mean()-cos_neg.mean()), "n": int(x.size(0))}


def hard_negative_targets(indices, labels, seed):
    rng = np.random.RandomState(seed)
    ids = indices.cpu().numpy(); yy = labels[indices].cpu().numpy()
    out = np.empty(len(ids), dtype=np.int64)
    all_perm = rng.permutation(ids)
    for lab in np.unique(yy):
        pos = np.where(yy == lab)[0]
        if len(pos) > 1:
            vals = ids[pos].copy(); rng.shuffle(vals); vals = np.roll(vals, 1)
            # Repair rare fixed points deterministically.
            if np.any(vals == ids[pos]): vals = np.roll(ids[pos], 1)
            out[pos] = vals
        else:
            i = pos[0]
            choices = all_perm[all_perm != ids[i]]
            out[i] = choices[0]
    assert np.all(out != ids)
    return torch.from_numpy(out)


def pair_set(a, b, labels, mask, sampler_seed):
    idx = mask.nonzero(as_tuple=True)[0]
    neg_idx = hard_negative_targets(idx, labels, sampler_seed)
    pos = torch.cat([a[idx], b[idx]], 1)
    neg = torch.cat([a[idx], b[neg_idx]], 1)
    x = torch.cat([pos,neg], 0)
    y = torch.cat([torch.ones(len(idx)),torch.zeros(len(idx))])
    return x, y, idx, neg_idx


class Matcher(nn.Module):
    def __init__(self, d, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,hidden),nn.ReLU(),nn.Dropout(dropout),
                                 nn.Linear(hidden,hidden//2),nn.ReLU(),nn.Dropout(dropout),
                                 nn.Linear(hidden//2,1))
    def forward(self,x): return self.net(x).squeeze(1)


def metrics(y, logits):
    yy=y.cpu().numpy(); s=torch.sigmoid(logits).cpu().numpy(); p=(s>=.5).astype(int)
    return {"auc":float(roc_auc_score(yy,s)),"ap":float(average_precision_score(yy,s)),
            "accuracy":float(accuracy_score(yy,p)),"f1":float(f1_score(yy,p))}


def train_match(a,b,labels,masks,seed,args,device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    tr=pair_set(a,b,labels,masks[0],args.sampler_seed)
    va=pair_set(a,b,labels,masks[1],args.sampler_seed+1)
    te=pair_set(a,b,labels,masks[2],args.sampler_seed+2)
    # Entity identity isolation is the essential leakage check.
    assert not (set(tr[2].tolist()) & set(va[2].tolist()))
    assert not (set(tr[2].tolist()) & set(te[2].tolist()))
    assert not (set(va[2].tolist()) & set(te[2].tolist()))
    model=Matcher(tr[0].size(1),args.hidden,args.dropout).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=1e-5)
    loss_fn=nn.BCEWithLogitsLoss(); best=-1.; state=None; stale=0; best_epoch=0
    xtr,ytr=tr[0].to(device),tr[1].to(device); xva,yva=va[0].to(device),va[1].to(device)
    gen=torch.Generator().manual_seed(seed)
    started=time.time()
    for epoch in range(1,args.epochs+1):
        model.train(); order=torch.randperm(len(ytr),generator=gen)
        for s in range(0,len(order),args.batch_size):
            ix=order[s:s+args.batch_size].to(device); opt.zero_grad()
            loss=loss_fn(model(xtr[ix]),ytr[ix]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): vm=metrics(yva,model(xva))
        if vm["ap"]>best+1e-8:
            best=vm["ap"]; stale=0; best_epoch=epoch
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale+=1
            if stale>=args.patience: break
    model.load_state_dict(state); model.eval()
    with torch.no_grad(): out=metrics(te[1],model(te[0].to(device)))
    out.update({"seed":seed,"best_epoch":best_epoch,"val_ap":best,"seconds":time.time()-started,
                "pair_counts":[len(tr[2]),len(va[2]),len(te[2])],"entity_overlap":0})
    return out


def agg(runs,key):
    v=[r[key] for r in runs]
    return {"mean":float(np.mean(v)),"std":float(np.std(v)),"values":[float(x) for x in v]}


def run(args):
    device=f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    graph=load_domain_graph(args.domain,exclude_relations=protocol_excluded_relations(args.domain))
    node_ids=graph["node_ids"]; labels=graph["y"]
    mods=["text","image","formula"] if args.domain=="arxiv" else ["text","image"]
    embs={}; coverage={}; files={}
    for m in mods:
        embs[m],coverage[m],files[m]=load_aligned(args.domain,m,node_ids)
    pairs=[(mods[i],mods[j]) for i in range(len(mods)) for j in range(i+1,len(mods))]
    vp=ROOT/"benchmark/results/protocol_views"/f"{args.domain}_protocol_views.pt"
    mp=ROOT/"benchmark/results/protocol_views"/f"{args.domain}_protocol_manifest.json"
    views=torch.load(vp,map_location="cpu",weights_only=False)
    protocols=list(PROTOCOLS) if args.protocols=="all" else args.protocols.split(",")
    outdir=ROOT/"benchmark/results/four_protocol_tasks/modality"; outdir.mkdir(parents=True,exist_ok=True)
    outpath=outdir/f"{args.domain}_modality_tasks.json"
    config={"sampler_seed":args.sampler_seed,"epochs_max":args.epochs,"hidden_dim":args.hidden,
            "dropout":args.dropout,"lr":args.lr,"weight_decay":1e-5,"patience":args.patience,
            "batch_size":args.batch_size,"retrieval_chunk":args.retrieval_chunk,"recall_ks":args.ks}
    if outpath.exists() and not args.force:
        previous=json.loads(outpath.read_text(encoding="utf-8"))
        if (previous.get("completed") is True and previous.get("protocols")==protocols
                and previous.get("seeds")==list(range(args.seeds))
                and previous.get("hyperparameters") in (None,config)
                and previous.get("input_sha256",{}).get("protocol_views")==sha256(vp)
                and previous.get("input_sha256",{}).get("protocol_manifest")==sha256(mp)):
            print(f"verified complete; skip {outpath}")
            return
    result={"schema_version":"four-protocol-modality-tasks-v1","domain":args.domain,
            "modalities":mods,"pairs":[f"{a}-{b}" for a,b in pairs],"protocols":protocols,
            "seeds":list(range(args.seeds)),"input_sha256":{"protocol_views":sha256(vp),"protocol_manifest":sha256(mp),
            **{f"{m}_embeddings":sha256(files[m]) for m in mods}},
            "hyperparameters":config,
            "policies":{"retrieval":"exact same-entity rank on test cohort only",
            "alignment":"test-cohort descriptive linear CKA and cosine gap",
            "matching":"train/val/test entity-disjoint; same-label hard negatives; validation AP selection",
            "topology_note":"modality tasks are node-local; protocols change cohorts, not frozen features"},"results":{}}
    for protocol in protocols:
        masks0=tuple(views[protocol]["masks"])
        for ma,mb in pairs:
            valid=coverage[ma]&coverage[mb]
            masks=tuple(m&valid for m in masks0)
            if min(int(m.sum()) for m in masks)<2: raise ValueError("insufficient paired coverage")
            key=f"{protocol}/{ma}-{mb}"; print(f"[{args.domain}/{key}]",flush=True)
            te=masks[2].nonzero(as_tuple=True)[0]; aa,bb=embs[ma][te],embs[mb][te]
            ret={f"{ma}_to_{mb}":retrieval(aa,bb,device,args.ks,args.retrieval_chunk),
                 f"{mb}_to_{ma}":retrieval(bb,aa,device,args.ks,args.retrieval_chunk)}
            ali=alignment(aa,bb,args.sampler_seed)
            runs=[]
            for seed in range(args.seeds):
                rr=train_match(embs[ma],embs[mb],labels,masks,seed,args,device); runs.append(rr)
                print(f"  seed={seed} match AUC={rr['auc']:.4f} AP={rr['ap']:.4f}",flush=True)
            result["results"][key]={"protocol":protocol,"pair":[ma,mb],
                "paired_node_counts":[int(m.sum()) for m in masks],"retrieval":ret,"alignment":ali,
                "matching":{"runs":runs,**{k:agg(runs,k) for k in ("auc","ap","accuracy","f1")}},
                "entity_overlap":0}
            outpath.write_text(json.dumps(result,indent=2),encoding="utf-8")
    result["completed"]=True; outpath.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(outpath)


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--domain",required=True,choices=["arxiv","steam","tmdb","wikiart"])
    p.add_argument("--protocols",default="all"); p.add_argument("--seeds",type=int,default=5)
    p.add_argument("--gpu",type=int,default=None); p.add_argument("--hidden",type=int,default=256)
    p.add_argument("--dropout",type=float,default=.3); p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--epochs",type=int,default=50); p.add_argument("--patience",type=int,default=8)
    p.add_argument("--batch-size",type=int,default=1024); p.add_argument("--sampler-seed",type=int,default=1)
    p.add_argument("--retrieval-chunk",type=int,default=512); p.add_argument("--ks",type=int,nargs="+",default=[1,5,10])
    p.add_argument("--force",action="store_true")
    run(p.parse_args())
