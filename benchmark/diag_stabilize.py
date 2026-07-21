#!/usr/bin/env python3
"""Validate the training-stability fix for the 'multimodal worse' artifact.

The collapse is concentrated in GIN (hyperparameter-fragile) + occasional GCNII bad seed.
The NC loop has NO gradient clipping (LP loop does). Test whether stabilization recovers
the collapsed cells to the stable-model range.

Target cells (the worst collapses from the headline):
  wikiart / gin / text+image   -> collapsed to 0.035 (majority class)
  wikiart / gin / text         -> 0.705 but high variance
  wikiart / gcnii / text+image -> seed-0 collapsed to 0.876 (dragged mean to 0.906)

Compares settings: default | +gradclip | lower-lr+clip | lower-dropout+clip
Usage: python benchmark/diag_stabilize.py --gpu 2
"""
import os, argparse, sys
_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--gpu", type=int, default=2)
_a, _ = _p.parse_known_args()
if _a.gpu is not None: os.environ["CUDA_VISIBLE_DEVICES"] = str(_a.gpu)

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from torch_geometric.data import Data
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from benchmark.data.domain_loader import SingleDomainDataset
from benchmark.data.splits import stratified_split, prune_cross_split_edges
from benchmark.models.gnn import MODEL_REGISTRY

SPLIT_SEED=1; EPOCHS=300

def train(model, data, tr, va, te, lr, clip, wd=5e-4, device="cuda"):
    model=model.to(device); data=data.to(device)
    opt=torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    crit=nn.CrossEntropyLoss(); best=-1; best_state=None; pat=0
    for ep in range(1,EPOCHS+1):
        model.train(); opt.zero_grad()
        out=model(data.x,data.edge_index)[tr]; loss=crit(out,data.y[tr]); loss.backward()
        if clip: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
        opt.step()
        if ep%10==0 or ep==EPOCHS:
            model.eval()
            with torch.no_grad():
                pred=model(data.x,data.edge_index).argmax(1)
                va_acc=(pred[va]==data.y[va]).float().mean().item()
            if va_acc>best: best=va_acc; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}; pat=0
            else:
                pat+=10
                if pat>=50: break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pred=model(data.x,data.edge_index).argmax(1)
        te=(pred[te]==data.y[te]).float().mean().item()
    return te

SETTINGS={
 "default(lr.01,do.5,noclip)": dict(lr=0.01,dropout=0.5,clip=None),
 "+gradclip1.0":               dict(lr=0.01,dropout=0.5,clip=1.0),
 "lr.005+clip":                dict(lr=0.005,dropout=0.5,clip=1.0),
 "do.3+lr.005+clip":           dict(lr=0.005,dropout=0.3,clip=1.0),
}

def main():
    dev="cuda" if torch.cuda.is_available() else "cpu"
    cases=[("wikiart","gin","text"),("wikiart","gin","text+image"),
           ("wikiart","gcnii","text+image"),("tmdb","gin","text+image")]
    for dom,mod,modal in cases:
        ds=SingleDomainDataset(dom,modality=modal)
        y=ds.data.y; ei=prune_cross_split_edges(ds.data.edge_index,stratified_split(y,(0.6,0.2,0.2),seed=SPLIT_SEED))
        masks=stratified_split(y,(0.6,0.2,0.2),seed=SPLIT_SEED); tr,va,te=masks
        data=Data(x=ds.data.x,edge_index=ei,y=y); data.num_nodes=ds.data.num_nodes
        out_dim=int(y.max())+1; in_dim=data.x.size(1)
        print(f"\n=== {dom}/{mod}/{modal} (in_dim={in_dim}) ===")
        for name,cfg in SETTINGS.items():
            accs=[]
            for s in range(5):
                torch.manual_seed(s)
                m=MODEL_REGISTRY[mod](in_dim,256,out_dim,num_layers=2,dropout=cfg["dropout"])
                accs.append(train(m,data,tr,va,te,cfg["lr"],cfg["clip"],device=dev))
            print(f"  {name:26s} mean={np.mean(accs):.4f} std={np.std(accs):.4f}  per_seed={[round(a,3) for a in accs]}")

if __name__=="__main__": main()
