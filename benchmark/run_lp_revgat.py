#!/usr/bin/env python3
"""RevGAT 专项：关闭梯度裁剪（可逆架构需完整梯度），5 seed，丢弃发散run。
用法: python benchmark/run_lp_revgat.py <modality> <gpu> [outdir]
覆盖 lp_ablation_final/lp_revgat_<modality>.json。
"""
import os, sys, json, time
MODALITY = sys.argv[1]
GPU = sys.argv[2] if len(sys.argv) > 2 else "0"
OUTDIR = sys.argv[3] if len(sys.argv) > 3 else "./benchmark/results/lp_ablation_final"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from loguru import logger
from benchmark.data.dataset import AMMAGDataset
from benchmark.models.gnn import get_model
from benchmark.models.link_predictor import DotProductDecoder
from benchmark.tasks.link_prediction import train_and_eval_link_prediction, negative_sampling_strict

DOMAINS = ["cs_si","cs_lo","math_lo","physics_socph"]
N_SEEDS = 5
HEALTHY_MIN_TIME = 30.0

def run_seeds(data, in_dim, dl, ei, ne, n_seeds=N_SEEDS, clip=False, lr=0.01):
    runs=[]
    for run in range(n_seeds):
        seed=run*42+1
        perm=torch.randperm(ne, generator=torch.Generator().manual_seed(seed))
        tr=int(ne*0.6); va=int(ne*0.2)
        te=ei[:,perm[:tr]]; vp=ei[:,perm[tr:tr+va]]; tp=ei[:,perm[tr+va:]]
        ap=torch.cat([te,vp,tp],dim=1)
        vn=negative_sampling_strict(ap,data.num_nodes,vp.size(1)*10,seed=seed+1000,labels=dl)
        tn=negative_sampling_strict(ap,data.num_nodes,tp.size(1)*10,seed=seed+2000,labels=dl)
        enc=get_model("revgat",in_dim,256,256,num_layers=2,dropout=0.5,proj_dim=None)
        dec=DotProductDecoder()
        try:
            r=train_and_eval_link_prediction(enc,dec,data,te,vp,vn,tp,tn,num_epochs=500,lr=lr,device="cuda",labels=dl,clip_grad=clip)
            r["seed"]=seed; r["clip_grad"]=clip; r["crash"]=r.get("training_time",0)<HEALTHY_MIN_TIME
        except Exception as e:
            logger.error(f"[revgat+{MODALITY}] seed={seed} clip={clip} EXC {e}")
            r={"mrr":0,"hits@1":0,"hits@3":0,"hits@10":0,"training_time":0,"seed":seed,"clip_grad":clip,"crash":True,"error":str(e)[:150]}
        runs.append(r)
        logger.info(f"[revgat+{MODALITY}] seed={seed} clip={clip} MRR={r.get('mrr',0):.4f} tt={r.get('training_time',0):.1f}s crash={r.get('crash')}")
    return runs

def main():
    os.makedirs(OUTDIR,exist_ok=True)
    ds=AMMAGDataset(root=".",domains=DOMAINS,modality=MODALITY,label_type="domain")
    data=ds.data; in_dim=data.x.size(1); dl=ds._domain_labels.cpu(); ei=data.edge_index.cpu(); ne=ei.size(1)
    logger.info(f"=== revgat no-clip: {MODALITY} in_dim={in_dim} gpu={GPU} ===")
    runs=run_seeds(data,in_dim,dl,ei,ne,clip=False)
    healthy=[r for r in runs if not r.get("crash")]
    # 若健康不足4，再补5 seed
    if len(healthy)<4:
        logger.warning(f"[revgat+{MODALITY}] only {len(healthy)}/{N_SEEDS} healthy, +5 more seeds")
        runs += run_seeds(data,in_dim,dl,ei,ne,clip=False)
        healthy=[r for r in runs if not r.get("crash")]
    use=healthy if healthy else runs
    s={}
    for m in ["mrr","hits@1","hits@3","hits@10","training_time"]:
        v=[r[m] for r in use]; s[f"{m}_mean"]=float(np.mean(v)); s[f"{m}_std"]=float(np.std(v))
    s.update({"num_runs":len(runs),"num_healthy":len(healthy),"all_crash":len(healthy)==0,"runs":runs,
              "model":"revgat","modality":MODALITY,"task":"link_prediction","in_dim":in_dim,
              "lr":0.01,"neg_ratio":10,"proj_dim":None,"clip_grad_used":False,"n_seeds":len(runs)})
    out=os.path.join(OUTDIR,f"lp_revgat_{MODALITY}.json")
    json.dump(s,open(out,"w"),indent=2,default=str)
    logger.info(f"[revgat+{MODALITY}] DONE MRR={s['mrr_mean']:.4f}±{s['mrr_std']:.4f} healthy={len(healthy)}/{len(runs)} clip=OFF -> {out}")

if __name__=="__main__":
    main()
