#!/usr/bin/env python3
"""定向重跑单个 LP 单元格（多 seed + 崩溃守卫），用于稳定异常单元格。
用法: python benchmark/run_lp_cell.py <model> <modality> <gpu> <n_seeds> <clip[0/1]>
覆盖 lp_ablation_final/lp_<model>_<modality>.json。
"""
import os, sys, json, time
MODEL=sys.argv[1]; MOD=sys.argv[2]; GPU=sys.argv[3]; N=int(sys.argv[4]); CLIP=sys.argv[5]=="1"
LR=float(sys.argv[6]) if len(sys.argv)>6 else 0.01
OUT=os.environ.get("LP_OUT","./benchmark/results/lp_ablation_final")
os.environ["CUDA_VISIBLE_DEVICES"]=GPU
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from loguru import logger
from benchmark.data.dataset import AMMAGDataset
from benchmark.models.gnn import get_model
from benchmark.models.link_predictor import DotProductDecoder
from benchmark.tasks.link_prediction import train_and_eval_link_prediction, negative_sampling_strict

ds=AMMAGDataset(root=".", domains=["cs_si","cs_lo","math_lo","physics_socph"], modality=MOD, label_type="domain")
data=ds.data; in_dim=data.x.size(1); dl=ds._domain_labels.cpu(); ei=data.edge_index.cpu(); ne=ei.size(1)
logger.info(f"=== cell {MODEL}+{MOD} in_dim={in_dim} gpu={GPU} seeds={N} clip={CLIP} lr={LR} ===")
runs=[]
for run in range(N):
    seed=run*42+1
    perm=torch.randperm(ne, generator=torch.Generator().manual_seed(seed))
    tr=int(ne*0.6); va=int(ne*0.2)
    te=ei[:,perm[:tr]]; vp=ei[:,perm[tr:tr+va]]; tp=ei[:,perm[tr+va:]]
    ap=torch.cat([te,vp,tp],dim=1)
    vn=negative_sampling_strict(ap,data.num_nodes,vp.size(1)*10,seed=seed+1000,labels=dl)
    tn=negative_sampling_strict(ap,data.num_nodes,tp.size(1)*10,seed=seed+2000,labels=dl)
    enc=get_model(MODEL,in_dim,256,256,num_layers=2,dropout=0.5,proj_dim=None); dec=DotProductDecoder()
    try:
        r=train_and_eval_link_prediction(enc,dec,data,te,vp,vn,tp,tn,num_epochs=500,lr=LR,device="cuda",labels=dl,clip_grad=CLIP)
        r["seed"]=seed; r["clip_grad"]=CLIP; r["crash"]=r.get("training_time",0)<30
    except Exception as e:
        logger.error(f"seed={seed} EXC {e}"); r={"mrr":0,"hits@1":0,"hits@3":0,"hits@10":0,"training_time":0,"seed":seed,"clip_grad":CLIP,"crash":True}
    runs.append(r); logger.info(f"seed={seed} MRR={r.get('mrr',0):.4f} tt={r.get('training_time',0):.1f}s crash={r.get('crash')}")
healthy=[r for r in runs if not r.get("crash")]; use=healthy if healthy else runs
s={}
for m in ["mrr","hits@1","hits@3","hits@10","training_time"]:
    v=[r[m] for r in use]; s[f"{m}_mean"]=float(np.mean(v)); s[f"{m}_std"]=float(np.std(v))
s.update({"num_runs":len(runs),"num_healthy":len(healthy),"all_crash":len(healthy)==0,"runs":runs,
          "model":MODEL,"modality":MOD,"task":"link_prediction","in_dim":in_dim,"lr":LR,"neg_ratio":10,
          "proj_dim":None,"clip_grad_used":CLIP,"n_seeds":N})
json.dump(s,open(f"{OUT}/lp_{MODEL}_{MOD}.json","w"),indent=2,default=str)
logger.info(f"DONE {MODEL}+{MOD} lr={LR} MRR={s['mrr_mean']:.4f}±{s['mrr_std']:.4f} healthy={len(healthy)}/{N}")
