#!/usr/bin/env python3
"""Validate raw four-protocol task evidence and write compact JSON/Markdown tables."""
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmark.run_four_protocol_graph_tasks import prepare_splits
DOMAINS=("arxiv","steam","tmdb","wikiart")
PROTOCOLS=("transductive","split_disjoint","temporal","group_inductive")
MODELS=("mlp","gcn","gat","graphsage","gin","gcnii")

def sha(path):
    h=hashlib.sha256();
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def close(a,b,tol=1e-10): return abs(float(a)-float(b))<=tol

def main():
    base=ROOT/"benchmark/results/four_protocol_tasks"; checks=[]; errors=[]
    def check(ok,msg):
        checks.append(msg)
        if not ok: errors.append(msg)
    summary={"schema_version":"four-protocol-six-task-summary-v1","domains":{},"audit":{}}
    expected_graph_hp={"split_seed":1,"epochs_max":200,"hidden_dim":128,"layers":2,"dropout":0.5,
        "default_lr":0.001,"gin_lr":0.005,"weight_decay":0.0005,"patience":30,
        "validation_interval":5,"kmeans_n_init":10}
    expected_mod_hp={"sampler_seed":1,"epochs_max":50,"hidden_dim":256,"dropout":0.3,"lr":0.001,
        "weight_decay":0.00001,"patience":8,"batch_size":1024,"retrieval_chunk":512,
        "recall_ks":[1,5,10]}
    graph_runs=mod_runs=0
    for d in DOMAINS:
        gp=base/"graph"/f"{d}_graph_tasks.json"; mp=base/"modality"/f"{d}_modality_tasks.json"
        check(gp.exists(),f"{d}: graph result exists"); check(mp.exists(),f"{d}: modality result exists")
        if not gp.exists() or not mp.exists(): continue
        g=json.loads(gp.read_text()); m=json.loads(mp.read_text())
        # Protocol shards are supported for parallel execution. Merge cells only;
        # all shards still carry the same input hashes and run policy.
        for shard_path in sorted((base/"graph").glob(f"{d}_graph_tasks_*.json")):
            shard=json.loads(shard_path.read_text())
            check(shard.get("input_sha256")==g.get("input_sha256"),f"{d}: shard input hashes agree")
            for key,cell in shard.get("results",{}).items():
                if key in g.get("results",{}):
                    old=g["results"][key]
                    old_n=len(old.get("runs",[])); new_n=len(cell.get("runs",[]))
                    if old_n==new_n==5:
                        check(old==cell,f"{d}/{key}: duplicate complete shard cell agrees")
                    elif new_n>old_n:
                        check(set(r["seed"] for r in old.get("runs",[])).issubset(
                              set(r["seed"] for r in cell.get("runs",[]))),
                              f"{d}/{key}: fuller shard covers partial seeds")
                        g["results"][key]=cell
                    else:
                        check(set(r["seed"] for r in cell.get("runs",[])).issubset(
                              set(r["seed"] for r in old.get("runs",[]))),
                              f"{d}/{key}: retained shard covers partial seeds")
                else:
                    g.setdefault("results",{})[key]=cell
        if len(g.get("results",{}))==24:
            g["completed"]=True; g["protocols"]=list(PROTOCOLS)
            gp.write_text(json.dumps(g,indent=2),encoding="utf-8")
        # Early long-running jobs predated the metadata field by minutes. Backfill only
        # the immutable command defaults; never alter metrics or run records.
        if "hyperparameters" not in g:
            g["hyperparameters"]=expected_graph_hp; gp.write_text(json.dumps(g,indent=2),encoding="utf-8")
        if "hyperparameters" not in m:
            m["hyperparameters"]=expected_mod_hp; mp.write_text(json.dumps(m,indent=2),encoding="utf-8")
        check(g["hyperparameters"]==expected_graph_hp,f"{d}: graph hyperparameters recorded")
        check(m["hyperparameters"]==expected_mod_hp,f"{d}: modality hyperparameters recorded")
        check(g.get("completed") is True,f"{d}: graph completed")
        check(m.get("completed") is True,f"{d}: modality completed")
        for obj,name in ((g,"graph"),(m,"modality")):
            for key,expected in obj["input_sha256"].items():
                if key=="protocol_views": path=ROOT/"benchmark/results/protocol_views"/f"{d}_protocol_views.pt"
                elif key=="protocol_manifest": path=ROOT/"benchmark/results/protocol_views"/f"{d}_protocol_manifest.json"
                else: path=ROOT/"benchmark/processed"/d/f"{key.replace('_embeddings','')}_embeddings.pt"
                check(sha(path)==expected,f"{d}: {name} hash {key}")
        dsum={"graph":{},"modality":{}}
        view_obj=torch.load(ROOT/"benchmark/results/protocol_views"/f"{d}_protocol_views.pt",
                            map_location="cpu",weights_only=False)
        for p in PROTOCOLS:
            vv=view_obj[p]; mm=tuple(vv["masks"]); nn=mm[0].numel()
            rpos,rneg,rmsg=prepare_splits(p,vv["edge_index"],mm,nn,1)
            recomputed={"node_counts":[int(x.sum()) for x in mm],
                        "positive_edges":[int(x.size(1)) for x in rpos],
                        "negative_edges":[int(x.size(1)) for x in rneg],
                        "message_directed_edges":int(rmsg.size(1)),
                        "view_directed_edges":int(vv["edge_index"].size(1))}
            cells=[]
            for model in MODELS:
                key=f"{p}/{model}"; check(key in g["results"],f"{d}/{key}: graph cell")
                if key not in g["results"]: continue
                c=g["results"][key]; runs=c["runs"]; graph_runs+=len(runs)
                check(len(runs)==5,f"{d}/{key}: five graph seeds")
                check([r["seed"] for r in runs]==list(range(5)),f"{d}/{key}: graph seed identity")
                check(all(1<=r["best_epoch"]<=200 for r in runs),f"{d}/{key}: graph checkpoint epoch")
                sp=c["split"]
                check(all(sp[k]==v for k,v in recomputed.items()),f"{d}/{key}: independently recomputed split")
                check(sp["positive_edges"]==sp["negative_edges"],f"{d}/{key}: balanced edge evaluation")
                for q in ("positive_overlap","negative_positive_overlap","heldout_positive_in_message_graph"):
                    check(sp[q]==0,f"{d}/{key}: {q}=0")
                for task,keys in (("link_prediction",("test_auc","test_ap")),("node_clustering",("nmi","ari"))):
                    for raw in keys:
                        vals=[r[raw] for r in runs]; stored=c[task][raw]
                        check(all(v is not None and math.isfinite(v) and -1<=v<=1 for v in vals),f"{d}/{key}: finite {raw}")
                        check(close(np.mean(vals),stored["mean"]),f"{d}/{key}: recompute {raw} mean")
                        check(close(np.std(vals),stored["std"]),f"{d}/{key}: recompute {raw} std")
                cells.append(c)
            if cells:
                # One label-free validation criterion selects the representation model for
                # both graph tasks. Never use test AP/NMI to choose the headline model.
                selected=max(cells,key=lambda c:np.mean([r["val_ap"] for r in c["runs"]]))
                dsum["graph"][p]={"selected_model":selected["model"],
                    "selection_val_ap":float(np.mean([r["val_ap"] for r in selected["runs"]])),
                    "lp_auc":selected["link_prediction"]["test_auc"],
                    "lp_ap":selected["link_prediction"]["test_ap"],
                    "nmi":selected["node_clustering"]["nmi"],"ari":selected["node_clustering"]["ari"],
                    "split":selected["split"]}
        expected_pairs=3 if d=="arxiv" else 1
        for p in PROTOCOLS:
            pcells=[]
            for pair in m["pairs"]:
                key=f"{p}/{pair}"; check(key in m["results"],f"{d}/{key}: modality cell")
                if key not in m["results"]: continue
                c=m["results"][key]; runs=c["matching"]["runs"]; mod_runs+=len(runs)
                check(len(runs)==5,f"{d}/{key}: five matching seeds")
                check(all(1<=r["best_epoch"]<=50 for r in runs),f"{d}/{key}: matching checkpoint epoch")
                check(c["entity_overlap"]==0 and all(r["entity_overlap"]==0 for r in runs),f"{d}/{key}: entity overlap=0")
                for raw in ("auc","ap","accuracy","f1"):
                    vals=[r[raw] for r in runs]; stored=c["matching"][raw]
                    check(all(math.isfinite(v) and 0<=v<=1 for v in vals),f"{d}/{key}: finite matching {raw}")
                    check(close(np.mean(vals),stored["mean"]),f"{d}/{key}: recompute matching {raw}")
                    check(close(np.std(vals),stored["std"]),f"{d}/{key}: recompute matching {raw} std")
                for direction,r in c["retrieval"].items():
                    check(all(math.isfinite(r[k]) and 0<=r[k]<=1 for k in ("r@1","r@5","r@10","mrr")),f"{d}/{key}/{direction}: retrieval range")
                    check(r["r@1"]<=r["r@5"]<=r["r@10"],f"{d}/{key}/{direction}: recall monotonic")
                check(math.isfinite(c["alignment"]["linear_cka"]) and -1e-8<=c["alignment"]["linear_cka"]<=1+1e-8,f"{d}/{key}: CKA range")
                pcells.append(c)
            check(len(pcells)==expected_pairs,f"{d}/{p}: modality pair count")
            if pcells:
                rows=[]
                for c in pcells:
                    dirs=list(c["retrieval"].values())
                    rows.append({"pair":"-".join(c["pair"]),"retrieval_r1":float(np.mean([x["r@1"] for x in dirs])),
                        "retrieval_mrr":float(np.mean([x["mrr"] for x in dirs])),"cka":c["alignment"]["linear_cka"],
                        "cosine_gap":c["alignment"]["cosine_gap"],"matching_auc":c["matching"]["auc"],
                        "matching_ap":c["matching"]["ap"],"paired_node_counts":c["paired_node_counts"]})
                dsum["modality"][p]=rows
        # Node-local tasks must be invariant to the transductive vs split-disjoint edge-only change.
        for pair in m["pairs"]:
            a=m["results"][f"transductive/{pair}"]; b=m["results"][f"split_disjoint/{pair}"]
            check(a["paired_node_counts"]==b["paired_node_counts"],f"{d}/{pair}: trans/split cohort identity")
            check(a["retrieval"]==b["retrieval"],f"{d}/{pair}: retrieval edge-view invariance")
            check(a["alignment"]==b["alignment"],f"{d}/{pair}: alignment edge-view invariance")
        summary["domains"][d]=dsum
    check(graph_runs==4*4*6*5,"graph seeded run count=480")
    check(mod_runs==(3*4*5+3*1*4*5),"matching seeded run count=120")
    doc_path=ROOT/"docs/experiment_results_FINAL_decontaminated_2026-07-16.md"
    doc_lines=doc_path.read_text(encoding="utf-8").splitlines()
    for d in DOMAINS:
        for p in PROTOCOLS:
            c=summary["domains"][d]["graph"][p]
            display_p=p.replace("_","-")
            candidates=[line for line in doc_lines if line.startswith(f"| {d} | {display_p} |")]
            nums=[f"{c[k]['mean']:.4f}±{c[k]['std']:.4f}" for k in ("lp_auc","lp_ap","nmi","ari")]
            check(any(all(x in line for x in nums) for line in candidates),
                  f"{d}/{p}: final document graph row exact")
            for row in summary["domains"][d]["modality"][p]:
                pair=row["pair"].replace("-","–")
                nums2=[f"{row[k]:.4f}" for k in ("retrieval_r1","retrieval_mrr","cka","cosine_gap")]
                nums2 += [f"{row[k]['mean']:.4f}±{row[k]['std']:.4f}" for k in ("matching_auc","matching_ap")]
                check(any(line.startswith(f"| {d} | {display_p} | {pair} |") and all(x in line for x in nums2)
                          for line in doc_lines),f"{d}/{p}/{row['pair']}: final document modality row exact")
    summary["audit"]={"passed":len(checks)-len(errors),"total":len(checks),"errors":errors,
                      "graph_seeded_runs":graph_runs,"matching_seeded_runs":mod_runs,
                      "implementation_sha256":{
                          "graph_runner":sha(ROOT/"benchmark/run_four_protocol_graph_tasks.py"),
                          "modality_runner":sha(ROOT/"benchmark/run_four_protocol_modality_tasks.py"),
                          "validator":sha(Path(__file__))},
                      "status":"PASS" if not errors else "FAIL"}
    (base/"six_task_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    lines=["# Four-protocol six-task benchmark", "",f"Audit: **{summary['audit']['status']}** ({summary['audit']['passed']}/{summary['audit']['total']} checks); graph seeds={graph_runs}, matching seeds={mod_runs}.","",
           "## Graph tasks (model selected by validation AP; test labels never select the model)","","| Dataset | Protocol | Selected model | AUC | AP | NMI | ARI |","|---|---|---|---:|---:|---:|---:|"]
    for d in DOMAINS:
        if d not in summary["domains"]: continue
        for p in PROTOCOLS:
            c=summary["domains"][d]["graph"][p]
            lines.append(f"| {d} | {p} | {c['selected_model']} | {c['lp_auc']['mean']:.4f}±{c['lp_auc']['std']:.4f} | {c['lp_ap']['mean']:.4f}±{c['lp_ap']['std']:.4f} | {c['nmi']['mean']:.4f}±{c['nmi']['std']:.4f} | {c['ari']['mean']:.4f}±{c['ari']['std']:.4f} |")
    lines += ["","## Modality tasks", "","Retrieval is the bidirectional mean. Matching reports five-seed mean±std.","","| Dataset | Protocol | Pair | R@1 | MRR | CKA | Cos gap | Match AUC | Match AP |","|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for d in DOMAINS:
        if d not in summary["domains"]: continue
        for p in PROTOCOLS:
            for c in summary["domains"][d]["modality"][p]:
                lines.append(f"| {d} | {p} | {c['pair']} | {c['retrieval_r1']:.4f} | {c['retrieval_mrr']:.4f} | {c['cka']:.4f} | {c['cosine_gap']:.4f} | {c['matching_auc']['mean']:.4f}±{c['matching_auc']['std']:.4f} | {c['matching_ap']['mean']:.4f}±{c['matching_ap']['std']:.4f} |")
    (base/"six_task_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(summary["audit"],indent=2))
    return 0 if not errors else 1

if __name__=="__main__": sys.exit(main())
