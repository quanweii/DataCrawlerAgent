#!/usr/bin/env python3
"""
批量重跑所有含Image的实验（Image L2归一化后）
包括：全监督NC、每域消融、Few-shot、链接预测、跨域迁移、子分类
"""
import os, sys, json, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from benchmark.data.dataset import AMMAGDataset
from benchmark.tasks.node_classification import run_node_classification
from benchmark.tasks.link_prediction import run_link_prediction
from benchmark.tasks.cross_domain import run_cross_domain_transfer
from benchmark.models.gnn import get_model
from benchmark.tasks.node_classification import train_and_eval
from loguru import logger

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = {
    "data": {"root": ROOT, "domains": ["cs_si","cs_lo","math_lo","physics_socph"],
             "processed_dir": os.path.join(ROOT, "benchmark", "processed"),
             "results_dir": os.path.join(ROOT, "benchmark", "results")},
    "training": {"hidden_dim": 256, "num_layers": 2, "dropout": 0.5,
                 "lr": 0.01, "num_epochs": 500},
    "experiments": {"node_classification": {"num_runs": 5},
                    "few_shot": {"num_runs": 10, "ks": [3, 10]},
                    "link_prediction": {"num_runs": 5},
                    "cross_domain": {"num_runs": 3, "models": ["gcn"]},
                    "per_domain_ablation": {"num_runs": 5, "model": "gcn"},
                    "subcat_nc": {"num_runs": 5, "models": ["gcn","gat","graphsage"]},
                    "modalities": ["text","image","formula","text+image","text+formula",
                                   "image+formula","text+image+formula"]},
    "models": ["mlp","gcn","gat","graphsage","gin","gt"],
}

results_dir = CONFIG["data"]["results_dir"]
os.makedirs(results_dir, exist_ok=True)
all_results = {}


def get_ds(modality, label_type="domain"):
    return AMMAGDataset(root=CONFIG["data"]["root"], domains=CONFIG["data"]["domains"],
                        modality=modality, label_type=label_type)


# ============================================================
# 1. 全监督节点分类：6模型 × 4个含Image模态
# ============================================================
logger.info("=" * 60)
logger.info("1. 全监督节点分类 (Image-related)")
image_combos = ["image", "text+image", "image+formula", "text+image+formula"]
for modality in image_combos:
    ds = get_ds(modality)
    for model_name in CONFIG["models"]:
        key = f"nc_{model_name}_{modality}"
        logger.info(f"  {key}")
        r = run_node_classification(ds, model_name, modality,
            num_runs=5, hidden_dim=256, num_layers=2, dropout=0.5, lr=0.01,
            num_epochs=500, device=DEVICE, output_dir=results_dir,
            domain_labels=ds._domain_labels)
        all_results[key] = r
        logger.info(f"  -> {r['test_acc_mean']:.2f}%")

# ============================================================
# 2. 每域消融：GCN × 4个含Image模态
# ============================================================
logger.info("=" * 60)
logger.info("2. 每域消融 (Image-related)")
for modality in image_combos:
    ds = get_ds(modality)
    key = f"pda_{modality}"
    logger.info(f"  {key}")
    r = run_node_classification(ds, "gcn", modality,
        num_runs=5, hidden_dim=256, num_layers=2, dropout=0.5, lr=0.01,
        num_epochs=500, device=DEVICE, output_dir=results_dir,
        domain_labels=ds._domain_labels, save_prefix="ablation_pd_v2")
    all_results[key] = r
    logger.info(f"  -> Overall {r['test_acc_mean']:.2f}%")

# ============================================================
# 3. Few-shot：5模型 × T+I+F × 2个k值
# ============================================================
logger.info("=" * 60)
logger.info("3. Few-shot (T+I+F)")
ds_tif = get_ds("text+image+formula")
for model_name in ["gcn","gat","graphsage","gin","gt"]:
    for k in [3, 10]:
        key = f"fs_{model_name}_t+f+i_k{k}"
        logger.info(f"  {key}")
        r = run_node_classification(ds_tif, model_name, "text+image+formula",
            num_runs=10, hidden_dim=256, num_layers=2, dropout=0.5, lr=0.01,
            num_epochs=500, device=DEVICE, output_dir=results_dir,
            split="few_shot", few_shot_k=k, domain_labels=ds_tif._domain_labels)
        all_results[key] = r
        logger.info(f"  -> {r['test_acc_mean']:.2f}%")

# ============================================================
# 4. 链接预测：4模型 × (T+I+F, I+F)
# ============================================================
logger.info("=" * 60)
logger.info("4. 链接预测 (Image-related)")
for modality in ["text+image+formula", "image+formula"]:
    ds = get_ds(modality)
    for model_name in ["gcn","graphsage","gat","gt"]:
        key = f"lp_{model_name}_{modality}"
        logger.info(f"  {key}")
        r = run_link_prediction(ds, model_name, modality,
            num_runs=5, hidden_dim=256, num_layers=2, dropout=0.5, lr=0.01,
            num_epochs=500, device=DEVICE, output_dir=results_dir)
        all_results[key] = r
        logger.info(f"  -> MRR={r['mrr_mean']:.4f}")

# ============================================================
# 5. 跨域迁移：GCN × T+I+F × 12对
# ============================================================
logger.info("=" * 60)
logger.info("5. 跨域迁移 (T+I+F)")
ds_tif2 = get_ds("text+image+formula")
domains = ["cs_si","cs_lo","math_lo","physics_socph"]
for src in domains:
    for tgt in domains:
        if src == tgt:
            continue
        key = f"cd_tif_{src}_to_{tgt}"
        logger.info(f"  {key}")
        r = run_cross_domain_transfer(ds_tif2, "gcn", "text+image+formula",
            source_domain=src, target_domain=tgt,
            num_runs=3, hidden_dim=256, num_layers=2, dropout=0.5, lr=0.01,
            num_epochs=500, device=DEVICE, output_dir=results_dir)
        all_results[key] = r
        logger.info(f"  -> gain={r.get('transfer_gain_mean',0):+.2f}pp")

# ============================================================
# 6. 子分类：4域 × 3模型 × T+I+F
# ============================================================
logger.info("=" * 60)
logger.info("6. 子分类 (T+I+F)")
ds_sub = get_ds("text+image+formula", label_type="subcat")
for domain in domains:
    subgraph = ds_sub.get_domain_subgraph(domain)
    in_dim = subgraph.x.size(1)
    out_dim = int(subgraph.y.max().item()) + 1
    num_nodes = subgraph.num_nodes
    for model_name in ["gcn","gat","graphsage"]:
        run_res = []
        for run in range(5):
            seed = run * 42 + 1
            rng = np.random.RandomState(seed)
            perm = rng.permutation(num_nodes)
            t, v = int(num_nodes*0.6), int(num_nodes*0.2)
            tm = torch.zeros(num_nodes, dtype=torch.bool)
            vm = torch.zeros(num_nodes, dtype=torch.bool)
            tsm = torch.zeros(num_nodes, dtype=torch.bool)
            tm[perm[:t]] = True; vm[perm[t:t+v]] = True; tsm[perm[t+v:]] = True
            model = get_model(model_name, in_dim, 256, out_dim, 2, 0.5)
            res = train_and_eval(model, subgraph, tm, vm, tsm, 500, 0.01, DEVICE)
            run_res.append(res)
        key = f"subcat_{domain}_{model_name}_t+f+i"
        acc_mean = np.mean([r['test_acc'] for r in run_res])
        acc_std = np.std([r['test_acc'] for r in run_res])
        all_results[key] = {"test_acc_mean": acc_mean, "test_acc_std": acc_std}
        logger.info(f"  {key}: {acc_mean:.2f}%")

# ============================================================
# 保存汇总
# ============================================================
save_path = os.path.join(results_dir, "v2_image_normalized_all.json")
with open(save_path, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
logger.info(f"All results saved to {save_path}")
logger.info("DONE!")
