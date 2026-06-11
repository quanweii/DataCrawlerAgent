"""
OpenMAG实验主脚本 - 完整复现论文第四章全部实验 (Q1-Q10)
所有模型统一接收768d输入，多模态特征通过可学习投影层降维
"""
import os
import sys
import json
import argparse
import time
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loguru import logger
from datetime import datetime
from sklearn.metrics import f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "benchmark", "results", "openmag")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "benchmark", "processed")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "benchmark", "configs", "openmag.yaml")

os.makedirs(RESULTS_DIR, exist_ok=True)

PROJECT_DIM = 768  # 统一投影维度

ALL_MODELS = ["mlp", "gcn", "gat", "graphsage", "gin", "gt", "revgat",
              "dgf", "mmgcn", "mgat", "lgmrec", "mhgat", "dmgc"]
EFFECTIVENESS_MODELS = ["mlp", "gcn", "gat", "dgf", "mmgcn", "mgat", "lgmrec", "dmgc", "mhgat"]
EXTENDED_MODELS = EFFECTIVENESS_MODELS + ["gin", "gcnii", "gatv2"]
ROBUSTNESS_MODELS = ["gat", "graphsage", "dgf", "mmgcn", "mgat", "lgmrec"]
NECESSITY_MODALITIES = ["text", "image", "text+image+formula"]


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    return {"training": {"hidden_dim": 256, "num_layers": 2, "dropout": 0.5,
            "lr": 0.001, "weight_decay": 0.0005, "num_epochs": 500, "patience": 25}}

def save_result(name, data):
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"保存: {path}")

def get_dataset(modality="text", label_type="domain"):
    from benchmark.data.dataset import AMMAGDataset
    return AMMAGDataset(root=PROJECT_ROOT, modality=modality, label_type=label_type)


# ==================== 模型创建（统一768d投影） ====================

def get_model(model_name, in_dim, num_classes, config):
    """创建模型，多模态输入自动投影到768d"""
    from benchmark.models.gnn import get_model as _get_model
    t = config["training"]
    return _get_model(model_name, in_dim, t["hidden_dim"], num_classes,
                      num_layers=t["num_layers"], dropout=t["dropout"], proj_dim=PROJECT_DIM)

def make_projected_cls(base_cls, proj_dim=PROJECT_DIM):
    """工厂：包装任意模型类，创建时自动投影到proj_dim"""
    from benchmark.models.gnn import ProjectedModel
    def factory(**kwargs):
        raw_dim = kwargs.get('in_dim', proj_dim)
        base_kwargs = {**kwargs, 'in_dim': proj_dim}
        base = base_cls(**base_kwargs)
        if raw_dim != proj_dim:
            return ProjectedModel(base, raw_dim, proj_dim)
        return base
    return factory


def train_and_eval(model, data, device="cuda", epochs=500, lr=0.001,
                   patience=25, weight_decay=0.0005):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)

    best_val_acc, best_state, p_counter = 0, None, 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = model(x, edge_index).argmax(dim=1)
            val_acc = (pred[val_mask] == y[val_mask]).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            p_counter = 0
        else:
            p_counter += 1
        if p_counter >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(x, edge_index).argmax(dim=1)
        test_acc = (pred[test_mask] == y[test_mask]).float().mean().item()
        test_f1 = f1_score(y[test_mask].cpu().numpy(), pred[test_mask].cpu().numpy(), average="macro")
    return {"test_acc": test_acc, "test_f1": test_f1}


# ==================== Q1: 单模态 vs 多模态 ====================

def run_q1_necessity(config, gpu=0):
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    num_runs = config.get("openmag", {}).get("num_runs", 3)
    models = ["mlp", "gcn", "gat", "graphsage"]
    all_results = []

    for model_name in models:
        for modality in NECESSITY_MODALITIES:
            dataset = get_dataset(modality)
            in_dim = dataset.data.x.size(1)
            num_classes = dataset.data.y.max().item() + 1
            accs, f1s = [], []
            for run in range(num_runs):
                torch.manual_seed(run * 42 + 1); np.random.seed(run * 42 + 1)
                dataset.data.train_mask, dataset.data.val_mask, dataset.data.test_mask = dataset.get_split_masks(seed=run*42+1)
                model = get_model(model_name, in_dim, num_classes, config)
                res = train_and_eval(model, dataset.data, device, epochs=config["training"]["num_epochs"],
                                     lr=config["training"]["lr"], patience=config["training"]["patience"])
                accs.append(res["test_acc"]); f1s.append(res["test_f1"])
            r = {"model": model_name, "modality": modality, "task": "NC",
                 "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                 "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s))}
            all_results.append(r)
            logger.info(f"Q1 NC | {model_name} | {modality}: acc={r['acc_mean']:.4f}±{r['acc_std']:.4f}")

    from benchmark.tasks.link_prediction import run_link_prediction
    for modality in NECESSITY_MODALITIES:
        dataset = get_dataset(modality)
        for model_name in ["mlp", "gcn", "gat"]:
            lp_res = run_link_prediction(dataset=dataset, model_name=model_name, modality=modality,
                                         num_runs=num_runs, num_epochs=config["training"]["num_epochs"],
                                         hidden_dim=config["training"]["hidden_dim"],
                                         num_layers=config["training"]["num_layers"],
                                         dropout=config["training"]["dropout"],
                                         lr=config["training"]["lr"], device=device, output_dir=RESULTS_DIR,
                                         proj_dim=PROJECT_DIM)
            r = {"model": model_name, "modality": modality, "task": "LP",
                 "mrr_mean": lp_res.get("mrr_mean", 0), "mrr_std": lp_res.get("mrr_std", 0),
                 "hits3_mean": lp_res.get("hits@3_mean", 0), "hits3_std": lp_res.get("hits@3_std", 0)}
            all_results.append(r)
            logger.info(f"Q1 LP | {model_name} | {modality}: MRR={r['mrr_mean']:.4f}")

    save_result("q1_necessity", {"results": all_results})
    return all_results


# ==================== Q2: 邻居数消融 ====================

def run_q2_neighbors(config, gpu=0):
    from benchmark.tasks.neighbor_ablation import run_neighbor_ablation
    from benchmark.models.gnn import GCN
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    num_runs = config.get("openmag", {}).get("num_runs", 3)
    dataset = get_dataset("text+image+formula")
    in_dim = dataset.data.x.size(1)
    num_classes = dataset.data.y.max().item() + 1
    dataset.data.train_mask, dataset.data.val_mask, dataset.data.test_mask = dataset.get_split_masks()

    projected_gcn = make_projected_cls(GCN)
    results = run_neighbor_ablation(
        projected_gcn, {"in_dim": in_dim, "hidden_dim": config["training"]["hidden_dim"],
                        "out_dim": num_classes, "num_layers": config["training"]["num_layers"],
                        "dropout": config["training"]["dropout"]},
        dataset.data, device=device, neighbor_counts=[0, 2, 4, 6, 8],
        num_runs=num_runs, epochs=config["training"]["num_epochs"], patience=config["training"]["patience"])
    save_result("q2_neighbor_ablation", results)
    return results


# ==================== Q3: 微调对比 ====================

def run_q3_finetune(config, gpu=0):
    from benchmark.tasks.finetune_comparison import run_finetune_comparison
    from benchmark.models.gnn import MLP
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    num_runs = config.get("openmag", {}).get("num_runs", 3)
    dataset = get_dataset("text+image")
    in_dim = dataset.data.x.size(1)
    num_classes = dataset.data.y.max().item() + 1
    dataset.data.train_mask, dataset.data.val_mask, dataset.data.test_mask = dataset.get_split_masks()

    projected_mlp = make_projected_cls(MLP)
    results = run_finetune_comparison(
        projected_mlp, {"in_dim": in_dim, "hidden_dim": config["training"]["hidden_dim"],
                        "out_dim": num_classes, "num_layers": 2, "dropout": config["training"]["dropout"]},
        dataset.data, device=device, num_runs=num_runs,
        epochs=config["training"]["num_epochs"], patience=config["training"]["patience"], dim=768)
    save_result("q3_finetune", {"results": results})
    return results


# ==================== Q4: 编码策略对比 ====================

def run_q4_encoding(config, gpu=0):
    from benchmark.tasks.encoding_comparison import run_encoding_comparison
    text_768 = torch.load(os.path.join(PROCESSED_DIR, "text_embeddings.pt"), map_location="cpu")["embeddings"]
    image_768 = torch.load(os.path.join(PROCESSED_DIR, "image_embeddings.pt"), map_location="cpu")["embeddings"]

    all_results = {}

    unified = run_encoding_comparison(text_768, image_768, sample_size=5000)
    if "unified" in unified:
        unified["unified"]["encoder"] = "CLIP-Large (Unified 768d)"
        all_results["clip_large_unified"] = unified["unified"]

    text_384_path = os.path.join(PROCESSED_DIR, "text_embeddings_384d.pt")
    image_512_path = os.path.join(PROCESSED_DIR, "image_embeddings_512d.pt")
    if os.path.exists(text_384_path) and os.path.exists(image_512_path):
        text_384 = torch.load(text_384_path, map_location="cpu")["embeddings"]
        image_512 = torch.load(image_512_path, map_location="cpu")["embeddings"]
        ind = run_encoding_comparison(text_768, image_768, text_384, image_512, sample_size=5000)
        if "independent" in ind:
            ind["independent"]["encoder"] = "SBERT(384d) + CLIP-Base(512d) [Independent]"
            all_results["sbert_clipbase"] = ind["independent"]

    specter_path = os.path.join(PROCESSED_DIR, "specter_text_embeddings.pt")
    if os.path.exists(specter_path):
        specter = torch.load(specter_path, map_location="cpu")["embeddings"]
        sp = run_encoding_comparison(specter, image_768, sample_size=5000)
        if "unified" in sp:
            sp["unified"]["encoder"] = "SPECTER(768d) + CLIP-Large(768d)"
            all_results["specter_clip"] = sp["unified"]

    save_result("q4_encoding_comparison", all_results)
    return all_results


# ==================== Q5: 有效性 ====================

def run_q5_effectiveness(config, gpu=0):
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    num_runs = config.get("openmag", {}).get("num_runs", 3)
    modality = "text+image+formula"

    nc_results = []
    dataset = get_dataset(modality)
    in_dim = dataset.data.x.size(1)
    num_classes = dataset.data.y.max().item() + 1
    for model_name in EFFECTIVENESS_MODELS:
        logger.info(f"Q5 NC | {model_name}")
        accs, f1s = [], []
        for run in range(num_runs):
            torch.manual_seed(run * 42 + 1); np.random.seed(run * 42 + 1)
            dataset.data.train_mask, dataset.data.val_mask, dataset.data.test_mask = dataset.get_split_masks(seed=run*42+1)
            model = get_model(model_name, in_dim, num_classes, config)
            res = train_and_eval(model, dataset.data, device, epochs=config["training"]["num_epochs"],
                                 lr=config["training"]["lr"], patience=config["training"]["patience"])
            accs.append(res["test_acc"]); f1s.append(res["test_f1"])
        nc_results.append({"model": model_name, "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                           "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s))})
        logger.info(f"  -> acc={np.mean(accs):.4f}±{np.std(accs):.4f}, f1={np.mean(f1s):.4f}±{np.std(f1s):.4f}")

    from benchmark.tasks.link_prediction import run_link_prediction
    lp_results = []
    lp_models = [m for m in ["mlp", "gcn", "gat", "dgf", "lgmrec"] if m in EFFECTIVENESS_MODELS]
    for model_name in lp_models:
        logger.info(f"Q5 LP | {model_name}")
        dataset_lp = get_dataset(modality)
        lp_res = run_link_prediction(dataset=dataset_lp, model_name=model_name, modality=modality,
                                     num_runs=num_runs, hidden_dim=config["training"]["hidden_dim"],
                                     num_layers=config["training"]["num_layers"],
                                     dropout=config["training"]["dropout"], lr=config["training"]["lr"],
                                     device=device, output_dir=RESULTS_DIR, proj_dim=PROJECT_DIM)
        lp_results.append({"model": model_name, "mrr_mean": lp_res.get("mrr_mean", 0),
                           "mrr_std": lp_res.get("mrr_std", 0),
                           "hits3_mean": lp_res.get("hits@3_mean", 0), "hits3_std": lp_res.get("hits@3_std", 0)})

    from benchmark.tasks.node_clustering import run_clustering_experiment
    cluster_results = []
    for model_name in EFFECTIVENESS_MODELS:
        logger.info(f"Q5 CL | {model_name}")
        dataset_cl = get_dataset(modality)
        in_dim_cl = dataset_cl.data.x.size(1)
        model = get_model(model_name, in_dim_cl, num_classes, config)
        cl_res = run_clustering_experiment(model_name, model, dataset_cl.data, num_classes,
                                           num_runs=num_runs, device=device, dataset=dataset_cl)
        cluster_results.append(cl_res)
        logger.info(f"  -> NMI={cl_res['nmi_mean']:.4f}, ARI={cl_res['ari_mean']:.4f}")

    all_results = {"node_classification": nc_results, "link_prediction": lp_results, "node_clustering": cluster_results}
    save_result("q5_effectiveness", all_results)
    return all_results


# ==================== Q6: PEFT策略 ====================

def run_q6_peft(config, gpu=0):
    from benchmark.tasks.peft_comparison import run_peft_comparison
    from benchmark.models.gnn import GCN
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    num_runs = config.get("openmag", {}).get("num_runs", 3)
    dataset = get_dataset("text+image+formula")
    in_dim = dataset.data.x.size(1)
    num_classes = dataset.data.y.max().item() + 1
    dataset.data.train_mask, dataset.data.val_mask, dataset.data.test_mask = dataset.get_split_masks()

    # GCN always takes 768d; PEFT operates on 768d; projection wraps everything
    results = run_peft_comparison(
        GCN, {"in_dim": PROJECT_DIM, "hidden_dim": config["training"]["hidden_dim"],
              "out_dim": num_classes, "num_layers": config["training"]["num_layers"],
              "dropout": config["training"]["dropout"]},
        dataset.data, device=device, num_runs=num_runs,
        epochs=config["training"]["num_epochs"], patience=config["training"]["patience"],
        dim=PROJECT_DIM, raw_dim=in_dim)
    save_result("q6_peft", {"results": results})
    return results


# ==================== Q7: 特征/标签噪声 ====================

def run_q7_robustness(config, gpu=0):
    from benchmark.tasks.noise_experiments import run_noise_experiment
    from benchmark.models.gnn import MODEL_REGISTRY
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    num_runs = config.get("openmag", {}).get("num_runs", 3)
    modality = "text+image+formula"
    dataset = get_dataset(modality)
    in_dim = dataset.data.x.size(1)
    num_classes = dataset.data.y.max().item() + 1
    dataset.data.train_mask, dataset.data.val_mask, dataset.data.test_mask = dataset.get_split_masks()

    all_results = {}
    for model_name in ROBUSTNESS_MODELS:
        all_results[model_name] = {}
        cls = MODEL_REGISTRY[model_name]
        projected_cls = make_projected_cls(cls)
        model_params = {"in_dim": in_dim, "hidden_dim": config["training"]["hidden_dim"],
                        "out_dim": num_classes, "num_layers": config["training"]["num_layers"],
                        "dropout": config["training"]["dropout"]}
        for noise_type in ["text", "image", "label"]:
            logger.info(f"Q7 | {model_name} | {noise_type} noise")
            result = run_noise_experiment(
                train_and_eval, dataset.data, projected_cls, model_params,
                noise_type=noise_type,
                noise_ratios=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                num_runs=num_runs, device=device,
                epochs=config["training"]["num_epochs"], lr=config["training"]["lr"],
                patience=config["training"]["patience"])
            all_results[model_name][noise_type] = result

    save_result("q7_robustness", all_results)
    return all_results


# ==================== Q8: 边噪声 ====================

def run_q8_edge_noise(config, gpu=0):
    from benchmark.tasks.noise_experiments import run_noise_experiment
    from benchmark.models.gnn import MODEL_REGISTRY
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    num_runs = config.get("openmag", {}).get("num_runs", 3)
    dataset = get_dataset("text+image+formula")
    in_dim = dataset.data.x.size(1)
    num_classes = dataset.data.y.max().item() + 1
    dataset.data.train_mask, dataset.data.val_mask, dataset.data.test_mask = dataset.get_split_masks()

    all_results = {}
    for model_name in ROBUSTNESS_MODELS:
        logger.info(f"Q8 | {model_name} | edge noise")
        cls = MODEL_REGISTRY[model_name]
        projected_cls = make_projected_cls(cls)
        model_params = {"in_dim": in_dim, "hidden_dim": config["training"]["hidden_dim"],
                        "out_dim": num_classes, "num_layers": config["training"]["num_layers"],
                        "dropout": config["training"]["dropout"]}
        result = run_noise_experiment(
            train_and_eval, dataset.data, projected_cls, model_params,
            noise_type="edge",
            noise_ratios=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            num_runs=num_runs, device=device,
            epochs=config["training"]["num_epochs"], lr=config["training"]["lr"],
            patience=config["training"]["patience"])
        all_results[model_name] = result

    save_result("q8_edge_noise", all_results)
    return all_results


# ==================== Q9-Q10: 效率分析 ====================

def run_q9_q10_efficiency(config, gpu=0):
    from benchmark.tasks.efficiency_profiling import run_efficiency_analysis, generate_complexity_table
    from benchmark.models.gnn import MODEL_REGISTRY

    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    dataset = get_dataset("text+image+formula")
    in_dim = dataset.data.x.size(1)
    num_classes = dataset.data.y.max().item() + 1
    dataset.data.train_mask, dataset.data.val_mask, dataset.data.test_mask = dataset.get_split_masks()

    hd = config["training"]["hidden_dim"]
    dp = config["training"]["dropout"]
    models_dict = {}
    for name in EFFECTIVENESS_MODELS:
        cls = MODEL_REGISTRY[name]
        projected_cls = make_projected_cls(cls)
        models_dict[name] = (projected_cls, {"in_dim": in_dim, "hidden_dim": hd, "out_dim": num_classes,
                                              "num_layers": 2, "dropout": dp})

    empirical = run_efficiency_analysis(models_dict, dataset.data, device=device)
    theoretical = generate_complexity_table()
    theoretical += [
        {"model": "MLP", "training": "O(L|V|d²)", "inference": "O(L|V|d²)", "memory": "O(L|V|d)"},
        {"model": "GCN", "training": "O(L|E|d + L|V|d²)", "inference": "O(L|E|d + L|V|d²)", "memory": "O(L|V|d + |E|)"},
        {"model": "GAT", "training": "O(L|E|dH + L|V|d²H)", "inference": "O(L|E|dH + L|V|d²H)", "memory": "O(L|V|dH + |E|)"},
        {"model": "DGF", "training": "O(K|E|d + |V|d²)", "inference": "O(K|E|d + |V|d²)", "memory": "O(|V|d + |E|)"},
        {"model": "MMGCN", "training": "O(ML|E|d)", "inference": "O(ML|E|d)", "memory": "O(ML|V|d)"},
        {"model": "MGAT", "training": "O(L|E|dH + M|V|d²)", "inference": "O(L|E|dH)", "memory": "O(M|V|d + |E|)"},
        {"model": "LGMRec", "training": "O(L|E|d + K|V|d)", "inference": "O(L|E|d + K|V|d)", "memory": "O((K+L)|V|d + |E|)"},
        {"model": "DMGC", "training": "O(|E|d + |V|d²)", "inference": "O(|E|d + |V|d²)", "memory": "O(|V|d + |E|)"},
        {"model": "MHGAT", "training": "O(L|E|dH M²)", "inference": "O(L|E|dH M²)", "memory": "O(M|V|d + |E|)"},
    ]
    results = {"theoretical_complexity": theoretical, "empirical_efficiency": empirical}
    save_result("q9_q10_efficiency", results)
    return results


# ==================== Q1b: Per-Domain Modality Preference ====================

def run_q1b_domain_preference(config, gpu=0):
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    num_runs = config.get("openmag", {}).get("num_runs", 3)
    models = ["mlp", "gcn", "gat", "graphsage"]
    modalities = ["text", "image", "text+image+formula"]
    all_results = []

    from benchmark.data.dataset import AMMAGDataset

    for model_name in models:
        for modality in modalities:
            dataset = get_dataset(modality)
            in_dim = dataset.data.x.size(1)
            num_classes = dataset.data.y.max().item() + 1
            domain_labels = dataset._domain_labels

            domain_accs = {d: [] for d in AMMAGDataset.DOMAIN_KEYS}
            overall_accs = []
            for run in range(num_runs):
                seed = run * 42 + 1
                torch.manual_seed(seed); np.random.seed(seed)
                dataset.data.train_mask, dataset.data.val_mask, dataset.data.test_mask = dataset.get_split_masks(seed=seed)
                model = get_model(model_name, in_dim, num_classes, config)
                res = train_and_eval(model, dataset.data, device, epochs=config["training"]["num_epochs"],
                                     lr=config["training"]["lr"], patience=config["training"]["patience"])
                overall_accs.append(res["test_acc"])

                # Per-domain accuracy on test set
                model.eval()
                with torch.no_grad():
                    x = dataset.data.x.to(device)
                    edge_index = dataset.data.edge_index.to(device)
                    pred = model(x, edge_index).argmax(dim=1).cpu()
                    y = dataset.data.y
                    test_mask = dataset.data.test_mask
                    for dkey in AMMAGDataset.DOMAIN_KEYS:
                        d_idx = AMMAGDataset.DOMAIN_MAP[dkey]
                        d_mask = (domain_labels[test_mask] == d_idx)
                        if d_mask.sum() > 0:
                            acc = (pred[test_mask][d_mask] == y[test_mask][d_mask]).float().mean().item()
                            domain_accs[dkey].append(acc)

            r = {"model": model_name, "modality": modality,
                 "overall_acc": float(np.mean(overall_accs)),
                 "overall_std": float(np.std(overall_accs))}
            for dkey in AMMAGDataset.DOMAIN_KEYS:
                if domain_accs[dkey]:
                    r[f"{dkey}_acc"] = float(np.mean(domain_accs[dkey]))
                    r[f"{dkey}_std"] = float(np.std(domain_accs[dkey]))
            all_results.append(r)
            logger.info(f"Q1b | {model_name} | {modality}: overall={r['overall_acc']:.4f}")

    save_result("q1b_domain_preference", {"results": all_results})
    return all_results


# ==================== Q5b: Extended Effectiveness (with GIN, GCNII, GATv2) ====================

def run_q5b_extended_effectiveness(config, gpu=0):
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    num_runs = config.get("openmag", {}).get("num_runs", 3)
    modality = "text+image+formula"

    nc_results = []
    dataset = get_dataset(modality)
    in_dim = dataset.data.x.size(1)
    num_classes = dataset.data.y.max().item() + 1

    for model_name in EXTENDED_MODELS:
        logger.info(f"Q5b NC | {model_name}")
        accs, f1s = [], []
        for run in range(num_runs):
            torch.manual_seed(run * 42 + 1); np.random.seed(run * 42 + 1)
            dataset.data.train_mask, dataset.data.val_mask, dataset.data.test_mask = dataset.get_split_masks(seed=run*42+1)
            model = get_model(model_name, in_dim, num_classes, config)
            res = train_and_eval(model, dataset.data, device, epochs=config["training"]["num_epochs"],
                                 lr=config["training"]["lr"], patience=config["training"]["patience"])
            accs.append(res["test_acc"]); f1s.append(res["test_f1"])
        nc_results.append({"model": model_name, "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                           "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s))})
        logger.info(f"  -> acc={np.mean(accs):.4f}±{np.std(accs):.4f}, f1={np.mean(f1s):.4f}±{np.std(f1s):.4f}")

    save_result("q5b_extended_effectiveness", {"node_classification": nc_results})
    return nc_results


# ==================== Q5c: Statistical Significance Tests ====================

def run_q5c_significance(config, gpu=0):
    from scipy import stats
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    num_runs = 10
    modality = "text+image+formula"

    all_run_accs = {}
    dataset = get_dataset(modality)
    in_dim = dataset.data.x.size(1)
    num_classes = dataset.data.y.max().item() + 1

    for model_name in EXTENDED_MODELS:
        logger.info(f"Q5c | {model_name} (10 runs)")
        accs = []
        for run in range(num_runs):
            torch.manual_seed(run * 42 + 1); np.random.seed(run * 42 + 1)
            dataset.data.train_mask, dataset.data.val_mask, dataset.data.test_mask = dataset.get_split_masks(seed=run*42+1)
            model = get_model(model_name, in_dim, num_classes, config)
            res = train_and_eval(model, dataset.data, device, epochs=config["training"]["num_epochs"],
                                 lr=config["training"]["lr"], patience=config["training"]["patience"])
            accs.append(res["test_acc"])
        all_run_accs[model_name] = accs
        logger.info(f"  -> {np.mean(accs):.4f}±{np.std(accs):.4f}")

    # Pairwise t-tests
    model_names = list(all_run_accs.keys())
    significance = {}
    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            if i >= j:
                continue
            t_stat, p_value = stats.ttest_rel(all_run_accs[m1], all_run_accs[m2])
            sig = "ns"
            if p_value < 0.01:
                sig = "**"
            elif p_value < 0.05:
                sig = "*"
            significance[f"{m1}_vs_{m2}"] = {
                "t_stat": float(t_stat), "p_value": float(p_value), "significance": sig,
                "m1_mean": float(np.mean(all_run_accs[m1])), "m2_mean": float(np.mean(all_run_accs[m2])),
            }

    # Summary with per-run data
    summary = []
    for mn in model_names:
        summary.append({"model": mn, "acc_mean": float(np.mean(all_run_accs[mn])),
                        "acc_std": float(np.std(all_run_accs[mn])), "accs": all_run_accs[mn]})

    save_result("q5c_significance", {"per_model": summary, "pairwise": significance})
    return significance


# ==================== Q5d: Modality Retrieval as Standalone Task ====================

def run_q5d_retrieval_task(config, gpu=0):
    from benchmark.tasks.encoding_comparison import compute_retrieval_metrics
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    num_runs = config.get("openmag", {}).get("num_runs", 3)
    modality = "text+image+formula"

    # Load raw embeddings for retrieval
    text_embs = torch.load(os.path.join(PROCESSED_DIR, "text_embeddings.pt"), map_location="cpu")["embeddings"]
    image_embs = torch.load(os.path.join(PROCESSED_DIR, "image_embeddings.pt"), map_location="cpu")["embeddings"]
    n_common = min(text_embs.size(0), image_embs.size(0))
    text_embs = text_embs[:n_common]
    image_embs = image_embs[:n_common]

    dataset = get_dataset(modality)
    in_dim = dataset.data.x.size(1)
    num_classes = dataset.data.y.max().item() + 1

    retrieval_results = []
    for model_name in EFFECTIVENESS_MODELS:
        logger.info(f"Q5d Retrieval | {model_name}")
        t2i_mrrs, i2t_mrrs, t2i_h10s, i2t_h10s = [], [], [], []
        for run in range(num_runs):
            torch.manual_seed(run * 42 + 1); np.random.seed(run * 42 + 1)
            dataset.data.train_mask, dataset.data.val_mask, dataset.data.test_mask = dataset.get_split_masks(seed=run*42+1)
            model = get_model(model_name, in_dim, num_classes, config)
            res = train_and_eval(model, dataset.data, device, epochs=config["training"]["num_epochs"],
                                 lr=config["training"]["lr"], patience=config["training"]["patience"])

            # Use projected 768d features (after projection layer) as node embeddings
            model.eval()
            with torch.no_grad():
                x = dataset.data.x.to(device)
                edge_index = dataset.data.edge_index.to(device)
                # Get the projected 768d representation
                if hasattr(model, 'projector'):
                    node_embs = model.projector(x).cpu()
                elif hasattr(model, 'model') and hasattr(model.model, 'proj'):
                    node_embs = model.model.proj(x).cpu()
                else:
                    node_embs = x[:, :768].cpu()

            # Compute retrieval on a sample to avoid OOM
            node_embs_aligned = node_embs[:n_common]
            sample_size = 5000
            perm = torch.randperm(n_common)[:sample_size]
            t2i_metrics = compute_retrieval_metrics(
                F.normalize(node_embs_aligned[perm].float(), dim=1),
                F.normalize(image_embs[perm].float(), dim=1),
                ks=(1, 3, 10, 20)
            )
            i2t_metrics = compute_retrieval_metrics(
                F.normalize(node_embs_aligned[perm].float(), dim=1),
                F.normalize(text_embs[perm].float(), dim=1),
                ks=(1, 3, 10, 20)
            )
            t2i_mrrs.append(t2i_metrics["t2i_mrr"])
            i2t_mrrs.append(i2t_metrics["t2i_mrr"])
            t2i_h10s.append(t2i_metrics.get("t2i_hits@10", 0))
            i2t_h10s.append(i2t_metrics.get("t2i_hits@10", 0))

        retrieval_results.append({
            "model": model_name,
            "node2img_mrr_mean": float(np.mean(t2i_mrrs)), "node2img_mrr_std": float(np.std(t2i_mrrs)),
            "node2txt_mrr_mean": float(np.mean(i2t_mrrs)), "node2txt_mrr_std": float(np.std(i2t_mrrs)),
            "node2img_hits10_mean": float(np.mean(t2i_h10s)), "node2txt_hits10_mean": float(np.mean(i2t_h10s)),
        })
        logger.info(f"  -> N2I MRR={np.mean(t2i_mrrs):.4f}, N2T MRR={np.mean(i2t_mrrs):.4f}")

    save_result("q5d_retrieval_task", {"retrieval_results": retrieval_results})
    return retrieval_results


# ==================== Q5e: Self-Supervised Baseline ====================

def run_q5e_selfsupervised(config, gpu=0):
    from benchmark.models.advanced_models import SelfSupervisedGNN
    from benchmark.models.gnn import ProjectedModel
    from benchmark.tasks.node_clustering import run_clustering_experiment
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    num_runs = config.get("openmag", {}).get("num_runs", 3)
    modality = "text+image+formula"
    dataset = get_dataset(modality)
    in_dim = dataset.data.x.size(1)
    num_classes = dataset.data.y.max().item() + 1

    results = {}
    for mode in ["selfsupervised", "supervised"]:
        logger.info(f"Q5e | mode={mode}")
        accs, f1s = [], []
        for run in range(num_runs):
            seed = run * 42 + 1
            torch.manual_seed(seed); np.random.seed(seed)
            dataset.data.train_mask, dataset.data.val_mask, dataset.data.test_mask = dataset.get_split_masks(seed=seed)

            model = SelfSupervisedGNN(PROJECT_DIM, config["training"]["hidden_dim"], num_classes,
                                       num_layers=config["training"]["num_layers"], dropout=config["training"]["dropout"])
            if in_dim != PROJECT_DIM:
                model = ProjectedModel(model, in_dim, PROJECT_DIM)
            model = model.to(device)

            if mode == "selfsupervised":
                # Phase 1: Self-supervised pretraining
                optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
                x, edge_index = dataset.data.x.to(device), dataset.data.edge_index.to(device)
                for epoch in range(200):
                    model.train()
                    optimizer.zero_grad()
                    inner = model.model if isinstance(model, ProjectedModel) else model
                    loss = inner.contrastive_loss(x if not isinstance(model, ProjectedModel) else model.projector(x), edge_index)
                    loss.backward()
                    optimizer.step()

                # Phase 2: Freeze encoder, train linear classifier
                if isinstance(model, ProjectedModel):
                    for p in model.model.encoder_convs.parameters():
                        p.requires_grad = False
                    for p in model.model.encoder_bn.parameters():
                        p.requires_grad = False
                else:
                    for p in model.encoder_convs.parameters():
                        p.requires_grad = False
                    for p in model.encoder_bn.parameters():
                        p.requires_grad = False

                res = train_and_eval(model, dataset.data, device, epochs=config["training"]["num_epochs"],
                                     lr=config["training"]["lr"] * 10, patience=config["training"]["patience"])
            else:
                res = train_and_eval(model, dataset.data, device, epochs=config["training"]["num_epochs"],
                                     lr=config["training"]["lr"], patience=config["training"]["patience"])
            accs.append(res["test_acc"]); f1s.append(res["test_f1"])

        results[mode] = {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                         "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s))}
        logger.info(f"  {mode}: acc={np.mean(accs):.4f}±{np.std(accs):.4f}")

    save_result("q5e_selfsupervised", results)
    return results


# ==================== Q7b: Modality Missing Robustness ====================

def run_q7b_modality_missing(config, gpu=0):
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    num_runs = config.get("openmag", {}).get("num_runs", 3)
    modality = "text+image+formula"
    dataset = get_dataset(modality)
    in_dim = dataset.data.x.size(1)
    num_classes = dataset.data.y.max().item() + 1
    dataset.data.train_mask, dataset.data.val_mask, dataset.data.test_mask = dataset.get_split_masks()

    # Missing scenarios: which dimension ranges to zero out
    # text: [0, 768), image: [768, 1536), formula: [1536, 2304)
    scenarios = {
        "text_only": {"keep_ranges": [(0, 768)]},
        "image_only": {"keep_ranges": [(768, 1536)]},
        "formula_only": {"keep_ranges": [(1536, 2304)]},
        "text_image": {"keep_ranges": [(0, 768), (768, 1536)]},
        "random_50pct": {"random_mask_ratio": 0.5},
    }

    from benchmark.models.gnn import MODEL_REGISTRY
    all_results = {}
    for model_name in ROBUSTNESS_MODELS:
        all_results[model_name] = {}
        cls = MODEL_REGISTRY[model_name]
        projected_cls = make_projected_cls(cls)
        model_params = {"in_dim": in_dim, "hidden_dim": config["training"]["hidden_dim"],
                        "out_dim": num_classes, "num_layers": config["training"]["num_layers"],
                        "dropout": config["training"]["dropout"]}

        for scenario_name, scenario_cfg in scenarios.items():
            logger.info(f"Q7b | {model_name} | {scenario_name}")
            accs = []
            for run in range(num_runs):
                seed = run * 42 + 1
                torch.manual_seed(seed); np.random.seed(seed)
                noisy_data = dataset.data.clone()

                if "keep_ranges" in scenario_cfg:
                    mask = torch.zeros(in_dim, dtype=torch.bool)
                    for start, end in scenario_cfg["keep_ranges"]:
                        mask[start:end] = True
                    noisy_data.x = noisy_data.x.clone()
                    noisy_data.x[:, ~mask] = 0.0
                elif "random_mask_ratio" in scenario_cfg:
                    ratio = scenario_cfg["random_mask_ratio"]
                    feat_mask = torch.bernoulli(torch.ones(in_dim) * (1 - ratio)).bool()
                    noisy_data.x = noisy_data.x.clone()
                    noisy_data.x[:, ~feat_mask] = 0.0

                model = projected_cls(**model_params)
                res = train_and_eval(model, noisy_data, device, epochs=config["training"]["num_epochs"],
                                     lr=config["training"]["lr"], patience=config["training"]["patience"])
                accs.append(res["test_acc"])

            all_results[model_name][scenario_name] = {
                "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            }
            logger.info(f"  -> acc={np.mean(accs):.4f}±{np.std(accs):.4f}")

    save_result("q7b_modality_missing", all_results)
    return all_results

PHASES = {
    "q1_necessity": run_q1_necessity,
    "q1b_domain_preference": run_q1b_domain_preference,
    "q2_neighbors": run_q2_neighbors,
    "q3_finetune": run_q3_finetune,
    "q4_encoding": run_q4_encoding,
    "q5_effectiveness": run_q5_effectiveness,
    "q5b_extended_effectiveness": run_q5b_extended_effectiveness,
    "q5c_significance": run_q5c_significance,
    "q5d_retrieval_task": run_q5d_retrieval_task,
    "q5e_selfsupervised": run_q5e_selfsupervised,
    "q6_peft": run_q6_peft,
    "q7_robustness": run_q7_robustness,
    "q7b_modality_missing": run_q7b_modality_missing,
    "q8_edge_noise": run_q8_edge_noise,
    "q9_q10_efficiency": run_q9_q10_efficiency,
}

def main():
    parser = argparse.ArgumentParser(description="OpenMAG Experiment Runner")
    parser.add_argument("--phase", type=str, required=True,
                        choices=list(PHASES.keys()) + ["all"], help="Experiment phase")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    args = parser.parse_args()
    config = load_config()
    logger.info(f"OpenMAG Experiment: phase={args.phase}, gpu={args.gpu}, proj_dim={PROJECT_DIM}")

    if args.phase == "all":
        for name, fn in PHASES.items():
            logger.info(f"\n{'='*60}\nRunning: {name}\n{'='*60}")
            try:
                fn(config, args.gpu)
            except Exception as e:
                logger.error(f"Phase {name} failed: {e}")
                import traceback; traceback.print_exc()
    else:
        PHASES[args.phase](config, args.gpu)
    logger.info("Done!")

if __name__ == "__main__":
    main()
