"""
补充实验脚本 - 补齐 Benchmark 报告的缺失实验
所有编码统一为 CLIP-ViT-Large 768d，多模态输入通过 ProjectedModel 投影到 768d

补充内容:
  Phase A: 7 种模态组合完整消融 (Q1 扩展)
  Phase B: Few-shot 学习曲线 (7 模态 × 5 shot × 5 模型)
  Phase C: 跨域迁移学习 (12 对域间迁移)
  Phase D: 跨域链接预测 (解决 LP 饱和问题)
  Phase E: 可视化 (t-SNE, 柱状图, 折线图)
"""
import os
import sys
import json
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.metrics import f1_score, accuracy_score
from sklearn.manifold import TSNE

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "benchmark", "results", "openmag")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "benchmark", "processed")
PROJ_DIM = 768

ALL_7_MODALITIES = ["text", "image", "formula", "text+image", "text+formula",
                    "image+formula", "text+image+formula"]
MODALITY_LABELS = {
    "text": "Text", "image": "Image", "formula": "Formula",
    "text+image": "T+I", "text+formula": "T+F",
    "image+formula": "I+F", "text+image+formula": "T+I+F",
}

os.makedirs(RESULTS_DIR, exist_ok=True)

# Matplotlib 中文字体
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def save_result(name, data):
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved: {path}")


def get_dataset(modality="text", label_type="domain"):
    from benchmark.data.dataset import AMMAGDataset
    return AMMAGDataset(root=PROJECT_ROOT, modality=modality, label_type=label_type)


def get_model(model_name, in_dim, num_classes, hidden_dim=256, num_layers=2,
              dropout=0.5, proj_dim=PROJ_DIM):
    from benchmark.models.gnn import get_model as _get_model
    return _get_model(model_name, in_dim, hidden_dim, num_classes,
                      num_layers=num_layers, dropout=dropout, proj_dim=proj_dim)


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


# ==================== Phase A: 7 种模态组合完整消融 ====================

def run_phase_a_full_modality_ablation(gpu=0):
    """7 种模态组合 × 4 模型 × NC + LP"""
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    num_runs = 3
    nc_models = ["mlp", "gcn", "gat", "graphsage"]
    lp_models = ["mlp", "gcn", "gat"]
    all_results = []

    # NC: 全部 7 种模态
    for model_name in nc_models:
        for modality in ALL_7_MODALITIES:
            logger.info(f"PhaseA NC | {model_name} | {modality}")
            dataset = get_dataset(modality)
            in_dim = dataset.data.x.size(1)
            num_classes = dataset.data.y.max().item() + 1
            accs, f1s = [], []
            for run in range(num_runs):
                torch.manual_seed(run * 42 + 1)
                np.random.seed(run * 42 + 1)
                dataset.data.train_mask, dataset.data.val_mask, dataset.data.test_mask = \
                    dataset.get_split_masks(seed=run * 42 + 1)
                model = get_model(model_name, in_dim, num_classes)
                res = train_and_eval(model, dataset.data, device)
                accs.append(res["test_acc"])
                f1s.append(res["test_f1"])
            r = {"model": model_name, "modality": modality, "task": "NC",
                 "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                 "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s))}
            all_results.append(r)
            logger.info(f"  -> acc={r['acc_mean']:.4f}±{r['acc_std']:.4f}, "
                        f"f1={r['f1_mean']:.4f}±{r['f1_std']:.4f}")

    # LP: 全部 7 种模态
    from benchmark.tasks.link_prediction import run_link_prediction
    for modality in ALL_7_MODALITIES:
        for model_name in lp_models:
            logger.info(f"PhaseA LP | {model_name} | {modality}")
            dataset = get_dataset(modality)
            lp_res = run_link_prediction(
                dataset=dataset, model_name=model_name, modality=modality,
                num_runs=num_runs, hidden_dim=256, num_layers=2, dropout=0.5,
                lr=0.001, device=device, output_dir=RESULTS_DIR, proj_dim=PROJ_DIM)
            r = {"model": model_name, "modality": modality, "task": "LP",
                 "mrr_mean": lp_res.get("mrr_mean", 0), "mrr_std": lp_res.get("mrr_std", 0),
                 "hits3_mean": lp_res.get("hits@3_mean", 0), "hits3_std": lp_res.get("hits@3_std", 0)}
            all_results.append(r)
            logger.info(f"  -> MRR={r['mrr_mean']:.4f}±{r['mrr_std']:.4f}")

    save_result("q1_full_modality_ablation", {"results": all_results})
    return all_results


# ==================== Phase B: Few-shot 学习曲线 ====================

def run_phase_b_few_shot(gpu=0):
    """7 模态 × 5 shot 值 × 5 模型"""
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    num_runs = 10
    models = ["gcn", "gat", "graphsage", "gin", "gt"]
    shot_list = [1, 3, 5, 10, 20]
    all_results = {}

    for model_name in models:
        for modality in ALL_7_MODALITIES:
            dataset = get_dataset(modality)
            in_dim = dataset.data.x.size(1)
            out_dim = int(dataset.data.y.max().item()) + 1
            data = dataset.data.to(device)

            for k in shot_list:
                key = f"{model_name}_{modality}_k{k}"
                run_accs, run_f1s = [], []
                for run in range(num_runs):
                    seed = run * 42 + 1
                    train_mask, val_mask, test_mask = dataset.get_split_masks(
                        split="few_shot", seed=seed, few_shot_k=k)
                    model = get_model(model_name, in_dim, out_dim)
                    res = _train_and_eval_simple(
                        model, data, train_mask, val_mask, test_mask, device)
                    run_accs.append(res["test_acc"])
                    run_f1s.append(res["test_f1"])

                summary = {
                    "model": model_name, "modality": modality, "k": k,
                    "acc_mean": float(np.mean(run_accs)),
                    "acc_std": float(np.std(run_accs)),
                    "f1_mean": float(np.mean(run_f1s)),
                    "f1_std": float(np.std(run_f1s)),
                    "num_runs": num_runs,
                }
                all_results[key] = summary
                logger.info(f"{key}: acc={summary['acc_mean']*100:.2f}±"
                            f"{summary['acc_std']*100:.2f}%")

    save_result("few_shot_full", all_results)
    return all_results


def _train_and_eval_simple(model, data, train_mask, val_mask, test_mask, device,
                           num_epochs=500, lr=0.001, patience=50):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    x, edge_index, y = data.x.to(device), data.edge_index.to(device), data.y.to(device)
    train_mask, val_mask, test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)

    best_val_acc, best_state, p_counter = 0, None, 0
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(x, edge_index).argmax(dim=1)
                val_acc = (pred[val_mask] == y[val_mask]).float().mean().item()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                p_counter = 0
            else:
                p_counter += 10
            if p_counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(x, edge_index).argmax(dim=1)
        test_acc = (pred[test_mask] == y[test_mask]).float().mean().item()
        test_f1 = f1_score(y[test_mask].cpu().numpy(), pred[test_mask].cpu().numpy(), average="macro")
    return {"test_acc": float(test_acc), "test_f1": float(test_f1)}


# ==================== Phase C: 跨域迁移学习 ====================

def run_phase_c_cross_domain(gpu=0):
    """4 域 × 3 目标域 = 12 对迁移"""
    from benchmark.tasks.cross_domain import run_cross_domain_transfer
    from benchmark.data.dataset import AMMAGDataset

    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    domains = AMMAGDataset.DOMAIN_KEYS
    models = ["gcn", "gat", "graphsage"]
    modality = "text+image+formula"

    all_results = {}
    for src in domains:
        for tgt in domains:
            if src == tgt:
                continue
            for model_name in models:
                key = f"{src}_to_{tgt}_{model_name}"
                logger.info(f"PhaseC | {src} -> {tgt} | {model_name}")
                dataset = get_dataset(modality, label_type="subcat")
                result = run_cross_domain_transfer(
                    dataset, model_name, modality, src, tgt,
                    num_runs=3, num_epochs=500, hidden_dim=256,
                    num_layers=2, dropout=0.5, lr=0.01,
                    device=device, output_dir=RESULTS_DIR)
                all_results[key] = result
                logger.info(f"  gain={result['transfer_gain_mean']:+.4f}")

    save_result("cross_domain_transfer", all_results)
    return all_results


# ==================== Phase D: 跨域链接预测 ====================

def run_phase_d_cross_domain_lp(gpu=0):
    """跨域链接预测 - 只预测跨域边，解决 LP 饱和问题"""
    from benchmark.tasks.cross_domain_link_prediction import run_cross_domain_link_prediction

    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    models = ["gcn", "gat", "graphsage", "mlp"]
    modalities = ["text", "image", "text+image+formula"]

    all_results = {}
    for modality in modalities:
        logger.info(f"PhaseD | Cross-domain LP | {modality}")
        dataset = get_dataset(modality)
        result = run_cross_domain_link_prediction(
            dataset, model_names=models, modalities=[modality],
            num_runs=3, num_epochs=500, hidden_dim=256,
            num_layers=2, dropout=0.5, lr=0.01,
            device=device, output_dir=RESULTS_DIR)
        all_results.update(result)

    save_result("cross_domain_lp", all_results)
    return all_results


# ==================== Phase E: 可视化 ====================

def run_phase_e_visualizations(gpu=0):
    """生成所有可视化图表"""
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    fig_dir = os.path.join(PROJECT_ROOT, "benchmark", "results", "openmag", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # --- 1. t-SNE 嵌入可视化 ---
    _plot_tsne(device, fig_dir)

    # --- 2. 模态消融柱状图 ---
    _plot_modality_ablation_bar(fig_dir)

    # --- 3. 噪声鲁棒性折线图 ---
    _plot_robustness_curves(fig_dir)

    # --- 4. Few-shot 学习曲线 ---
    _plot_few_shot_curves(fig_dir)

    # --- 5. 跨域迁移热力图 ---
    _plot_transfer_heatmap(fig_dir)

    logger.info(f"All figures saved to {fig_dir}")


def _plot_tsne(device, fig_dir):
    """t-SNE 嵌入可视化 - 展示 4 个域的节点聚类"""
    logger.info("Generating t-SNE visualization...")
    dataset = get_dataset("text+image+formula")
    data = dataset.data
    x = data.x.numpy()
    labels = dataset._domain_labels.numpy()
    domain_names = ["cs.SI", "cs.LO", "math.LO", "physics.soc-ph"]

    # Subsample for speed (max 10000 nodes)
    n = min(len(x), 10000)
    idx = np.random.RandomState(42).choice(len(x), n, replace=False)
    x_sub = x[idx]
    labels_sub = labels[idx]

    # Project to 768d first if needed (use text slice)
    if x_sub.shape[1] > 768:
        # Use learned projection: just take first 768 dims (text) for visualization
        # Better: train a simple model and extract embeddings
        # For now, use the full concatenated features
        pass

    logger.info(f"  Running t-SNE on {n} samples, dim={x_sub.shape[1]}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    x_2d = tsne.fit_transform(x_sub)

    # Train a model and extract GNN embeddings for t-SNE
    in_dim = data.x.size(1)
    num_classes = int(data.y.max().item()) + 1
    model = get_model("gcn", in_dim, num_classes)
    data_copy = data.clone()
    data_copy.train_mask, data_copy.val_mask, data_copy.test_mask = dataset.get_split_masks(seed=42)

    # Train briefly
    model = model.to(device)
    model = _quick_train(model, data_copy.to(device), epochs=200)

    # Extract GNN embeddings
    model.eval()
    with torch.no_grad():
        x_dev = data.x.to(device)
        ei_dev = data.edge_index.to(device)
        if hasattr(model, 'projector'):
            x_proj = model.projector(x_dev)
            h = model.model.convs[0](x_proj, ei_dev)
        elif hasattr(model, 'model'):
            h = model.model(x_dev, ei_dev)
        else:
            h = model.convs[0](x_dev, ei_dev)
        h = F.relu(h)
        gnn_embs = h.cpu().numpy()

    # t-SNE on GNN embeddings (768d -> 2d)
    gnn_sub = gnn_embs[idx]
    logger.info(f"  Running t-SNE on GNN embeddings ({gnn_sub.shape[1]}d)...")
    gnn_2d = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000).fit_transform(gnn_sub)

    # Plot: raw features vs GNN embeddings
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

    for ax, embs_2d, title in [(axes[0], x_2d, "Raw Features (CLIP 768d×3)"),
                                 (axes[1], gnn_2d, "GCN Embeddings (256d)")]:
        for d in range(4):
            mask = labels_sub == d
            ax.scatter(embs_2d[mask, 0], embs_2d[mask, 1], c=colors[d],
                       label=domain_names[d], alpha=0.5, s=8)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "tsne_embeddings.pdf"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(fig_dir, "tsne_embeddings.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  t-SNE figure saved.")


def _quick_train(model, data, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    return model


def _plot_modality_ablation_bar(fig_dir):
    """模态消融柱状图"""
    logger.info("Generating modality ablation bar chart...")
    result_path = os.path.join(RESULTS_DIR, "q1_full_modality_ablation.json")
    if not os.path.exists(result_path):
        logger.warning("q1_full_modality_ablation.json not found, skipping bar chart")
        return

    with open(result_path) as f:
        data = json.load(f)

    # Extract NC results grouped by model
    nc_results = [r for r in data["results"] if r["task"] == "NC"]
    models = sorted(set(r["model"] for r in nc_results))

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(ALL_7_MODALITIES))
    width = 0.18
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

    for i, model_name in enumerate(models):
        model_data = [r for r in nc_results if r["model"] == model_name]
        accs = []
        stds = []
        for mod in ALL_7_MODALITIES:
            match = [r for r in model_data if r["modality"] == mod]
            if match:
                accs.append(match[0]["acc_mean"] * 100)
                stds.append(match[0]["acc_std"] * 100)
            else:
                accs.append(0)
                stds.append(0)
        ax.bar(x + i * width, accs, width, yerr=stds, label=model_name.upper(),
               color=colors[i % len(colors)], alpha=0.85, capsize=3)

    ax.set_xlabel("Modality Combination", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Node Classification: Modality Ablation Study", fontsize=14)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([MODALITY_LABELS[m] for m in ALL_7_MODALITIES], fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(bottom=max(0, ax.get_ylim()[0]))
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "modality_ablation_bar.pdf"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(fig_dir, "modality_ablation_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Modality ablation bar chart saved.")


def _plot_robustness_curves(fig_dir):
    """噪声鲁棒性折线图"""
    logger.info("Generating robustness curves...")
    noise_path = os.path.join(RESULTS_DIR, "q7_robustness.json")
    if not os.path.exists(noise_path):
        logger.warning("q7_robustness.json not found, skipping robustness curves")
        return

    with open(noise_path) as f:
        data = json.load(f)

    noise_types = ["text", "image", "label"]
    noise_labels = ["Feature Noise (Text)", "Feature Noise (Image)", "Label Noise"]
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, noise_type, title in zip(axes, noise_types, noise_labels):
        for i, (model_name, model_data) in enumerate(data.items()):
            if noise_type not in model_data:
                continue
            noise_data = model_data[noise_type]
            results = noise_data.get("results", noise_data)
            ratios = sorted([float(k.replace("ratio_", "")) for k in results.keys()])
            accs = [results[f"ratio_{r}"]["mean"] * 100 for r in ratios]
            ax.plot(ratios, accs, marker='o', markersize=4, label=model_name.upper(),
                    color=colors[i % len(colors)], linewidth=2)
        ax.set_xlabel("Noise Ratio", fontsize=11)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 0.85)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "robustness_curves.pdf"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(fig_dir, "robustness_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Robustness curves saved.")


def _plot_few_shot_curves(fig_dir):
    """Few-shot 学习曲线"""
    logger.info("Generating few-shot curves...")
    fs_path = os.path.join(RESULTS_DIR, "few_shot_full.json")
    if not os.path.exists(fs_path):
        logger.warning("few_shot_full.json not found, skipping few-shot curves")
        return

    with open(fs_path) as f:
        data = json.load(f)

    shot_list = [1, 3, 5, 10, 20]
    models = sorted(set(v["model"] for v in data.values()))
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

    # Plot per modality
    key_modalities = ["text", "image", "formula", "text+image+formula"]
    fig, axes = plt.subplots(1, len(key_modalities), figsize=(20, 4.5))

    for ax, modality in zip(axes, key_modalities):
        for i, model_name in enumerate(models):
            accs = []
            for k in shot_list:
                key = f"{model_name}_{modality}_k{k}"
                if key in data:
                    accs.append(data[key]["acc_mean"] * 100)
                else:
                    accs.append(None)
            if any(a is not None for a in accs):
                ax.plot(shot_list, accs, marker='o', markersize=5, label=model_name.upper(),
                        color=colors[i % len(colors)], linewidth=2)
        ax.set_xlabel("Number of Shots", fontsize=11)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_title(MODALITY_LABELS.get(modality, modality), fontsize=13)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(shot_list)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "few_shot_curves.pdf"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(fig_dir, "few_shot_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Few-shot curves saved.")


def _plot_transfer_heatmap(fig_dir):
    """跨域迁移热力图"""
    logger.info("Generating transfer heatmap...")
    cd_path = os.path.join(RESULTS_DIR, "cross_domain_transfer.json")
    if not os.path.exists(cd_path):
        logger.warning("cross_domain_transfer.json not found, skipping heatmap")
        return

    with open(cd_path) as f:
        data = json.load(f)

    from benchmark.data.dataset import AMMAGDataset
    domains = AMMAGDataset.DOMAIN_KEYS
    domain_short = ["cs.SI", "cs.LO", "math.LO", "physics.soc-ph"]

    # Average transfer gain across models
    matrix = np.zeros((4, 4))
    count = np.zeros((4, 4))
    for key, val in data.items():
        parts = key.split("_to_")
        if len(parts) != 2:
            continue
        src = parts[0]
        tgt_model = parts[1].rsplit("_", 1)
        tgt = tgt_model[0]
        if src in domains and tgt in domains:
            si = domains.index(src)
            ti = domains.index(tgt)
            gain = val.get("transfer_gain_mean", 0)
            matrix[si][ti] += gain
            count[si][ti] += 1

    count[count == 0] = 1
    matrix = matrix / count

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(domain_short, fontsize=11)
    ax.set_yticklabels(domain_short, fontsize=11)
    ax.set_xlabel("Target Domain", fontsize=12)
    ax.set_ylabel("Source Domain", fontsize=12)
    ax.set_title("Cross-Domain Transfer Gain (Accuracy)", fontsize=13)

    for i in range(4):
        for j in range(4):
            if i != j:
                ax.text(j, i, f"{matrix[i][j]:+.3f}", ha="center", va="center",
                        fontsize=10, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Transfer Gain")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "transfer_heatmap.pdf"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(fig_dir, "transfer_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Transfer heatmap saved.")


# ==================== Main ====================

PHASES = {
    "a_modality_ablation": run_phase_a_full_modality_ablation,
    "b_few_shot": run_phase_b_few_shot,
    "c_cross_domain": run_phase_c_cross_domain,
    "d_cross_domain_lp": run_phase_d_cross_domain_lp,
    "e_visualizations": run_phase_e_visualizations,
}


def main():
    parser = argparse.ArgumentParser(description="Supplementary Experiments")
    parser.add_argument("--phase", type=str, required=True,
                        choices=list(PHASES.keys()) + ["all", "abcd"],
                        help="Experiment phase")
    parser.add_argument("--gpu", type=int, default=2, help="GPU ID")
    args = parser.parse_args()

    logger.info(f"Supplementary Experiments: phase={args.phase}, gpu={args.gpu}")

    if args.phase == "all":
        for name, fn in PHASES.items():
            logger.info(f"\n{'='*60}\nRunning: {name}\n{'='*60}")
            try:
                fn(args.gpu)
            except Exception as e:
                logger.error(f"Phase {name} failed: {e}")
                import traceback
                traceback.print_exc()
    elif args.phase == "abcd":
        for name in ["a_modality_ablation", "b_few_shot", "c_cross_domain", "d_cross_domain_lp"]:
            logger.info(f"\n{'='*60}\nRunning: {name}\n{'='*60}")
            try:
                PHASES[name](args.gpu)
            except Exception as e:
                logger.error(f"Phase {name} failed: {e}")
                import traceback
                traceback.print_exc()
    else:
        PHASES[args.phase](args.gpu)

    logger.info("Done!")


if __name__ == "__main__":
    main()
