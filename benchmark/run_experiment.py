#!/usr/bin/env python3
"""
AMMAG Benchmark 实验主入口
支持全监督节点分类、链接预测、Few-shot、模态消融、跨域迁移、子分类分类

Usage:
    python run_experiment.py --step preprocess --gpu 0
    python run_experiment.py --step encode --gpu 0
    python run_experiment.py --step node_classification --model gcn --modality text --gpu 0
    python run_experiment.py --step ablation --model gcn --gpu 0
    python run_experiment.py --step few_shot --model gcn --k 3 --gpu 0
    python run_experiment.py --step link_prediction --model gcn --modality text --gpu 0
    python run_experiment.py --step cross_domain --model gcn --gpu 0
    python run_experiment.py --step per_domain_ablation --gpu 0
    python run_experiment.py --step subcat_nc --domain cs_si --model gcn --gpu 0
    python run_experiment.py --step statistics
    python run_experiment.py --step visualize --gpu 0
"""
import os
import sys
import argparse
import yaml
import json
import torch
import numpy as np
from loguru import logger

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force offline mode for HuggingFace models
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _get_dataset(config, args, modality=None, label_type="domain"):
    from benchmark.data.dataset import AMMAGDataset
    domains = args.domains or config["data"]["domains"]
    return AMMAGDataset(
        root=config.get("project_root", _project_root),
        domains=domains,
        modality=modality or args.modality or "text",
        label_type=label_type,
    )


# ---- Step: Preprocess ----

def step_preprocess(args, config):
    from benchmark.data.preprocessor import AMMAGPreprocessor
    domains = args.domains or config["data"]["domains"]
    processor = AMMAGPreprocessor(data_root=config["data"]["root"])
    result = processor.build_graph_data(
        domains=domains,
        output_dir=config["data"]["processed_dir"],
    )
    logger.info(f"预处理完成: {result['statistics']['num_nodes']} nodes, "
                 f"{result['statistics']['num_edges']} edges")


# ---- Step: Encode ----

def step_encode(args, config):
    domains = args.domains or config["data"]["domains"]
    modalities = args.modalities or ["text", "image", "formula"]
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    data_root = config["data"]["root"]
    processed_dir = config["data"]["processed_dir"]

    from benchmark.data.preprocessor import AMMAGPreprocessor
    processor = AMMAGPreprocessor(data_root=data_root)
    paper_id_map = {}

    all_md_paths = []
    all_ids = []
    all_image_dirs = {}

    for domain in domains:
        papers = processor.scan_papers(domain)
        for p in papers:
            if p["arxiv_id"] not in paper_id_map:
                paper_id_map[p["arxiv_id"]] = len(paper_id_map)
                all_md_paths.append(p["md_path"])
                all_ids.append(p["arxiv_id"])
                if p["has_images"]:
                    all_image_dirs[p["arxiv_id"]] = p["images_dir"]

    logger.info(f"编码 {len(all_ids)} 篇论文, modalities={modalities}")
    embeddings_dict = {}

    if "text" in modalities:
        from benchmark.encoders.text_encoder import TextEncoder
        encoder = TextEncoder(model_name=config["modalities"]["text"]["encoder"], device=device)
        result = encoder.encode_batch(all_md_paths, processed_dir, all_ids)
        embeddings_dict["text"] = torch.load(result["path"])

    if "image" in modalities:
        from benchmark.encoders.image_encoder import ImageEncoder
        encoder = ImageEncoder(model_name=config["modalities"]["image"]["encoder"], device=device)
        result = encoder.encode_batch(all_image_dirs, processed_dir)
        embeddings_dict["image"] = torch.load(result["path"])

    if "formula" in modalities:
        from benchmark.encoders.formula_encoder import FormulaEncoder
        encoder = FormulaEncoder(model_name=config["modalities"]["formula"]["encoder"], device=device)
        result = encoder.encode_batch(all_md_paths, all_ids, processed_dir)
        embeddings_dict["formula"] = torch.load(result["path"])

    if len(modalities) > 1:
        from benchmark.encoders.multimodal_encoder import MultimodalEncoder
        fusion = MultimodalEncoder(fusion="concat")
        for combo in config.get("modality_combos", []):
            mods = combo.split("+")
            if all(m in embeddings_dict for m in mods):
                fusion.fuse(embeddings_dict, all_ids, processed_dir, mods)

    logger.info("特征编码完成!")


# ---- Step: Node Classification ----

def step_node_classification(args, config):
    from benchmark.tasks.node_classification import run_node_classification
    model_name = args.model or "gcn"
    modality = args.modality or "text"
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    dataset = _get_dataset(config, args, modality=modality)
    domain_labels = dataset._domain_labels

    return run_node_classification(
        dataset=dataset,
        model_name=model_name,
        modality=modality,
        num_runs=config["experiments"]["node_classification"]["num_runs"],
        hidden_dim=config["training"]["hidden_dim"],
        num_layers=config["training"]["num_layers"],
        dropout=config["training"]["dropout"],
        lr=config["training"]["lr"],
        num_epochs=config["training"]["num_epochs"],
        device=device,
        output_dir=config["data"]["results_dir"],
        domain_labels=domain_labels,
    )


# ---- Step: Ablation (all models × all modalities) ----

def step_ablation(args, config):
    from benchmark.tasks.node_classification import run_node_classification
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    models = [args.model] if args.model else config["models"]
    combos = config.get("modality_combos", ["text", "image", "formula",
               "text+image", "text+formula", "image+formula", "text+image+formula"])
    all_results = {}

    for modality in combos:
        try:
            dataset = _get_dataset(config, args, modality=modality)
        except Exception as e:
            logger.error(f"加载modality={modality}失败: {e}")
            continue
        domain_labels = dataset._domain_labels

        for model_name in models:
            result = run_node_classification(
                dataset=dataset,
                model_name=model_name,
                modality=modality,
                num_runs=config["experiments"]["node_classification"]["num_runs"],
                hidden_dim=config["training"]["hidden_dim"],
                num_layers=config["training"]["num_layers"],
                dropout=config["training"]["dropout"],
                lr=config["training"]["lr"],
                num_epochs=config["training"]["num_epochs"],
                device=device,
                output_dir=config["data"]["results_dir"],
                domain_labels=domain_labels,
            )
            all_results[f"{model_name}_{modality}"] = result

    summary_path = os.path.join(config["data"]["results_dir"], "ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    return all_results


# ---- Step: Few-Shot ----

def step_few_shot(args, config):
    from benchmark.tasks.node_classification import run_node_classification
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    model_name = args.model or "gcn"
    ks = [args.k] if args.k else config["experiments"]["few_shot"]["ks"]
    modalities = args.modality or "text+image+formula"
    # Support comma-separated modalities for batch
    mod_list = modalities.split(",") if "," in modalities else [modalities]

    all_results = {}
    for modality in mod_list:
        dataset = _get_dataset(config, args, modality=modality)
        domain_labels = dataset._domain_labels

        for k in ks:
            result = run_node_classification(
                dataset=dataset,
                model_name=model_name,
                modality=modality,
                num_runs=config["experiments"]["few_shot"]["num_runs"],
                hidden_dim=config["training"]["hidden_dim"],
                num_layers=config["training"]["num_layers"],
                dropout=config["training"]["dropout"],
                lr=config["training"]["lr"],
                num_epochs=config["training"]["num_epochs"],
                device=device,
                split="few_shot",
                few_shot_k=k,
                output_dir=config["data"]["results_dir"],
                domain_labels=domain_labels,
            )
            all_results[f"{model_name}_{modality}_k={k}"] = result

    return all_results


# ---- Step: Link Prediction ----

def step_link_prediction(args, config):
    from benchmark.data.dataset import AMMAGDataset
    from benchmark.tasks.link_prediction import run_link_prediction
    model_name = args.model or "gcn"
    modality = args.modality or "text"
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    dataset = _get_dataset(config, args, modality=modality)
    return run_link_prediction(
        dataset=dataset,
        model_name=model_name,
        modality=modality,
        num_runs=config["experiments"]["link_prediction"]["num_runs"],
        hidden_dim=config["training"]["hidden_dim"],
        num_layers=config["training"]["num_layers"],
        dropout=config["training"]["dropout"],
        lr=config["training"]["lr"],
        num_epochs=config["training"]["num_epochs"],
        device=device,
        output_dir=config["data"]["results_dir"],
    )


# ---- Step: Per-Domain Modality Ablation ----

def step_per_domain_ablation(args, config):
    """
    核心创新实验：在完整图上跑多个模态组合，收集per-domain accuracy
    展示公式对math.LO增益最大，图像对physics.soc-ph增益最大
    """
    from benchmark.tasks.node_classification import run_node_classification
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    model_name = args.model or config["experiments"].get("per_domain_ablation", {}).get("model", "gcn")
    combos = config.get("modality_combos", ["text", "image", "formula",
               "text+image", "text+formula", "image+formula", "text+image+formula"])
    num_runs = config["experiments"].get("per_domain_ablation", {}).get("num_runs", 5)

    all_results = {}
    for modality in combos:
        logger.info(f"=== Per-Domain Ablation: {model_name} + {modality} ===")
        try:
            dataset = _get_dataset(config, args, modality=modality)
        except Exception as e:
            logger.error(f"加载modality={modality}失败: {e}")
            continue

        result = run_node_classification(
            dataset=dataset,
            model_name=model_name,
            modality=modality,
            num_runs=num_runs,
            hidden_dim=config["training"]["hidden_dim"],
            num_layers=config["training"]["num_layers"],
            dropout=config["training"]["dropout"],
            lr=config["training"]["lr"],
            num_epochs=config["training"]["num_epochs"],
            device=device,
            output_dir=config["data"]["results_dir"],
            domain_labels=dataset._domain_labels,
            save_prefix="ablation_pd",
        )
        all_results[modality] = result

    # 保存消融汇总
    summary_path = os.path.join(config["data"]["results_dir"], "per_domain_ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Per-domain ablation汇总: {summary_path}")
    return all_results


# ---- Step: Cross-Domain Transfer ----

def step_cross_domain(args, config):
    """跨域迁移：Train on domain A, test on domain B"""
    from benchmark.tasks.cross_domain import run_cross_domain_transfer
    from benchmark.data.dataset import AMMAGDataset
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    cd_config = config["experiments"].get("cross_domain", {})
    models = [args.model] if args.model else cd_config.get("models", ["gcn"])
    num_runs = cd_config.get("num_runs", 3)
    modalities = ["text", "text+image+formula"]
    domains = ["cs_si", "cs_lo", "math_lo", "physics_socph"]

    all_results = {}
    for modality in modalities:
        dataset = _get_dataset(config, args, modality=modality)
        for model_name in models:
            for src in domains:
                for tgt in domains:
                    if src == tgt:
                        continue
                    key = f"{model_name}_{modality}_{src}_to_{tgt}"
                    result = run_cross_domain_transfer(
                        dataset=dataset,
                        model_name=model_name,
                        modality=modality,
                        source_domain=src,
                        target_domain=tgt,
                        num_runs=num_runs,
                        hidden_dim=config["training"]["hidden_dim"],
                        num_layers=config["training"]["num_layers"],
                        dropout=config["training"]["dropout"],
                        lr=config["training"]["lr"],
                        num_epochs=config["training"]["num_epochs"],
                        device=device,
                        output_dir=config["data"]["results_dir"],
                    )
                    all_results[key] = result

    summary_path = os.path.join(config["data"]["results_dir"], "cross_domain_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    return all_results


# ---- Step: Per-Domain Subcategory NC ----

def step_subcat_nc(args, config):
    """每域子图上的子分类节点分类"""
    from benchmark.tasks.node_classification import run_node_classification
    from benchmark.data.dataset import AMMAGDataset
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    sc_config = config["experiments"].get("subcat_nc", {})
    models = [args.model] if args.model else sc_config.get("models", ["gcn", "gat", "graphsage"])
    modalities = [args.modality] if args.modality else sc_config.get("modalities", ["text", "text+image+formula"])
    num_runs = sc_config.get("num_runs", 5)
    target_domain = args.domain  # 单域运行
    domains = [target_domain] if target_domain else ["cs_si", "cs_lo", "math_lo", "physics_socph"]

    all_results = {}
    for domain in domains:
        for modality in modalities:
            # 加载完整数据集
            dataset = _get_dataset(config, args, modality=modality, label_type="subcat")
            # 提取子图
            subgraph = dataset.get_domain_subgraph(domain)

            from benchmark.models.gnn import get_model
            in_dim = subgraph.x.size(1)
            out_dim = int(subgraph.y.max().item()) + 1
            num_nodes = subgraph.num_nodes
            logger.info(f"[Subcat-NC] {domain}+{modality}: {num_nodes} nodes, {out_dim} classes")

            run_results = []
            for run in range(num_runs):
                seed = run * 42 + 1
                rng = np.random.RandomState(seed)
                indices = np.arange(num_nodes)
                rng.shuffle(indices)
                t = int(num_nodes * 0.6)
                v = int(num_nodes * 0.2)

                train_mask = torch.zeros(num_nodes, dtype=torch.bool)
                val_mask = torch.zeros(num_nodes, dtype=torch.bool)
                test_mask = torch.zeros(num_nodes, dtype=torch.bool)
                train_mask[indices[:t]] = True
                val_mask[indices[t:t+v]] = True
                test_mask[indices[t+v:]] = True

                model = get_model("gcn" if len(models) == 1 else models[0],
                                  in_dim, config["training"]["hidden_dim"], out_dim,
                                  num_layers=config["training"]["num_layers"],
                                  dropout=config["training"]["dropout"])

                from benchmark.tasks.node_classification import train_and_eval
                result = train_and_eval(
                    model, subgraph, train_mask, val_mask, test_mask,
                    num_epochs=config["training"]["num_epochs"],
                    lr=config["training"]["lr"],
                    device=device,
                )
                run_results.append(result)

            for model_name in models:
                # 重复上述过程对每个模型
                if model_name == models[0] and len(run_results) > 0:
                    pass  # already ran for first model
                else:
                    run_results_m = []
                    for run in range(num_runs):
                        seed = run * 42 + 1
                        rng = np.random.RandomState(seed)
                        indices = np.arange(num_nodes)
                        rng.shuffle(indices)
                        t = int(num_nodes * 0.6)
                        v = int(num_nodes * 0.2)

                        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
                        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
                        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
                        train_mask[indices[:t]] = True
                        val_mask[indices[t:t+v]] = True
                        test_mask[indices[t+v:]] = True

                        model = get_model(model_name, in_dim,
                                          config["training"]["hidden_dim"], out_dim,
                                          num_layers=config["training"]["num_layers"],
                                          dropout=config["training"]["dropout"])
                        result = train_and_eval(
                            model, subgraph, train_mask, val_mask, test_mask,
                            num_epochs=config["training"]["num_epochs"],
                            lr=config["training"]["lr"],
                            device=device,
                        )
                        run_results_m.append(result)

                    # 汇总
                    from benchmark.tasks.node_classification import run_node_classification
                    # Use simplified summary logic
                    metrics = ["test_acc", "test_f1"]
                    summary = {}
                    for m in metrics:
                        vals = [r[m] for r in run_results_m]
                        summary[f"{m}_mean"] = float(np.mean(vals))
                        summary[f"{m}_std"] = float(np.std(vals))
                    summary["model"] = model_name
                    summary["modality"] = modality
                    summary["domain"] = domain
                    summary["num_runs"] = num_runs
                    key = f"{domain}_{model_name}_{modality}"
                    all_results[key] = summary

                    save_name = f"subcat_nc_{domain}_{model_name}_{modality}_supervised"
                    save_path = os.path.join(config["data"]["results_dir"], f"{save_name}.json")
                    with open(save_path, "w") as f:
                        json.dump(summary, f, indent=2, default=str)

            # Handle first model results
            if len(run_results) > 0 and len(models) > 0:
                metrics = ["test_acc", "test_f1"]
                summary = {}
                for m in metrics:
                    vals = [r[m] for r in run_results]
                    summary[f"{m}_mean"] = float(np.mean(vals))
                    summary[f"{m}_std"] = float(np.std(vals))
                summary["model"] = models[0]
                summary["modality"] = modality
                summary["domain"] = domain
                summary["num_runs"] = num_runs
                key = f"{domain}_{models[0]}_{modality}"
                if key not in all_results:
                    all_results[key] = summary

                    save_name = f"subcat_nc_{domain}_{models[0]}_{modality}_supervised"
                    save_path = os.path.join(config["data"]["results_dir"], f"{save_name}.json")
                    with open(save_path, "w") as f:
                        json.dump(summary, f, indent=2, default=str)

    summary_path = os.path.join(config["data"]["results_dir"], "subcat_nc_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    return all_results


# ---- Step: Statistics ----

def step_statistics(args, config):
    from benchmark.analysis.statistics import (
        generate_report, plot_degree_distribution,
        plot_label_distribution, plot_modality_distribution,
    )
    processed_dir = config["data"]["processed_dir"]
    output_dir = os.path.join(config["data"]["results_dir"], "analysis")
    os.makedirs(output_dir, exist_ok=True)

    stats_path = os.path.join(processed_dir, "statistics.json")
    if not os.path.exists(stats_path):
        logger.error("请先运行 --step preprocess")
        return

    with open(stats_path, "r") as f:
        stats = json.load(f)

    generate_report(stats, output_dir)
    plot_label_distribution(stats, output_dir)
    plot_modality_distribution(stats, output_dir)

    graph_path = os.path.join(processed_dir, "graph_data.pt")
    if os.path.exists(graph_path):
        data = torch.load(graph_path, map_location="cpu")
        if "edge_index" in data:
            plot_degree_distribution(data["edge_index"], output_dir)

    logger.info("统计分析完成!")


# ---- Step: Visualize ----

def step_visualize(args, config):
    from benchmark.analysis.visualization import plot_tsne
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    processed_dir = config["data"]["processed_dir"]
    output_dir = os.path.join(config["data"]["results_dir"], "analysis")
    os.makedirs(output_dir, exist_ok=True)

    graph_path = os.path.join(processed_dir, "graph_data.pt")
    data = torch.load(graph_path, map_location="cpu")
    labels = data["labels"].numpy()
    label_names = ["cs.SI", "cs.LO", "math.LO", "physics.soc-ph"]

    for mod in ["text", "image", "formula"]:
        emb_path = os.path.join(processed_dir, f"{mod}_embeddings.pt")
        if os.path.exists(emb_path):
            emb_data = torch.load(emb_path, map_location="cpu")
            plot_tsne(
                emb_data["embeddings"].numpy(),
                labels[:emb_data["embeddings"].size(0)],
                title=f"t-SNE: {mod.capitalize()} Embeddings",
                output_path=os.path.join(output_dir, f"tsne_{mod}.png"),
                label_names=label_names,
                sample_size=5000,
            )

    # 多模态
    for combo in config.get("modality_combos", []):
        fname = f"multimodal_{combo}_embeddings.pt"
        emb_path = os.path.join(processed_dir, fname)
        if os.path.exists(emb_path):
            emb_data = torch.load(emb_path, map_location="cpu")
            plot_tsne(
                emb_data["embeddings"].numpy(),
                labels[:emb_data["embeddings"].size(0)],
                title=f"t-SNE: {combo} Embeddings",
                output_path=os.path.join(output_dir, f"tsne_{combo.replace('+', '_')}.png"),
                label_names=label_names,
                sample_size=5000,
            )

    logger.info("可视化完成!")


# ---- Step: Hyperparameter Sensitivity ----

def step_hyperparameter_sensitivity(args, config):
    """超参数敏感性分析：learning rate, hidden_dim, dropout, num_layers"""
    from benchmark.tasks.node_classification import run_node_classification
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    model_name = args.model or "graphsage"
    modality = args.modality or "text+image+formula"
    num_runs = 3

    all_results = {}

    # (1) Learning rate sensitivity
    lr_values = [0.001, 0.005, 0.01, 0.02, 0.05]
    logger.info("=== Learning Rate Sensitivity ===")
    for lr in lr_values:
        dataset = _get_dataset(config, args, modality=modality)
        domain_labels = dataset._domain_labels
        result = run_node_classification(
            dataset=dataset, model_name=model_name, modality=modality,
            num_runs=num_runs, hidden_dim=256, num_layers=2, dropout=0.5,
            lr=lr, num_epochs=500, device=device,
            output_dir=os.path.join(config["data"]["results_dir"], "sensitivity"),
            domain_labels=domain_labels,
        )
        all_results[f"lr_{lr}"] = {
            "lr": lr, "acc": result["test_acc_mean"],
            "f1": result["test_f1_mean"], "std": result["test_acc_std"],
        }
        logger.info(f"lr={lr}: acc={result['test_acc_mean']:.4f}±{result['test_acc_std']:.4f}")

    # (2) Hidden dimension sensitivity
    hidden_dims = [64, 128, 256, 512]
    logger.info("=== Hidden Dim Sensitivity ===")
    for hdim in hidden_dims:
        dataset = _get_dataset(config, args, modality=modality)
        domain_labels = dataset._domain_labels
        result = run_node_classification(
            dataset=dataset, model_name=model_name, modality=modality,
            num_runs=num_runs, hidden_dim=hdim, num_layers=2, dropout=0.5,
            lr=0.01, num_epochs=500, device=device,
            output_dir=os.path.join(config["data"]["results_dir"], "sensitivity"),
            domain_labels=domain_labels,
        )
        all_results[f"hdim_{hdim}"] = {
            "hidden_dim": hdim, "acc": result["test_acc_mean"],
            "f1": result["test_f1_mean"], "std": result["test_acc_std"],
        }
        logger.info(f"hidden_dim={hdim}: acc={result['test_acc_mean']:.4f}±{result['test_acc_std']:.4f}")

    # (3) Dropout sensitivity
    dropouts = [0.0, 0.3, 0.5, 0.7]
    logger.info("=== Dropout Sensitivity ===")
    for dp in dropouts:
        dataset = _get_dataset(config, args, modality=modality)
        domain_labels = dataset._domain_labels
        result = run_node_classification(
            dataset=dataset, model_name=model_name, modality=modality,
            num_runs=num_runs, hidden_dim=256, num_layers=2, dropout=dp,
            lr=0.01, num_epochs=500, device=device,
            output_dir=os.path.join(config["data"]["results_dir"], "sensitivity"),
            domain_labels=domain_labels,
        )
        all_results[f"dropout_{dp}"] = {
            "dropout": dp, "acc": result["test_acc_mean"],
            "f1": result["test_f1_mean"], "std": result["test_acc_std"],
        }
        logger.info(f"dropout={dp}: acc={result['test_acc_mean']:.4f}±{result['test_acc_std']:.4f}")

    # (4) Number of GNN layers
    num_layers_list = [1, 2, 3, 4]
    logger.info("=== Num Layers Sensitivity ===")
    for nl in num_layers_list:
        dataset = _get_dataset(config, args, modality=modality)
        domain_labels = dataset._domain_labels
        result = run_node_classification(
            dataset=dataset, model_name=model_name, modality=modality,
            num_runs=num_runs, hidden_dim=256, num_layers=nl, dropout=0.5,
            lr=0.01, num_epochs=500, device=device,
            output_dir=os.path.join(config["data"]["results_dir"], "sensitivity"),
            domain_labels=domain_labels,
        )
        all_results[f"layers_{nl}"] = {
            "num_layers": nl, "acc": result["test_acc_mean"],
            "f1": result["test_f1_mean"], "std": result["test_acc_std"],
        }
        logger.info(f"num_layers={nl}: acc={result['test_acc_mean']:.4f}±{result['test_acc_std']:.4f}")

    save_path = os.path.join(config["data"]["results_dir"], "sensitivity", "hyperparameter_sensitivity.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"超参数敏感性结果保存到 {save_path}")
    return all_results


# ---- Step Registry ----

STEPS = {
    "preprocess": step_preprocess,
    "encode": step_encode,
    "node_classification": step_node_classification,
    "link_prediction": step_link_prediction,
    "ablation": step_ablation,
    "few_shot": step_few_shot,
    "per_domain_ablation": step_per_domain_ablation,
    "cross_domain": step_cross_domain,
    "subcat_nc": step_subcat_nc,
    "statistics": step_statistics,
    "visualize": step_visualize,
    "sensitivity": step_hyperparameter_sensitivity,
}


def main():
    parser = argparse.ArgumentParser(description="AMMAG Benchmark Experiments")
    parser.add_argument("--step", type=str, required=True,
                        choices=list(STEPS.keys()) + ["all"],
                        help="实验步骤")
    parser.add_argument("--config", type=str, default="./benchmark/configs/default.yaml")
    parser.add_argument("--domains", nargs="+", default=None,
                        help="数据领域 (cs_si, cs_lo, math_lo, physics_socph)")
    parser.add_argument("--domain", type=str, default=None,
                        help="单域（用于subcat_nc）")
    parser.add_argument("--model", type=str, default=None,
                        help="模型名称 (mlp, gcn, gat, graphsage, gin, gt)")
    parser.add_argument("--modality", type=str, default=None,
                        help="模态 (text, image, formula, text+image+formula)")
    parser.add_argument("--modalities", nargs="+", default=None,
                        help="编码的模态列表")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--k", type=int, default=None, help="Few-shot k值")

    args = parser.parse_args()
    config = load_config(args.config)

    logger.info(f"Step: {args.step}, GPU: {args.gpu}")

    if args.step == "all":
        step_preprocess(args, config)
        step_encode(args, config)
        for model in config["models"]:
            for combo in config.get("modality_combos", ["text"]):
                args.model = model
                args.modality = combo
                step_node_classification(args, config)
        step_statistics(args, config)
        step_visualize(args, config)
    else:
        STEPS[args.step](args, config)


if __name__ == "__main__":
    main()
