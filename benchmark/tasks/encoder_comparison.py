"""
多编码器对比实验 - OpenMAG Q3/Q4
对比不同编码器（CLIP, SentenceBERT, DINOv2, SPECTER, MathBERT）在768d统一空间下的表现
所有编码器输出统一投影到768d进行公平对比
"""
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
from loguru import logger


# 编码器注册表
ENCODER_REGISTRY = {
    "clip_large": {
        "name": "CLIP-ViT-Large-Patch14",
        "dim": 768,
        "type": "unified",
    },
    "sentence_bert": {
        "name": "all-MiniLM-L6-v2",
        "dim": 384,
        "type": "text_only",
    },
    "specter": {
        "name": "allenai/specter",
        "dim": 768,
        "type": "text_only",
    },
    "mathbert": {
        "name": "tbs17/MathBERT",
        "dim": 768,
        "type": "text_only",
    },
}


def encode_with_sentence_bert(texts: list, batch_size: int = 64, device: str = "cuda") -> torch.Tensor:
    """使用SentenceBERT编码文本"""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                               convert_to_tensor=True)
    return embeddings.cpu()


def encode_with_specter(texts: list, batch_size: int = 32, device: str = "cuda") -> torch.Tensor:
    """使用SPECTER编码学术文本"""
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
    model = AutoModel.from_pretrained("allenai/specter").to(device)
    model.eval()

    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512,
                           return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embs = outputs.last_hidden_state[:, 0, :]
        all_embs.append(embs.cpu())

    return torch.cat(all_embs, dim=0)


def encode_with_mathbert(texts: list, batch_size: int = 32, device: str = "cuda") -> torch.Tensor:
    """使用MathBERT编码数学相关文本"""
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("tbs17/MathBERT")
    model = AutoModel.from_pretrained("tbs17/MathBERT").to(device)
    model.eval()

    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512,
                           return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embs = outputs.last_hidden_state[:, 0, :]
        all_embs.append(embs.cpu())

    return torch.cat(all_embs, dim=0)


def load_or_encode_texts(encoder_name: str, paper_ids: list, texts: list,
                          cache_dir: str, device: str = "cuda") -> torch.Tensor:
    """加载缓存或重新编码文本特征"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{encoder_name}_text_embeddings.pt")

    if os.path.exists(cache_path):
        logger.info(f"加载缓存编码: {cache_path}")
        embs = torch.load(cache_path, map_location="cpu")
        if embs.size(0) == len(paper_ids):
            return embs
        logger.warning(f"缓存大小不匹配 ({embs.size(0)} vs {len(paper_ids)})，重新编码")

    logger.info(f"使用 {encoder_name} 编码 {len(texts)} 篇论文...")

    if encoder_name == "sentence_bert":
        embs = encode_with_sentence_bert(texts, device=device)
    elif encoder_name == "specter":
        embs = encode_with_specter(texts, device=device)
    elif encoder_name == "mathbert":
        embs = encode_with_mathbert(texts, device=device)
    else:
        raise ValueError(f"未知编码器: {encoder_name}")

    torch.save(embs, cache_path)
    logger.info(f"编码缓存到 {cache_path}, shape={embs.shape}")
    return embs


def extract_paper_texts(data_dir: str, paper_ids: list, domains: list) -> list:
    """从论文markdown文件中提取文本"""
    texts = []
    for pid in paper_ids:
        found = False
        for domain in domains:
            md_path = os.path.join(data_dir, f"{domain}_papers", pid, "paper.md")
            if os.path.exists(md_path):
                with open(md_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read(2000)
                texts.append(text)
                found = True
                break
        if not found:
            texts.append("")
    return texts


def run_encoder_comparison_experiment(
    dataset,
    data_dir: str = "./data",
    domains: list = None,
    encoder_names: list = None,
    model_name: str = "gcn",
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.5,
    lr: float = 0.001,
    num_epochs: int = 500,
    num_runs: int = 3,
    device: str = "cuda",
    output_dir: str = "./benchmark/results/encoder_comparison",
) -> Dict:
    """
    多编码器对比实验：不同编码器 + 统一768d投影 → GNN分类
    """
    from benchmark.models.gnn import get_model

    if domains is None:
        domains = ["cs_si", "cs_lo", "math_lo", "physics_socph"]
    if encoder_names is None:
        encoder_names = ["clip_large", "sentence_bert", "specter", "mathbert"]

    os.makedirs(output_dir, exist_ok=True)

    data = dataset.data
    paper_ids = []
    if hasattr(dataset, '_graph_raw') and 'paper_ids' in dataset._graph_raw:
        paper_ids = dataset._graph_raw['paper_ids']
    elif hasattr(dataset, '_paper_ids'):
        paper_ids = dataset._paper_ids
    elif hasattr(data, 'paper_ids'):
        paper_ids = list(data.paper_ids)

    # 准备文本数据
    logger.info(f"提取 {len(paper_ids)} 篇论文的文本...")
    texts = extract_paper_texts(data_dir, paper_ids, domains)

    # 获取分割masks
    train_mask, val_mask, test_mask = dataset.get_split_masks()
    labels = data.y
    edge_index = data.edge_index

    all_results = {}

    for enc_name in encoder_names:
        enc_info = ENCODER_REGISTRY[enc_name]
        logger.info(f"\n{'='*50}")
        logger.info(f"编码器: {enc_info['name']} (dim={enc_info['dim']})")
        logger.info(f"{'='*50}")

        # 加载编码
        cache_dir = os.path.join(data_dir, "encoder_cache")
        if enc_name == "clip_large":
            x = data.x.clone()
        else:
            x_raw = load_or_encode_texts(enc_name, paper_ids, texts, cache_dir, device)
            # 统一投影到768d
            raw_dim = x_raw.size(1)
            proj = nn.Linear(raw_dim, 768)
            nn.init.xavier_normal_(proj.weight)
            with torch.no_grad():
                x = proj(x_raw)
            del proj

        x = x.to(device)
        in_dim = x.size(1)

        run_results = []
        for run in range(num_runs):
            seed = run * 42 + 1
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = get_model(model_name, in_dim, hidden_dim,
                              int(labels.max()) + 1,
                              num_layers=num_layers, dropout=dropout).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

            best_val_acc = 0
            best_state = None
            patience_counter = 0

            for epoch in range(1, num_epochs + 1):
                model.train()
                optimizer.zero_grad()
                out = model(x, edge_index.to(device))
                loss = F.cross_entropy(out[train_mask], labels[train_mask].to(device))
                loss.backward()
                optimizer.step()

                if epoch % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        out = model(x, edge_index.to(device))
                        pred = out.argmax(dim=1)
                        val_acc = (pred[val_mask] == labels[val_mask].to(device)).float().mean().item()
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                            patience_counter = 0
                        else:
                            patience_counter += 10
                        if patience_counter >= 50:
                            break

            # 测试
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                out = model(x, edge_index.to(device))
                pred = out.argmax(dim=1)
                test_acc = (pred[test_mask] == labels[test_mask].to(device)).float().mean().item()

            run_results.append({"test_accuracy": test_acc})
            logger.info(f"[Run {run+1}/{num_runs}] {enc_name}: test_acc={test_acc:.4f}")

        accs = [r["test_accuracy"] for r in run_results]
        summary = {
            "encoder": enc_name,
            "encoder_full_name": enc_info["name"],
            "raw_dim": enc_info["dim"],
            "projected_dim": 768,
            "model": model_name,
            "test_accuracy_mean": float(np.mean(accs)),
            "test_accuracy_std": float(np.std(accs)),
            "num_runs": num_runs,
            "runs": run_results,
        }
        all_results[enc_name] = summary

    # 保存结果
    save_path = os.path.join(output_dir, "encoder_comparison_results.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"编码器对比结果保存到 {save_path}")

    # 打印对比表
    logger.info(f"\n{'='*60}")
    logger.info("编码器对比结果")
    logger.info(f"{'='*60}")
    for enc_name, res in all_results.items():
        logger.info(f"  {res['encoder_full_name']:<35} Acc={res['test_accuracy_mean']:.4f}±{res['test_accuracy_std']:.4f}")

    return all_results
