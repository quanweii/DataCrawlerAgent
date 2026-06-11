"""
跨域迁移实验 - 正确的transfer learning评估
1. 在source domain子图上预训练GNN (subcat labels)
2. 冻结encoder，在target domain上训练linear probe
3. 对比：从零训练 vs 预训练+微调
"""
import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score


def run_cross_domain_transfer(
    dataset,
    model_name: str,
    modality: str,
    source_domain: str,
    target_domain: str,
    num_runs: int = 3,
    num_epochs: int = 500,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.5,
    lr: float = 0.01,
    device: str = "cuda",
    output_dir: str = "./benchmark/results",
) -> Dict:
    """
    跨域迁移：
    1. 在source domain子图上训练GNN (subcat labels)
    2. 复制encoder权重（GNN卷积层），跳过分类头
    3. 在target domain上微调全部参数
    4. 对比from-scratch vs 预训练+微调
    """
    from benchmark.models.gnn import get_model
    from benchmark.data.dataset import AMMAGDataset

    os.makedirs(output_dir, exist_ok=True)

    ds_subcat = AMMAGDataset(
        root=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        modality=modality, label_type="subcat"
    )
    data = ds_subcat.data.to(device)

    src_indices = ds_subcat.get_domain_indices(source_domain)
    tgt_indices = ds_subcat.get_domain_indices(target_domain)

    src_subcat = ds_subcat.data.y[src_indices]
    tgt_subcat = ds_subcat.data.y[tgt_indices]

    src_num_classes = int(src_subcat.max().item()) + 1
    tgt_num_classes = int(tgt_subcat.max().item()) + 1

    in_dim = data.x.size(1)

    all_results = []
    for run in range(num_runs):
        seed = run * 42 + 1
        rng = np.random.RandomState(seed)

        # === Phase 1: 在source domain子图上预训练 ===
        src_subgraph = ds_subcat.get_domain_subgraph(source_domain).to(device)
        n_src = src_subgraph.num_nodes
        perm = rng.permutation(n_src)
        src_train_size = int(n_src * 0.6)
        src_val_size = int(n_src * 0.2)

        src_train_mask = torch.zeros(n_src, dtype=torch.bool)
        src_val_mask = torch.zeros(n_src, dtype=torch.bool)
        src_test_mask = torch.zeros(n_src, dtype=torch.bool)
        src_train_mask[perm[:src_train_size]] = True
        src_val_mask[perm[src_train_size:src_train_size + src_val_size]] = True
        src_test_mask[perm[src_train_size + src_val_size:]] = True

        pretrain_model = get_model(
            model_name, in_dim, hidden_dim, src_num_classes,
            num_layers=num_layers, dropout=dropout,
        ).to(device)

        optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=lr, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        best_src_acc = 0
        best_state = None

        for epoch in range(1, num_epochs + 1):
            pretrain_model.train()
            optimizer.zero_grad()
            out = pretrain_model(src_subgraph.x, src_subgraph.edge_index)
            loss = criterion(out[src_train_mask], src_subgraph.y[src_train_mask])
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                pretrain_model.eval()
                with torch.no_grad():
                    out = pretrain_model(src_subgraph.x, src_subgraph.edge_index)
                    pred = out.argmax(dim=-1)
                    val_acc = accuracy_score(
                        src_subgraph.y[src_val_mask].cpu(), pred[src_val_mask].cpu()
                    )
                    if val_acc > best_src_acc:
                        best_src_acc = val_acc
                        best_state = {k: v.cpu().clone() for k, v in pretrain_model.state_dict().items()}

        pretrain_model.load_state_dict(best_state)
        pretrain_model.eval()
        with torch.no_grad():
            out = pretrain_model(src_subgraph.x, src_subgraph.edge_index)
            pred = out.argmax(dim=-1)
            src_test_acc = accuracy_score(
                src_subgraph.y[src_test_mask].cpu(), pred[src_test_mask].cpu()
            )

        # === Phase 2: Transfer to target domain ===
        tgt_subgraph = ds_subcat.get_domain_subgraph(target_domain).to(device)
        n_tgt = tgt_subgraph.num_nodes
        perm_tgt = rng.permutation(n_tgt)
        tgt_train_size = int(n_tgt * 0.6)
        tgt_val_size = int(n_tgt * 0.2)

        tgt_train_mask = torch.zeros(n_tgt, dtype=torch.bool)
        tgt_val_mask = torch.zeros(n_tgt, dtype=torch.bool)
        tgt_test_mask = torch.zeros(n_tgt, dtype=torch.bool)
        tgt_train_mask[perm_tgt[:tgt_train_size]] = True
        tgt_val_mask[perm_tgt[tgt_train_size:tgt_train_size + tgt_val_size]] = True
        tgt_test_mask[perm_tgt[tgt_train_size + tgt_val_size:]] = True

        # --- Method A: From scratch baseline ---
        scratch_model = get_model(
            model_name, in_dim, hidden_dim, tgt_num_classes,
            num_layers=num_layers, dropout=dropout,
        ).to(device)
        _train_on_subgraph(scratch_model, tgt_subgraph, tgt_train_mask, tgt_val_mask,
                           num_epochs, lr, device)
        scratch_result = _eval_on_subgraph(scratch_model, tgt_subgraph, tgt_test_mask, device)

        # --- Method B: Pretrained + finetune ---
        pretrained_model = get_model(
            model_name, in_dim, hidden_dim, tgt_num_classes,
            num_layers=num_layers, dropout=dropout,
        ).to(device)

        _transfer_encoder_weights(best_state, pretrained_model, num_layers)

        _train_on_subgraph(pretrained_model, tgt_subgraph, tgt_train_mask, tgt_val_mask,
                           num_epochs, lr, device)
        finetune_result = _eval_on_subgraph(pretrained_model, tgt_subgraph, tgt_test_mask, device)

        result = {
            "source_test_acc": float(src_test_acc),
            "target_scratch_acc": float(scratch_result["acc"]),
            "target_scratch_f1": float(scratch_result["f1"]),
            "target_finetune_acc": float(finetune_result["acc"]),
            "target_finetune_f1": float(finetune_result["f1"]),
            "transfer_gain": float(finetune_result["acc"] - scratch_result["acc"]),
        }
        all_results.append(result)
        logger.info(f"[Run {run+1}/{num_runs}] {source_domain}→{target_domain} "
                     f"{model_name}+{modality}: "
                     f"src={src_test_acc:.4f}, scratch={scratch_result['acc']:.4f}, "
                     f"finetune={finetune_result['acc']:.4f}, "
                     f"gain={finetune_result['acc']-scratch_result['acc']:+.4f}")

    # 汇总
    summary = {}
    for m in ["source_test_acc", "target_scratch_acc", "target_scratch_f1",
              "target_finetune_acc", "target_finetune_f1", "transfer_gain"]:
        values = [r[m] for r in all_results]
        summary[f"{m}_mean"] = float(np.mean(values))
        summary[f"{m}_std"] = float(np.std(values))

    summary.update({
        "num_runs": num_runs, "model": model_name, "modality": modality,
        "source_domain": source_domain, "target_domain": target_domain,
        "task": "cross_domain", "runs": all_results,
    })

    save_name = f"cd_{source_domain}_to_{target_domain}_{model_name}_{modality}"
    save_path = os.path.join(output_dir, f"{save_name}.json")
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"跨域迁移结果: {source_domain}→{target_domain}, "
                 f"scratch={summary['target_scratch_acc_mean']:.4f}, "
                 f"finetune={summary['target_finetune_acc_mean']:.4f}, "
                 f"gain={summary['transfer_gain_mean']:+.4f}")
    return summary


def _train_on_subgraph(model, subgraph, train_mask, val_mask,
                       num_epochs, lr, device, patience=50):
    """在子图上训练模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(subgraph.x, subgraph.edge_index)
        loss = criterion(out[train_mask], subgraph.y[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                out = model(subgraph.x, subgraph.edge_index)
                pred = out.argmax(dim=-1)
                val_acc = accuracy_score(
                    subgraph.y[val_mask].cpu(), pred[val_mask].cpu()
                )
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 10
                if patience_counter >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)


def _eval_on_subgraph(model, subgraph, test_mask, device):
    """在子图上评估，返回acc和f1"""
    model.eval()
    with torch.no_grad():
        out = model(subgraph.x, subgraph.edge_index)
        pred = out.argmax(dim=-1)
        acc = accuracy_score(subgraph.y[test_mask].cpu(), pred[test_mask].cpu())
        f1 = f1_score(subgraph.y[test_mask].cpu(), pred[test_mask].cpu(), average="macro")
    return {"acc": float(acc), "f1": float(f1)}


def _transfer_encoder_weights(source_state, target_model, num_layers):
    """
    正确的encoder权重复制：
    只复制GNN卷积层（convs[0]到convs[-2]）的参数，跳过最后一层分类头（convs[-1]）。
    所有GNN模型（GCN/GAT/GraphSAGE/GIN/GT）都使用 self.convs 存储层参数。
    MLP使用 self.layers，最后一层也是分类头。
    """
    target_dict = target_model.state_dict()

    # 识别最后一层（分类头）的参数名前缀
    # convs[-1] 的参数名格式为 "convs.{num_layers-1}.xxx"
    # layers[-1] 的参数名格式为 "layers.{num_layers-1}.xxx"
    last_conv_prefix = f"convs.{num_layers - 1}."
    last_layer_prefix = f"layers.{num_layers - 1}."

    transferred_keys = []
    skipped_keys = []

    for k, v in source_state.items():
        # 跳过分类头（最后一个conv/linear层）
        if k.startswith(last_conv_prefix) or k.startswith(last_layer_prefix):
            skipped_keys.append(k)
            continue
        # 只复制维度匹配的encoder参数
        if k in target_dict and v.shape == target_dict[k].shape:
            target_dict[k] = v
            transferred_keys.append(k)

    target_model.load_state_dict(target_dict)
    logger.debug(f"权重复制: 传输{len(transferred_keys)}个参数, "
                 f"跳过分类头{len(skipped_keys)}个参数")
