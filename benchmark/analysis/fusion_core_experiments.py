"""
Proj-Concat融合方法的核心实验重跑
1. 每域消融 (GCN, 7模态组合, 5 runs)
2. Few-shot (GCN, 4模态, 3-shot + 10-shot, 10 runs)
"""
import os, sys, json, time
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


class ProjConcatGCN(nn.Module):
    """各模态投影到统一维度后拼接 + GCN"""
    def __init__(self, modality_dims, proj_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, proj_dim)
            for name, dim in modality_dims.items()
        })
        fused_dim = proj_dim * len(modality_dims)

        from torch_geometric.nn import GCNConv
        self.convs = nn.ModuleList([
            GCNConv(fused_dim, hidden_dim),
            GCNConv(hidden_dim, out_dim),
        ])
        self.dropout = dropout

    def forward(self, modality_embs, edge_index):
        projected = [self.projections[n](modality_embs[n]) for n in self.modality_names]
        x = torch.cat(projected, dim=1)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class ConcatGCN(nn.Module):
    """简单拼接 + GCN (基线)"""
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.convs = nn.ModuleList([
            GCNConv(in_dim, hidden_dim),
            GCNConv(hidden_dim, out_dim),
        ])
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


def train_eval(model, x_or_embs, edge_index, labels, train_mask, val_mask, test_mask,
               num_epochs=500, lr=0.01, device="cuda", use_modality_dict=False):
    model = model.to(device)
    edge_index = edge_index.to(device)
    labels = labels.to(device)
    if use_modality_dict:
        for k in x_or_embs:
            x_or_embs[k] = x_or_embs[k].to(device)
    else:
        x_or_embs = x_or_embs.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    best_val, best_state, patience = 0, None, 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x_or_embs, edge_index) if use_modality_dict else model(x_or_embs, edge_index)
        loss = criterion(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                out = model(x_or_embs, edge_index) if use_modality_dict else model(x_or_embs, edge_index)
                pred = out.argmax(dim=-1)
                va = accuracy_score(labels[val_mask].cpu(), pred[val_mask].cpu())
                if va > best_val:
                    best_val = va
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 10
                if patience >= 50:
                    break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(x_or_embs, edge_index) if use_modality_dict else model(x_or_embs, edge_index)
        pred = out.argmax(dim=-1)
        acc = accuracy_score(labels[test_mask].cpu(), pred[test_mask].cpu())
        f1 = f1_score(labels[test_mask].cpu(), pred[test_mask].cpu(), average="macro")
    return {"test_acc": float(acc), "test_f1": float(f1)}


def load_data():
    processed = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "processed")
    graph = torch.load(os.path.join(processed, "graph_data.pt"), map_location="cpu")
    text = torch.load(os.path.join(processed, "text_embeddings.pt"), map_location="cpu")
    image = torch.load(os.path.join(processed, "image_embeddings.pt"), map_location="cpu")
    formula = torch.load(os.path.join(processed, "formula_embeddings.pt"), map_location="cpu")

    ids = graph["paper_ids"]
    t_map = {pid: i for i, pid in enumerate(text["ids"])}
    i_map = {pid: i for i, pid in enumerate(image["ids"])}
    f_map = {pid: i for i, pid in enumerate(formula["ids"])}

    N = len(ids)
    t_emb = text["embeddings"][torch.tensor([t_map[pid] for pid in ids])]
    i_emb = image["embeddings"][torch.tensor([i_map[pid] for pid in ids])]
    f_emb = formula["embeddings"][torch.tensor([f_map[pid] for pid in ids])]
    labels = graph["labels"]
    domain_labels = graph.get("domain_labels", graph["labels"])
    ei = graph["edge_index"]
    num_classes = int(labels.max().item()) + 1

    return t_emb, i_emb, f_emb, ei, labels, domain_labels, num_classes, N


def get_split(N, seed, split="supervised", few_shot_k=-1, labels=None, num_classes=4):
    rng = np.random.RandomState(seed)
    if split == "few_shot" and few_shot_k > 0:
        train_mask = torch.zeros(N, dtype=torch.bool)
        for c in range(num_classes):
            idxs = torch.where(labels == c)[0].numpy()
            rng.shuffle(idxs)
            train_mask[idxs[:few_shot_k]] = True
        remaining = torch.where(~train_mask)[0].numpy()
        rng.shuffle(remaining)
        v = int(len(remaining) * 0.2)
        val_mask = torch.zeros(N, dtype=torch.bool)
        test_mask = torch.zeros(N, dtype=torch.bool)
        val_mask[remaining[:v]] = True
        test_mask[remaining[v:]] = True
    else:
        perm = rng.permutation(N)
        t = int(N * 0.6)
        v = int(N * 0.2)
        train_mask = torch.zeros(N, dtype=torch.bool)
        val_mask = torch.zeros(N, dtype=torch.bool)
        test_mask = torch.zeros(N, dtype=torch.bool)
        train_mask[perm[:t]] = True
        val_mask[perm[t:t+v]] = True
        test_mask[perm[t+v:]] = True
    return train_mask, val_mask, test_mask


def per_domain_acc(pred, labels, domain_labels, test_mask):
    result = {}
    test_pred = pred[test_mask]
    test_labels = labels[test_mask]
    test_domains = domain_labels[test_mask]
    for d in range(4):
        mask = (test_domains == d)
        if mask.sum() > 0:
            acc = accuracy_score(test_labels[mask].cpu(), test_pred[mask].cpu())
            result[str(d)] = float(acc)
    return result


def main():
    device = "cuda:3"
    t_emb, i_emb, f_emb, ei, labels, domain_labels, num_classes, N = load_data()
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "results", "fusion")
    os.makedirs(output_dir, exist_ok=True)

    ALL_DIMS = {"text": 384, "image": 512, "formula": 384}
    MODALITY_CONFIGS = {
        "text": ["text"],
        "image": ["image"],
        "formula": ["formula"],
        "text+image": ["text", "image"],
        "text+formula": ["text", "formula"],
        "image+formula": ["image", "formula"],
        "text+image+formula": ["text", "image", "formula"],
    }
    EMB_MAP = {"text": t_emb, "image": i_emb, "formula": f_emb}

    # ============ Experiment 1: Per-domain ablation ============
    logger.info("=" * 60)
    logger.info("Experiment 1: Per-domain ablation (Concat vs Proj-Concat)")
    logger.info("=" * 60)

    ablation_results = {}
    for combo_name, mods in MODALITY_CONFIGS.items():
        for fusion_type in ["concat", "proj_concat"]:
            key = f"{combo_name}_{fusion_type}"
            run_accs = []
            run_pd = []

            for run in range(5):
                seed = run * 42 + 1
                tr, va, te = get_split(N, seed, "supervised", labels=labels, num_classes=num_classes)

                if fusion_type == "proj_concat":
                    mod_dims = {m: ALL_DIMS[m] for m in mods}
                    model = ProjConcatGCN(mod_dims, proj_dim=256, hidden_dim=256,
                                          out_dim=num_classes)
                    mod_embs = {m: EMB_MAP[m].clone() for m in mods}
                    r = train_eval(model, mod_embs, ei, labels, tr, va, te, device=device,
                                   use_modality_dict=True)
                else:
                    x = torch.cat([EMB_MAP[m] for m in mods], dim=1)
                    model = ConcatGCN(x.size(1), 256, num_classes)
                    r = train_eval(model, x, ei, labels, tr, va, te, device=device,
                                   use_modality_dict=False)

                run_accs.append(r["test_acc"])

                # Per-domain
                model.eval()
                with torch.no_grad():
                    if fusion_type == "proj_concat":
                        mod_embs2 = {m: EMB_MAP[m].clone().to(device) for m in mods}
                        out = model(mod_embs2, ei.to(device))
                    else:
                        out = model(x.to(device), ei.to(device))
                    pred = out.argmax(dim=-1).cpu()
                pd = per_domain_acc(pred, labels, domain_labels, te)
                run_pd.append(pd)

            ablation_results[key] = {
                "acc_mean": float(np.mean(run_accs)),
                "acc_std": float(np.std(run_accs)),
                "per_domain": {
                    d: {"mean": float(np.mean([p[d] for p in run_pd])),
                        "std": float(np.std([p[d] for p in run_pd]))}
                    for d in run_pd[0]
                }
            }
            logger.info(f"  {key}: {np.mean(run_accs)*100:.2f}±{np.std(run_accs)*100:.2f}%")

    # ============ Experiment 2: Few-shot ============
    logger.info("\n" + "=" * 60)
    logger.info("Experiment 2: Few-shot (Concat vs Proj-Concat)")
    logger.info("=" * 60)

    fewshot_configs = [
        ("text", ["text"]),
        ("text+formula", ["text", "formula"]),
        ("text+image+formula", ["text", "image", "formula"]),
    ]

    fewshot_results = {}
    for combo_name, mods in fewshot_configs:
        for fusion_type in ["concat", "proj_concat"]:
            for k in [3, 10]:
                key = f"{combo_name}_{fusion_type}_k{k}"
                run_accs = []
                for run in range(10):
                    seed = run * 42 + 1
                    tr, va, te = get_split(N, seed, "few_shot", few_shot_k=k,
                                           labels=labels, num_classes=num_classes)
                    if fusion_type == "proj_concat":
                        mod_dims = {m: ALL_DIMS[m] for m in mods}
                        model = ProjConcatGCN(mod_dims, proj_dim=256, hidden_dim=256,
                                              out_dim=num_classes)
                        mod_embs = {m: EMB_MAP[m].clone() for m in mods}
                        r = train_eval(model, mod_embs, ei, labels, tr, va, te, device=device,
                                       use_modality_dict=True)
                    else:
                        x = torch.cat([EMB_MAP[m] for m in mods], dim=1)
                        model = ConcatGCN(x.size(1), 256, num_classes)
                        r = train_eval(model, x, ei, labels, tr, va, te, device=device,
                                       use_modality_dict=False)
                    run_accs.append(r["test_acc"])

                fewshot_results[key] = {
                    "acc_mean": float(np.mean(run_accs)),
                    "acc_std": float(np.std(run_accs)),
                }
                logger.info(f"  {key}: {np.mean(run_accs)*100:.2f}±{np.std(run_accs)*100:.2f}%")

    # Save
    all_results = {"ablation": ablation_results, "fewshot": fewshot_results}
    with open(os.path.join(output_dir, "proj_concat_full.json"), "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 70)
    print("Per-Domain Ablation: Concat vs Proj-Concat (GCN)")
    print("=" * 70)
    print(f"{'模态':<20} {'Concat':>12} {'Proj-Concat':>12} {'Δ':>8}")
    print("-" * 55)
    for combo in MODALITY_CONFIGS:
        c = ablation_results[f"{combo}_concat"]
        p = ablation_results[f"{combo}_proj_concat"]
        delta = (p["acc_mean"] - c["acc_mean"]) * 100
        print(f"{combo:<20} {c['acc_mean']*100:>7.2f}±{c['acc_std']*100:.2f}% "
              f"{p['acc_mean']*100:>7.2f}±{p['acc_std']*100:.2f}% {delta:>+7.2f}")

    print("\n" + "=" * 70)
    print("Per-Domain Detail: T+I+F Concat vs Proj-Concat")
    print("=" * 70)
    domain_names = {"0": "cs.SI", "1": "cs.LO", "2": "math.LO", "3": "phys.soc-ph"}
    c_pd = ablation_results["text+image+formula_concat"]["per_domain"]
    p_pd = ablation_results["text+image+formula_proj_concat"]["per_domain"]
    print(f"{'域':<15} {'Concat':>10} {'Proj-Concat':>12} {'Δ':>8}")
    print("-" * 50)
    for d, name in domain_names.items():
        if d in c_pd and d in p_pd:
            delta = (p_pd[d]["mean"] - c_pd[d]["mean"]) * 100
            print(f"{name:<15} {c_pd[d]['mean']*100:>7.2f}% {p_pd[d]['mean']*100:>7.2f}% {delta:>+7.2f}")

    print("\n" + "=" * 70)
    print("Few-shot: Concat vs Proj-Concat")
    print("=" * 70)
    for combo in ["text", "text+formula", "text+image+formula"]:
        for k in [3, 10]:
            ck = fewshot_results.get(f"{combo}_concat_k{k}", {})
            pk = fewshot_results.get(f"{combo}_proj_concat_k{k}", {})
            if ck and pk:
                delta = (pk["acc_mean"] - ck["acc_mean"]) * 100
                print(f"  {combo} {k}-shot: Concat={ck['acc_mean']*100:.2f}% "
                      f"Proj={pk['acc_mean']*100:.2f}% Δ={delta:+.2f}pp")

    logger.info(f"\nResults saved to {output_dir}/proj_concat_full.json")


if __name__ == "__main__":
    main()
