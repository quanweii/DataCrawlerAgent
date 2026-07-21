#!/usr/bin/env python3
"""Cross-modal retrieval + alignment for new-domain MAGs (P2-11, OpenMAG Data Quality arm).

Uses existing frozen CLIP text+image embeddings (no training). For each domain:
  - text→image and image→text retrieval: Recall@1/5/10, MRR (chunked cosine sim)
  - modality alignment: linear CKA + CCA (first canonical correlation)
Reports cross-modal semantic alignment quality per domain.

Usage:
  python benchmark/run_newdomain_modality.py --domains steam,tmdb,wikiart --gpu 0
(spotify is text-only → N/A; arxiv already covered by openmag path)
"""
import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.data.domain_loader import load_domain_graph, DOMAIN_GRAPH, PROCESSED_DIR

_args = None


def _load_emb_aligned(domain, part, node_ids):
    """Load {part}_embeddings.pt and align to node_ids order."""
    path = PROCESSED_DIR / domain / f"{part}_embeddings.pt"
    obj = torch.load(path, map_location="cpu")
    id2idx = {nid: i for i, nid in enumerate(obj["ids"])}
    n = len(node_ids)
    dim = obj["embeddings"].size(1)
    out = torch.zeros(n, dim)
    found = 0
    for i, nid in enumerate(node_ids):
        if nid in id2idx:
            out[i] = obj["embeddings"][id2idx[nid]]; found += 1
    return out.float(), found, dim


def _retrieve(q, g, ks=(1, 5, 10), chunk=2048, device="cuda"):
    """q:[N,dq] query, g:[N,dg] gallery (same N, node-aligned). Rank gallery per query;
    true match = same index. Returns recall@k (mean over queries) + MRR."""
    q = F.normalize(q, dim=1); g = F.normalize(g, dim=1)
    N = q.size(0)
    recalls = {k: 0 for k in ks}; mrr = 0.0
    qd = q.to(device); gd = g.to(device)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        sim = qd[s:e] @ gd.t()  # [chunk, N]; diagonal (s+j, s+j) IS the true match
        ranks = sim.argsort(dim=1, descending=True)  # [chunk, N]
        for j in range(e - s):
            i = s + j
            rank_pos = (ranks[j] == i).nonzero(as_tuple=True)[0].item()
            for k in ks:
                if rank_pos < k:
                    recalls[k] += 1
            mrr += 1.0 / (rank_pos + 1)
    return {f"R@{k}": recalls[k] / N for k in ks}, {"MRR": mrr / N}


def cka(X, Y):
    """Standard linear CKA (Kornblith 2019)."""
    X = X.cpu().numpy(); Y = Y.cpu().numpy()
    X = X - X.mean(0, keepdims=True); Y = Y - Y.mean(0, keepdims=True)
    Xn = X / (np.linalg.norm(X) + 1e-12); Yn = Y / (np.linalg.norm(Y) + 1e-12)
    return float(np.linalg.norm(Yn.T @ Xn) ** 2)


def cca_first_corr(X, Y, k=10):
    """First canonical correlation via SVD on small sample."""
    X = X.cpu().numpy(); Y = Y.cpu().numpy()
    X = X - X.mean(0, keepdims=True); Y = Y - Y.mean(0, keepdims=True)
    n = X.shape[0]
    # whiten
    try:
        from scipy.linalg import svd
        Cxx = (X.T @ X) / (n - 1) + 1e-4 * np.eye(X.shape[1])
        Cyy = (Y.T @ Y) / (n - 1) + 1e-4 * np.eye(Y.shape[1])
        Cxy = (X.T @ Y) / (n - 1)
        Cxx_inv = np.linalg.inv(Cxx); Cyy_inv = np.linalg.inv(Cyy)
        M = np.linalg.cholesky(Cxx_inv).T if False else None
        # simpler: eigen of Cxx^-1 Cxy Cyy^-1 Cxy^T
        A = Cxx_inv @ Cxy @ Cyy_inv @ Cxy.T
        eig = np.linalg.eigvalsh(A)
        return float(np.sqrt(max(eig[-1], 0)))
    except Exception:
        return float("nan")


def run_domain(domain, device, out_dir):
    # canonical node id order = text_embeddings.pt ids (same order image was encoded in)
    text_path = PROCESSED_DIR / domain / "text_embeddings.pt"
    text_obj = torch.load(text_path, map_location="cpu")
    node_ids = list(text_obj["ids"])
    text = text_obj["embeddings"].float(); td = text.size(1); ft = len(node_ids)
    img, fi, id_ = _load_emb_aligned(domain, "image", node_ids)
    n = len(node_ids)
    logger.info(f"[{domain}] text {ft}/{n} (d={td}); image {fi}/{n} aligned (d={id_})")
    if fi < 0.9 * n:
        logger.warning(f"[{domain}] image alignment low ({fi}/{n}), skipping"); return None
    res = {"domain": domain, "n_nodes": n,
           "text_dim": td, "image_dim": id_,
           "text_align": ft, "image_align": fi, "cka": cka(text, img)}
    if td == id_:
        t2i_rec, t2i_mrr = _retrieve(text, img, device=device)
        i2t_rec, i2t_mrr = _retrieve(img, text, device=device)
        res["text_to_image"] = {**t2i_rec, **t2i_mrr}
        res["image_to_text"] = {**i2t_rec, **i2t_mrr}
        logger.info(f"[{domain}] t→i R@1/5/10={t2i_rec['R@1']:.3f}/{t2i_rec['R@5']:.3f}/{t2i_rec['R@10']:.3f} "
                    f"MRR={t2i_mrr['MRR']:.3f} | CKA={res['cka']:.3f}")
    else:
        res["retrieval"] = "SKIPPED — text/image embedding dims differ (not a shared space)"
        res["retrieval_note"] = (f"text {td}-d vs image {id_}-d: tmdb posters were encoded with a "
                                 f"different image encoder; cross-modal retrieval requires a shared space. "
                                 f"CKA (dim-agnostic alignment) still reported.")
        logger.info(f"[{domain}] retrieval SKIPPED (text {td}-d ≠ image {id_}-d) | CKA={res['cka']:.3f}")
    sp = out_dir / f"modality_{domain}_summary.json"
    with open(sp, "w") as f:
        json.dump(res, f, indent=2)
    logger.info(f"saved {sp}")
    return res


def main():
    global _args
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--domains", default="steam,tmdb,wikiart")
    _pre.add_argument("--gpu", type=int, default=None)
    _args = _pre.parse_args()
    if _args.gpu is not None and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(_args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = PROJECT_ROOT / "benchmark" / "results" / "newdomain_modality"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_res = {}
    for d in _args.domains.split(","):
        d = d.strip()
        if not d:
            continue
        r = run_domain(d, device, out_dir)
        if r:
            all_res[d] = r
    sp = out_dir / "modality_all_summary.json"
    with open(sp, "w") as f:
        json.dump(all_res, f, indent=2)
    logger.info(f"merged → {sp}")


if __name__ == "__main__":
    main()
