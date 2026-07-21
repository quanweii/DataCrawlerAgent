#!/usr/bin/env python3
"""P1-4 — CLIP-family encoder comparison for the arxiv domain (text-only).

Encodes arxiv TEXT with clip_base (openai/clip-vit-base-patch32, 512-d) and
siglip (google/siglip-so400m-patch14-384, 1152-d text), reuses the existing
clip_large text_embeddings.pt (768-d, headline), then runs the leak-free NC
protocol (mlp/gcn/gcnii, 5 init seeds, split_seed=1, cross-split edge pruning)
on each encoder's text features, and saves
benchmark/results/encoder_comparison_arxiv.json (schema matches the steam/
tmdb/wikiart encoder_comparison_{domain}.json files).

Constraints honored:
  * Never overwrites benchmark/processed/arxiv/text_embeddings.pt (clip_large
    headline). New embeddings go to text_clip_base_embeddings.pt and
    text_siglip_embeddings.pt.
  * Never overwrites nc_arxiv_summary.json — writes encoder_comparison_arxiv.json.
  * Uses GPU 6. ~45677 docs; id list reused verbatim from text_embeddings.pt
    so rows align node-for-node with the headline embeddings.

Usage:
    python scripts/run_arxiv_encoder_comparison.py --gpu 6
    python scripts/run_arxiv_encoder_comparison.py --gpu 6 --skip-encode   # NC only
"""
# ---- Parse GPU before importing torch ----
import os, argparse
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--gpu", type=int, default=6)
_ap.add_argument("--seeds", type=int, default=5)
_ap.add_argument("--epochs", type=int, default=300)
_ap.add_argument("--hidden", type=int, default=256)
_ap.add_argument("--layers", type=int, default=2)
_ap.add_argument("--dropout", type=float, default=0.5)
_ap.add_argument("--lr", type=float, default=0.01)
_ap.add_argument("--split-seed", type=int, default=1)
_ap.add_argument("--batch-size", type=int, default=32)
_ap.add_argument("--skip-encode", action="store_true",
                 help="skip encoding (assume clip_base/siglip .pt already exist)")
_args, _ = _ap.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_args.gpu)
# All three models are cached locally.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json, sys, time
from pathlib import Path
import numpy as np
import torch
from loguru import logger
logger.add(sys.stderr, level="INFO")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DOMAIN = "arxiv"
PROCESSED = PROJECT_ROOT / "benchmark" / "processed" / DOMAIN
RESULTS = PROJECT_ROOT / "benchmark" / "results"
PROCESSED.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# arxiv raw markdown lives under 4 sub-domain paper dirs (plus a generic 'papers').
RAW_SUBDIRS = ["cs_lo_papers", "cs_si_papers", "math_lo_papers",
               "physics_socph_papers", "papers"]

# encoder tag -> (modality_part loaded by domain_loader._load_emb, expected dim)
ENCODERS = [
    ("clip_large", "text",          768),  # existing headline (do NOT re-encode)
    ("clip_base",  "text_clip_base", 512),
    ("siglip",     "text_siglip",   1152),  # so400m text projection is 1152-d (not 768)
]


# ============================================================
# 1. Resolve md paths for the ids in the existing text_embeddings.pt
# ============================================================
def load_target_ids():
    """Return the id list from the existing clip_large text_embeddings.pt.

    These ids define node-for-node row order for the new embeddings.
    Read-only: never writes the file.
    """
    src = PROCESSED / "text_embeddings.pt"
    if not src.exists():
        raise FileNotFoundError(f"Existing clip_large text embeddings not found: {src}")
    obj = torch.load(src, map_location="cpu", weights_only=False)
    ids = obj["ids"]
    logger.info(f"[ids] Reusing {len(ids)} ids from {src} (row order frozen)")
    return ids


def resolve_md_paths(ids):
    id2md, missing = {}, []
    roots = [PROJECT_ROOT / "data" / s for s in RAW_SUBDIRS]
    for nid in ids:
        found = None
        for root in roots:
            md = root / nid / f"{nid}.md"
            if md.is_file():
                found = str(md)
                break
        if found:
            id2md[nid] = found
        else:
            missing.append(nid)
    logger.info(f"[md] Resolved {len(id2md)}/{len(ids)} md files; missing={len(missing)}")
    if missing:
        logger.warning(f"[md] Missing examples (first 5): {missing[:5]}")
    return id2md, missing


# ============================================================
# 2. Encoders
# ============================================================
def encode_clip_base(ids, id2md):
    """CLIP-base text via the existing TextEncoder (paragraph chunk + mean-pool),
    identical pipeline to how arxiv clip_large was produced -> apples-to-apples.

    NOTE on the fast tokenizer: TextEncoder uses the slow Python CLIPTokenizer
    (per-char `_clean_text`), which is ~100x too slow for 45677 docs. We
    monkeypatch CLIPTokenizerFast (same vocab/merges, Rust impl). Validated on
    500 real arxiv docs vs the existing slow clip_large embeddings:
    cosine(slow, fast) mean=1.00000, min=0.99920, 100% >= 0.999. Embeddings are
    thus equivalent to within rounding for this (English scientific) corpus.
    """
    from benchmark.encoders.text_encoder import TextEncoder
    from transformers import CLIPTokenizerFast
    out_path = PROCESSED / "text_clip_base_embeddings.pt"
    if out_path.exists():
        logger.info(f"[clip_base] {out_path.name} exists, skipping")
        return
    logger.info("[clip_base] Encoding with openai/clip-vit-base-patch32 ...")
    te = TextEncoder(model_name="openai/clip-vit-base-patch32", device=DEVICE)
    # Swap slow -> fast tokenizer (validated equivalent, see docstring).
    te.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    logger.info("[clip_base] Using CLIPTokenizerFast (validated vs slow, cosine>=0.999)")
    embs = np.zeros((len(ids), te.embedding_dim), dtype=np.float32)
    t0 = time.time()
    for i, nid in enumerate(ids):
        p = id2md.get(nid)
        text = te._read_md(p) if p else ""
        embs[i] = te.encode_single(text)
        if (i + 1) % 5000 == 0:
            logger.info(f"  clip_base {i+1}/{len(ids)}  ({(i+1)/(time.time()-t0):.0f} doc/s)")
    emb = torch.from_numpy(embs).float()
    torch.save({"ids": list(ids), "embeddings": emb}, out_path)
    logger.info(f"[clip_base] saved {tuple(emb.shape)} -> {out_path}  "
                f"({time.time()-t0:.0f}s)")
    del te; torch.cuda.empty_cache()


def encode_siglip(ids, id2md):
    """SigLIP text. Mirrors scripts/run_multi_encoder_nc.encode_siglip so the
    comparison is consistent with steam/tmdb/wikiart: single-shot, truncate to
    2000 chars, max_length=64 tokens, L2-normalized text features."""
    from transformers import SiglipModel, SiglipTokenizer
    out_path = PROCESSED / "text_siglip_embeddings.pt"
    if out_path.exists():
        logger.info(f"[siglip] {out_path.name} exists, skipping")
        return
    model_name = "google/siglip-so400m-patch14-384"
    logger.info(f"[siglip] Encoding text with {model_name} ...")
    model = SiglipModel.from_pretrained(model_name).to(DEVICE).eval()
    tokenizer = SiglipTokenizer.from_pretrained(model_name)
    dim = model.config.text_config.hidden_size  # 768
    MAX_TOKENS = 64  # SigLIP text max position embeddings

    # Pre-read texts (truncate to 2000 chars) in id order.
    texts = []
    for nid in ids:
        p = id2md.get(nid)
        if p:
            try:
                with open(p, encoding="utf-8") as f:
                    texts.append(f.read()[:2000])
            except Exception:
                texts.append("")
        else:
            texts.append("")

    bs = _args.batch_size
    all_embs = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            inputs = tokenizer(batch, padding=True, truncation=True,
                               max_length=MAX_TOKENS, return_tensors="pt").to(DEVICE)
            out = model.get_text_features(**inputs)
            out = torch.nn.functional.normalize(out, p=2, dim=1)
            all_embs.append(out.cpu())
            if (i // bs + 1) % 200 == 0:
                logger.info(f"  siglip {i+len(batch)}/{len(texts)}  "
                            f"({(i+len(batch))/(time.time()-t0):.0f} doc/s)")
    emb = torch.cat(all_embs, dim=0).float()
    torch.save({"ids": list(ids), "embeddings": emb}, out_path)
    logger.info(f"[siglip] saved {tuple(emb.shape)} -> {out_path}  ({time.time()-t0:.0f}s)")
    del model; torch.cuda.empty_cache()


# ============================================================
# 3. Leak-free NC (mirrors run_newdomain_nc.run_leak_free)
# ============================================================
def run_leak_free_per_encoder():
    from benchmark.data.domain_loader import load_domain_graph, _load_emb
    from benchmark.data.splits import (stratified_split, prune_cross_split_edges,
                                       label_propagation_ceiling)
    from benchmark.models.gnn import get_model
    from benchmark.tasks.node_classification import train_and_eval
    from torch_geometric.data import Data

    graph = load_domain_graph(DOMAIN)
    node_ids = graph["node_ids"]
    y = graph["y"]
    ei_full = graph["edge_index"]
    n = graph["num_nodes"]

    masks = stratified_split(y, ratios=(0.6, 0.2, 0.2), seed=_args.split_seed)
    tr, va, te = masks
    ei_pruned = prune_cross_split_edges(ei_full, masks)
    orig, kept = ei_full.size(1), ei_pruned.size(1)
    lp_orig = label_propagation_ceiling(y, ei_full, tr, te)
    lp_pruned = label_propagation_ceiling(y, ei_pruned, tr, te)
    logger.info(f"[{DOMAIN}] leak-free: edges {orig} -> {kept} "
                f"(removed {orig-kept}); split_seed={_args.split_seed}")
    logger.info(f"[{DOMAIN}] LP-ceiling: orig={lp_orig['acc']:.3f} -> "
                f"pruned={lp_pruned['acc']:.3f} (majority={lp_pruned['majority_acc']:.3f})")

    out_dim = int(y.max().item()) + 1
    models = ["mlp", "gcn", "gcnii"]
    results = {}

    for tag, modality_part, expected_dim in ENCODERS:
        pt = PROCESSED / f"{modality_part}_embeddings.pt"
        if not pt.exists():
            logger.warning(f"[{tag}] missing {pt}, skipping NC")
            continue
        # _load_emb aligns emb rows to node_ids order (zero-fill missing),
        # exactly as the headline harness does.
        x = _load_emb(DOMAIN, modality_part, node_ids).float()
        # L2-normalize (CLIP convention; matches build_features).
        x = x / x.norm(dim=1, keepdim=True).clamp(min=1e-8)
        in_dim = x.size(1)
        logger.info(f"\n{'='*60}\n[{DOMAIN}] encoder={tag} features={tuple(x.shape)} "
                    f"(expected_dim={expected_dim})\n{'='*60}")
        data = Data(x=x, edge_index=ei_pruned, y=y); data.num_nodes = n

        for model_name in models:
            accs, f1s = [], []
            for seed in range(_args.seeds):
                torch.manual_seed(seed)
                m = get_model(model_name, in_dim, _args.hidden, out_dim,
                              num_layers=_args.layers, dropout=_args.dropout)
                r = train_and_eval(m, data, tr, va, te,
                                   num_epochs=_args.epochs, lr=_args.lr, device=DEVICE)
                accs.append(r["test_acc"]); f1s.append(r["test_f1"])
                logger.info(f"  [{tag}/{model_name} seed {seed}] "
                            f"acc={r['test_acc']:.4f} f1={r['test_f1']:.4f}")
            key = f"{model_name}+{tag}"
            results[key] = {
                "acc": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                "f1": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
            }
            logger.info(f"  => {key}: acc={np.mean(accs):.4f}±{np.std(accs):.4f} "
                        f"f1={np.mean(f1s):.4f}±{np.std(f1s):.4f}")

    return lp_pruned, results


# ============================================================
# 4. Main
# ============================================================
def main():
    logger.info(f"DEVICE={DEVICE}  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    ids = load_target_ids()
    id2md, _ = resolve_md_paths(ids)
    if not _args.skip_encode:
        encode_clip_base(ids, id2md)
        encode_siglip(ids, id2md)
    else:
        logger.info("[encode] --skip-encode: skipping encoding step")

    lp_pruned, results = run_leak_free_per_encoder()

    out = {
        "domain": DOMAIN,
        "lp_ceiling_pruned": lp_pruned,
        "seeds": _args.seeds,
        "protocol": "leak_free_pruned",
        "split_seed": _args.split_seed,
        "results": results,
    }
    sp = RESULTS / "encoder_comparison_arxiv.json"
    with open(sp, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"\nSaved -> {sp}")

    # ---- Comparison table ----
    logger.info(f"\n{'='*70}\n[{DOMAIN}] encoder comparison (leak-free, "
                f"{_args.seeds} seeds)  test_acc / macro-F1\n{'='*70}")
    tags = [t for t, _, _ in ENCODERS]
    models = ["mlp", "gcn", "gcnii"]
    header = f"{'model':<8}" + "".join(f"{('acc  / f1'):<22}" + f"{' '*4}" for _ in tags)
    logger.info(f"{'model':<8}" + "".join(f"{t:<26}" for t in tags))
    for mn in models:
        row = f"{mn:<8}"
        for t in tags:
            r = results.get(f"{mn}+{t}")
            row += (f"{r['acc']:.4f}/{r['f1']:.4f}  " if r else f"{'N/A':<26}")
        logger.info(row)
    logger.info("done.")


if __name__ == "__main__":
    main()
