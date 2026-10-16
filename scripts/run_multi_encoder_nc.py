#!/usr/bin/env python3
"""Multi-encoder NC comparison: CLIP-large vs CLIP-base vs SigLIP on steam (smallest domain).

Usage: python scripts/run_multi_encoder_nc.py [--gpu 6]
"""
import os, sys, json, argparse
from pathlib import Path
# Load .env for HF_ENDPOINT
from dotenv import load_dotenv; load_dotenv()
# Remove HF offline flags so we can download through mirror
for k in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]:
    os.environ.pop(k, None)

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--epochs", type=int, default=300)
ap.add_argument("--domain", default="steam")
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from loguru import logger
logger.add(sys.stderr, level="INFO")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DOMAIN = args.domain

# ============================================================
# 1. Load graph
# ============================================================
from benchmark.data.domain_loader import DOMAIN_GRAPH, DOMAIN_RAW, _abs, load_domain_graph
with open(_abs(DOMAIN_GRAPH[DOMAIN]), encoding="utf-8") as f:
    g = json.load(f)
node_ids = [n["id"] for n in g["nodes"]]
raw_root = _abs(DOMAIN_RAW[DOMAIN])

md_paths = {}
image_dirs = {}
for nid in node_ids:
    d = raw_root / nid
    if not d.is_dir(): continue
    md = d / f"{nid}.md"
    if md.is_file(): md_paths[nid] = str(md)
    img = d / "images"
    if img.is_dir(): image_dirs[nid] = str(img)

out_dir = PROJECT_ROOT / "benchmark" / "processed" / DOMAIN
out_dir.mkdir(parents=True, exist_ok=True)
logger.info(f"Domain={DOMAIN}: {len(md_paths)} md, {len(image_dirs)} img dirs")

# ============================================================
# 2. Encoder families
# ============================================================
ENCODERS = {
    "clip_large": {"hf": "openai/clip-vit-large-patch14", "dim": 768},
    "clip_base":  {"hf": "openai/clip-vit-base-patch32",  "dim": 512},
    "siglip":     {"hf": "google/siglip-so400m-patch14-384", "dim": 768},
}

def encode_clip(model_name, tag):
    """Use existing TextEncoder/ImageEncoder for CLIP-family models."""
    from benchmark.encoders.text_encoder import TextEncoder
    from benchmark.encoders.image_encoder import ImageEncoder

    logger.info(f"[{tag}] Encoding text with {model_name}...")
    te = TextEncoder(model_name=model_name, device=DEVICE)
    ids_list = list(md_paths.keys())
    te.encode_batch([md_paths[i] for i in ids_list], str(out_dir), ids=ids_list)
    # Move output to tagged name
    default = out_dir / "text_embeddings.pt"
    target = out_dir / f"text_{tag}_embeddings.pt"
    if default.exists():
        import shutil; shutil.move(str(default), str(target))
    del te; torch.cuda.empty_cache()

    logger.info(f"[{tag}] Encoding images with {model_name}...")
    ie = ImageEncoder(model_name=model_name, device=DEVICE)
    ie.encode_batch(image_dirs, str(out_dir))
    default2 = out_dir / "image_embeddings.pt"
    target2 = out_dir / f"image_{tag}_embeddings.pt"
    if default2.exists():
        import shutil; shutil.move(str(default2), str(target2))
    del ie; torch.cuda.empty_cache()
    logger.info(f"[{tag}] Done → {target}, {target2}")

def encode_siglip(tag="siglip"):
    """Encode text+images with SigLIP (not CLIP family — different model class)."""
    model_name = "google/siglip-so400m-patch14-384"
    from transformers import SiglipModel, SiglipProcessor, SiglipTokenizer
    logger.info(f"[siglip] Loading {model_name}...")
    model = SiglipModel.from_pretrained(model_name).to(DEVICE)
    processor = SiglipProcessor.from_pretrained(model_name)
    tokenizer = SiglipTokenizer.from_pretrained(model_name)
    model.eval()

    dim = model.config.vision_config.hidden_size
    MAX_TOKENS = 64  # SigLIP max

    @torch.no_grad()
    def encode_texts(paths, ids, batch_size=32):
        all_embs = []
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i+batch_size]
            texts = []
            for p in batch_paths:
                try:
                    with open(p, encoding="utf-8") as f:
                        texts.append(f.read()[:2000])
                except Exception:
                    texts.append("")
            inputs = tokenizer(texts, padding=True, truncation=True,
                               max_length=MAX_TOKENS, return_tensors="pt").to(DEVICE)
            out = model.get_text_features(**inputs)
            out = torch.nn.functional.normalize(out, p=2, dim=1)
            all_embs.append(out.cpu())
        emb = torch.cat(all_embs, dim=0).float()
        torch.save({"ids": ids, "embeddings": emb}, out_dir / f"text_{tag}_embeddings.pt")
        logger.info(f"[siglip] Text: {len(ids)} docs → {tuple(emb.shape)}")

    @torch.no_grad()
    def encode_images(img_dirs, batch_size=16):
        from PIL import Image
        all_ids, all_embs = [], []
        dirs_list = list(img_dirs.items())
        for i in range(0, len(dirs_list), batch_size):
            batch = dirs_list[i:i+batch_size]
            images, batch_ids = [], []
            for nid, d in batch:
                imgs = sorted(Path(d).glob("*.jpg")) + sorted(Path(d).glob("*.png"))
                if imgs:
                    try:
                        images.append(Image.open(imgs[0]).convert("RGB"))
                        batch_ids.append(nid)
                    except Exception:
                        continue
            if not images: continue
            inputs = processor(images=images, return_tensors="pt").to(DEVICE)
            out = model.get_image_features(**inputs)
            out = torch.nn.functional.normalize(out, p=2, dim=1)
            all_embs.append(out.cpu())
            all_ids.extend(batch_ids)
        emb = torch.cat(all_embs, dim=0).float()
        torch.save({"ids": all_ids, "embeddings": emb}, out_dir / f"image_{tag}_embeddings.pt")
        logger.info(f"[siglip] Image: {len(all_ids)} imgs → {tuple(emb.shape)}")

    ids_text = list(md_paths.keys())
    encode_texts([md_paths[i] for i in ids_text], ids_text)
    encode_images(image_dirs)
    del model; torch.cuda.empty_cache()

# ============================================================
# 3. Run encoding (skip if already done)
# ============================================================
for tag, cfg in ENCODERS.items():
    text_file = out_dir / f"text_{tag}_embeddings.pt"
    image_file = out_dir / f"image_{tag}_embeddings.pt"
    if text_file.exists() and image_file.exists():
        logger.info(f"[{tag}] Already encoded, skipping")
        continue
    if tag == "siglip":
        encode_siglip(tag)
    else:
        encode_clip(cfg["hf"], tag)

# ============================================================
# 4. Run leak-free NC per encoder
# ============================================================
from benchmark.data.splits import stratified_split, prune_cross_split_edges, label_propagation_ceiling
from benchmark.models.gnn import get_model
from benchmark.tasks.node_classification import train_and_eval
from torch_geometric.data import Data

graph = load_domain_graph(DOMAIN)
y = graph["y"]; ei_full = graph["edge_index"]
masks = stratified_split(y, (0.6, 0.2, 0.2), seed=1)
tr, va, te = masks
ei_pruned = prune_cross_split_edges(ei_full, masks)
lp_pruned = label_propagation_ceiling(y, ei_pruned, tr, te)

results = {}
for tag, cfg in ENCODERS.items():
    logger.info(f"\n{'='*50}\n[{DOMAIN}] NC with {tag} (dim={cfg['dim']})\n{'='*50}")

    # Build features
    text_pt = out_dir / f"text_{tag}_embeddings.pt"
    image_pt = out_dir / f"image_{tag}_embeddings.pt"
    if not text_pt.exists() or not image_pt.exists():
        logger.warning(f"[{tag}] Missing embeddings, skipping")
        continue

    t_ids = torch.load(text_pt, map_location="cpu")["ids"]
    t_emb = torch.load(text_pt, map_location="cpu")["embeddings"].float()
    i_ids = torch.load(image_pt, map_location="cpu")["ids"]
    i_emb = torch.load(image_pt, map_location="cpu")["embeddings"].float()

    t_id2row = {sid: i for i, sid in enumerate(t_ids)}
    i_id2row = {sid: i for i, sid in enumerate(i_ids)}

    x_list = []
    for nid in graph["node_ids"]:
        ti = t_id2row.get(nid); ii = i_id2row.get(nid)
        t = t_emb[ti] if ti is not None else torch.zeros(t_emb.size(1))
        i = i_emb[ii] if ii is not None else torch.zeros(i_emb.size(1))
        x_list.append(torch.cat([t, i]))
    x = torch.stack(x_list)
    x = x / x.norm(dim=1, keepdim=True).clamp(min=1e-8)

    data = Data(x=x, edge_index=ei_pruned, y=y); data.num_nodes = len(graph["node_ids"])
    in_dim, out_dim = x.size(1), int(y.max().item()) + 1

    logger.info(f"  Features: {tuple(x.shape)}, LP-ceil(pruned)={lp_pruned['acc']:.3f}")

    for model_name in ["mlp", "gcn", "gat"]:
        accs, f1s = [], []
        for s in range(args.seeds):
            torch.manual_seed(s)
            m = get_model(model_name, in_dim, 256, out_dim)
            r = train_and_eval(m, data, tr, va, te, num_epochs=args.epochs, lr=0.01, device=DEVICE)
            accs.append(r["test_acc"]); f1s.append(r["test_f1"])
        key = f"{model_name}+{tag}"
        results[key] = {"acc": np.mean(accs), "acc_std": np.std(accs),
                        "f1": np.mean(f1s), "f1_std": np.std(f1s)}
        logger.info(f"  {model_name}: acc={np.mean(accs):.3f}±{np.std(accs):.3f} f1={np.mean(f1s):.3f}")

# ============================================================
# 5. Comparison table
# ============================================================
logger.info(f"\n{'='*70}")
logger.info(f"[{DOMAIN}] Multi-encoder NC comparison (leak-free, {args.seeds} seeds)")
logger.info(f"{'='*70}")
tags = [t for t in ["clip_large","clip_base","siglip"] if f"mlp+{t}" in results]
header = f"{'model':<10}" + "".join(f"{t+' (acc/f1)':<28}" for t in tags)
logger.info(header)
for model_name in ["mlp","gcn","gat"]:
    row = f"{model_name:<10}"
    for tag in tags:
        r = results.get(f"{model_name}+{tag}")
        if r: row += f"{r['acc']:.3f}/{r['f1']:.3f}±{r['f1_std']:.3f}          "
        else: row += f"{'N/A':<28}"
    logger.info(row)

# Save
sp = PROJECT_ROOT / "benchmark" / "results" / f"encoder_comparison_{DOMAIN}.json"
with open(sp, "w") as f:
    json.dump({"domain": DOMAIN, "lp_ceiling_pruned": lp_pruned,
               "seeds": args.seeds, "results": results}, f, indent=2)
logger.info(f"\nSaved → {sp}")
logger.info("✓ Multi-encoder comparison complete")
