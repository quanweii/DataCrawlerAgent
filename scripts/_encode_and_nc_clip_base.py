#!/usr/bin/env python3
"""
Multi-encoder validation: encode steam with CLIP-base, run NC, compare vs CLIP-large.

Usage:
  python scripts/_encode_and_nc_clip_base.py
"""
import os, sys, json
sys.path.insert(0, '.')
from pathlib import Path
import torch
import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DOMAIN = "steam"
MODEL = "openai/clip-vit-base-patch32"
SUFFIX = "clip_base"

logger.add(sys.stderr, level="INFO")
logger.info(f"Multi-encoder test: {DOMAIN} with {MODEL}")

# 1. Load node list from agent graph
from benchmark.data.domain_loader import DOMAIN_GRAPH, DOMAIN_RAW, _abs
with open(_abs(DOMAIN_GRAPH[DOMAIN]), encoding="utf-8") as f:
    g = json.load(f)
node_ids = [n["id"] for n in g["nodes"]]
raw_root = _abs(DOMAIN_RAW[DOMAIN])

# 2. Paths
md_paths, image_dirs = {}, {}
for nid in node_ids:
    d = raw_root / nid
    if not d.is_dir(): continue
    md = d / f"{nid}.md"
    if md.is_file(): md_paths[nid] = str(md)
    img = d / "images"
    if img.is_dir(): image_dirs[nid] = str(img)

out_dir = PROJECT_ROOT / "benchmark" / "processed" / DOMAIN
out_dir.mkdir(parents=True, exist_ok=True)
text_file = out_dir / f"text_{SUFFIX}_embeddings.pt"
image_file = out_dir / f"image_{SUFFIX}_embeddings.pt"

# 3. Encode text
logger.info(f"Encoding text: {len(md_paths)} docs with {MODEL}")
from benchmark.encoders.text_encoder import TextEncoder
te = TextEncoder(model_name=MODEL, device=DEVICE)
ids = list(md_paths.keys())
paths = [md_paths[i] for i in ids]
embs_text = te.encode_batch(paths, str(out_dir), ids=ids)  # writes to file
del te; torch.cuda.empty_cache()

# 4. Encode images
logger.info(f"Encoding images: {len(image_dirs)} dirs with {MODEL}")
from benchmark.encoders.image_encoder import ImageEncoder
ie = ImageEncoder(model_name=MODEL, device=DEVICE)
ie.encode_batch(image_dirs, str(out_dir))  # writes to file
# Need to rename output to avoid overwriting CLIP-large embeddings
# encode_batch writes {modality}_embeddings.pt — rename to {modality}_{SUFFIX}_embeddings.pt
import shutil
default_text = out_dir / "text_embeddings.pt"
default_image = out_dir / "image_embeddings.pt"
if default_text.exists() and not text_file.exists():
    shutil.move(str(default_text), str(text_file))
if default_image.exists() and not image_file.exists():
    shutil.move(str(default_image), str(image_file))

del ie; torch.cuda.empty_cache()
logger.info(f"Encoded: {text_file}, {image_file}")

# 5. Run NC with CLIP-base embeddings
logger.info("Running NC comparison...")
from benchmark.data.domain_loader import load_domain_graph, build_features
from benchmark.models.gnn import get_model
from benchmark.tasks.node_classification import train_and_eval
from benchmark.data.splits import stratified_split, prune_cross_split_edges, label_propagation_ceiling
from torch_geometric.data import Data

# Hack: override _load_emb to use clip_base embeddings
import benchmark.data.domain_loader as dl
orig_load_emb = dl._load_emb

def _load_emb_clip_base(domain, modality_part, node_ids_list):
    path = dl.PROCESSED_DIR / domain / f"{modality_part}_{SUFFIX}_embeddings.pt"
    if not path.exists():
        # fallback
        print(f"  {path} not found, falling back to default")
        path = dl.PROCESSED_DIR / domain / f"{modality_part}_embeddings.pt"
    obj = torch.load(path, map_location="cpu")
    ids_emb = obj["ids"]
    embs = obj["embeddings"].float()
    id2row = {sid: i for i, sid in enumerate(ids_emb)}
    dim = embs.size(1)
    out = torch.zeros((len(node_ids_list), dim), dtype=torch.float32)
    found = 0
    for i, nid in enumerate(node_ids_list):
        if nid in id2row:
            out[i] = embs[id2row[nid]]
            found += 1
    logger.info(f"[{domain}] {modality_part}({SUFFIX}) aligned {found}/{len(node_ids_list)}, dim={dim}")
    return out

dl._load_emb = _load_emb_clip_base

graph = load_domain_graph(DOMAIN)
x = build_features(DOMAIN, graph, "text+image")
y = graph["y"]
ei_full = graph["edge_index"]

masks = stratified_split(y, (0.6, 0.2, 0.2), seed=1)
tr, va, te = masks
ei_pruned = prune_cross_split_edges(ei_full, masks)
lp_info = label_propagation_ceiling(y, ei_pruned, tr, te)

data = Data(x=x, edge_index=ei_pruned, y=y); data.num_nodes = len(graph["node_ids"])
in_dim, out_dim = x.size(1), int(y.max().item()) + 1

logger.info(f"CLIP-base dim={in_dim}, classes={out_dim}")
logger.info(f"LP-ceiling(pruned)={lp_info['acc']:.3f}, majority={lp_info['majority_acc']:.3f}")

results = {}
for model_name in ["mlp", "gcn"]:
    accs, f1s = [], []
    for seed in range(3):
        torch.manual_seed(seed)
        m = get_model(model_name, in_dim, 256, out_dim)
        r = train_and_eval(m, data, tr, va, te, num_epochs=300, lr=0.01, device=DEVICE)
        accs.append(r["test_acc"]); f1s.append(r["test_f1"])
    results[model_name] = {"acc": np.mean(accs), "acc_std": np.std(accs),
                           "f1": np.mean(f1s), "f1_std": np.std(f1s)}
    logger.info(f"  {model_name}(clip-base): acc={np.mean(accs):.3f}±{np.std(accs):.3f} f1={np.mean(f1s):.3f}")

# Restore original
dl._load_emb = orig_load_emb

# 6. Compare with CLIP-large (from saved summary)
import json
nc_summary = PROJECT_ROOT / "benchmark" / "results" / "newdomain_nc" / f"nc_{DOMAIN}_summary.json"
if nc_summary.exists():
    nc = json.load(open(nc_summary))
    logger.info("\n=== Encoder comparison ===")
    logger.info(f"  model     CLIP-large(acc/f1)     CLIP-base(acc/f1)")
    for model_name in ["mlp", "gcn"]:
        r_large = nc.get("table", {}).get(f"{model_name}+text+image", {})
        r_base = results.get(model_name, {})
        logger.info(f"  {model_name:<9} {r_large.get('acc', '?'):.3f}/{r_large.get('f1', '?'):.3f}              "
                    f"{r_base.get('acc', '?'):.3f}/{r_base.get('f1', '?'):.3f}")

logger.info("\n✓ Multi-encoder comparison complete")
