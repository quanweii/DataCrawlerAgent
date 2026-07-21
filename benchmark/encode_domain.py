"""
按域离线编码新域（steam/tmdb/wikiart）的文本与图像 → CLIP 768d 嵌入。

复用 benchmark.encoders.{TextEncoder, ImageEncoder}（均为 CLIP-ViT-Large-Patch14，
与 arxiv 规范一致）。注意：benchmark/encode_offline.py 是 arxiv 专用且误用了
clip-vit-base-patch32，本脚本才是新域的规范编码入口。

2026-07-15: 新增 per-domain text decontamination — 编码前自动 strip label-bearing
元数据行 (Style:/Genres:)，消除特征级 label 泄露。

输出: benchmark/processed/{domain}/{text,image}_embeddings.pt
      格式: {"ids": [str,...], "embeddings": FloatTensor[N,768]}
      （只编码 v3 图中存在的节点 id，保证与 loader 对齐）

用法:
  python benchmark/encode_domain.py --domain steam --gpu 0 --modalities text,image
  python benchmark/encode_domain.py --domain steam --modalities text            # CPU
  python benchmark/encode_domain.py --domain tmdb --gpu 1 --limit 100           # 试跑前100
"""
# ---- CLI 必须在 import torch/transformers 之前解析（控制 CUDA 可见性 + HF 离线）----
import os
import argparse

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--domain", required=True, choices=["steam", "tmdb", "wikiart", "spotify"])
_pre.add_argument("--gpu", type=int, default=None)
_pre.add_argument("--modalities", default="text,image")
_pre.add_argument("--limit", type=int, default=-1, help="只编码前 N 个节点（试跑）")
_args, _ = _pre.parse_known_args()

if _args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_args.gpu)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import sys
import json
from pathlib import Path

import torch
from loguru import logger

logger.add(sys.stderr, level="INFO")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.data.domain_loader import DOMAIN_GRAPH, DOMAIN_RAW, _abs

# Per-domain label-bearing field stripping patterns (2026-07-15 decontamination).
# These patterns match ENTIRE LINES (re.MULTILINE) whose presence in the text
# gives away the label.  See docs/benchmark_text_decontamination_2026-07-15.md.
TEXT_STRIP_PATTERNS = {
    # wikiart .md is a fixed template: #Title / Artist / Style / Year / body-sentence / ##About.
    # The body sentence always contains "was created" and names the style/artist.
    # After stripping these, only the artwork title remains (~30-60 chars).
    "wikiart": [
        r"^Artist\s*:.*$",                              # artist → style shortcut (76% map 1:1)
        r"^Style\s*:.*$",                                # label in metadata field
        r"^Year\s*:.*$",                                 # year → period → style shortcut
        r"^.*\bwas\s+created\b.*$",                       # body template sentence (variations: c.YEAR, no year, "by Artist", etc.)
        r"^##\s+About\s+the\s+Artist[\s\S]*$",           # biography; may mention styles/movements
    ],
    "steam":   [r"^Genres?\s*:.*$"],
    "tmdb":    [r"^Genres?\s*:.*$"],
    # spotify is text-only but the label (playlist name) is NOT written into the
    # track markdown in a structured metadata field — safe to encode as-is.
    # arxiv paper abstracts do not contain the cs_si / math_lo category label.
}


def main():
    domain = _args.domain
    modalities = [m.strip() for m in _args.modalities.split(",") if m.strip()]
    limit = _args.limit

    # 只编码 v3 图中存在的节点 id（与 loader 对齐）
    with open(_abs(DOMAIN_GRAPH[domain]), encoding="utf-8") as f:
        g = json.load(f)
    node_ids = [n["id"] for n in g["nodes"]]
    if limit > 0:
        node_ids = node_ids[:limit]
    logger.info(f"[{domain}] 将编码 {len(node_ids)} 个节点, modalities={modalities}")

    raw_root = _abs(DOMAIN_RAW[domain])

    # 预检：原始目录 + .md + images/
    md_paths: dict = {}
    image_dirs: dict = {}
    missing_dir = missing_md = missing_img = 0
    for nid in node_ids:
        d = raw_root / nid
        if not d.is_dir():
            missing_dir += 1
            continue
        md = d / f"{nid}.md"
        if md.is_file():
            md_paths[nid] = str(md)
        else:
            missing_md += 1
        img = d / "images"
        if img.is_dir():
            image_dirs[nid] = str(img)
        else:
            missing_img += 1
    if missing_dir:
        logger.warning(f"[{domain}] 缺原始目录: {missing_dir}")
    if missing_md:
        logger.warning(f"[{domain}] 缺 .md: {missing_md}")
    if missing_img:
        logger.warning(f"[{domain}] 缺 images/: {missing_img}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device = {device}")
    out_dir = PROJECT_ROOT / "benchmark" / "processed" / domain
    out_dir.mkdir(parents=True, exist_ok=True)

    if "text" in modalities:
        from benchmark.encoders.text_encoder import TextEncoder
        ids = list(md_paths.keys())
        paths = [md_paths[i] for i in ids]
        patterns = TEXT_STRIP_PATTERNS.get(domain)
        if patterns:
            logger.info(f"[{domain}] ⚠ 编码前 strip label-bearing 字段: {patterns}")
        logger.info(f"[{domain}] 编码文本 {len(ids)} 篇 ...")
        enc = TextEncoder(
            model_name="openai/clip-vit-large-patch14", device=device,
            strip_patterns=patterns,
        )
        enc.encode_batch(paths, str(out_dir), ids=ids)
        del enc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if "image" in modalities:
        from benchmark.encoders.image_encoder import ImageEncoder
        logger.info(f"[{domain}] 编码图像 {len(image_dirs)} 个目录 ...")
        enc = ImageEncoder(model_name="openai/clip-vit-large-patch14", device=device)
        enc.encode_batch(image_dirs, str(out_dir))
        del enc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(f"✅ [{domain}] 编码完成 → {out_dir}")


if __name__ == "__main__":
    main()
