"""
Offline encoding script - sets HF offline env vars before importing anything
Usage: python benchmark/encode_offline.py --modality image --gpu 2
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import sys
import json
import torch
from loguru import logger
from pathlib import Path

logger.add(sys.stderr, level="INFO")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def encode_image(gpu_id):
    from benchmark.encoders.image_encoder import ImageEncoder
    from benchmark.data.preprocessor import AMMAGPreprocessor

    data_root = str(PROJECT_ROOT / "data")
    output_dir = str(PROJECT_ROOT / "benchmark" / "processed")

    processor = AMMAGPreprocessor(data_root=data_root)
    device = f"cuda:{gpu_id}"

    all_image_dirs = {}
    paper_id_map = {}
    for domain in ["cs_si", "cs_lo", "math_lo", "physics_socph"]:
        papers = processor.scan_papers(domain)
        for p in papers:
            if p["arxiv_id"] not in paper_id_map:
                paper_id_map[p["arxiv_id"]] = len(paper_id_map)
                if p["has_images"]:
                    all_image_dirs[p["arxiv_id"]] = p["images_dir"]

    logger.info(f"Image encoding: {len(paper_id_map)} papers, {len(all_image_dirs)} with images")

    encoder = ImageEncoder(model_name="openai/clip-vit-base-patch32", device=device)
    result = encoder.encode_batch(all_image_dirs, output_dir)
    logger.info(f"Image encoding done: {result}")


def encode_formula(gpu_id):
    from benchmark.encoders.formula_encoder import FormulaEncoder
    from benchmark.data.preprocessor import AMMAGPreprocessor

    data_root = str(PROJECT_ROOT / "data")
    output_dir = str(PROJECT_ROOT / "benchmark" / "processed")

    processor = AMMAGPreprocessor(data_root=data_root)
    device = f"cuda:{gpu_id}"

    all_md_paths = []
    all_ids = []
    seen = set()
    for domain in ["cs_si", "cs_lo", "math_lo", "physics_socph"]:
        papers = processor.scan_papers(domain)
        for p in papers:
            if p["arxiv_id"] not in seen:
                seen.add(p["arxiv_id"])
                all_md_paths.append(p["md_path"])
                all_ids.append(p["arxiv_id"])

    logger.info(f"Formula encoding: {len(all_ids)} papers")

    encoder = FormulaEncoder(model_name="all-MiniLM-L6-v2", device=device)
    result = encoder.encode_batch(all_md_paths, all_ids, output_dir)
    logger.info(f"Formula encoding done: {result}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["image", "formula"], required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if args.modality == "image":
        encode_image(args.gpu)
    else:
        encode_formula(args.gpu)
