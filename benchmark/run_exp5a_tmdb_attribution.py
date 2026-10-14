#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exp5a — TMDB R@1 genre-keyword 归因实验（2026-07-31, 堵正确性活口）

活口：去泄露后 TMDB text→image R@1 = 0.761（CKA≈0, random≈1/18=0.056）。
假设：高分由 overview 正文里的 genre 关键词驱动（现有去泄露只 strip "Genres:" 行，
不碰正文；42.8% overview 正文仍含 genre 词）。

实验：在编码前用词边界 regex 把 18 个 genre 词从正文里也剥掉 → 重编码 text →
复算 retrieval/CKA → 看 R@1 是否回落至 ≈random(0.056)。
  - 回落 → genre 词在做功（natural correlation，非定义级硬泄露，但需分档叙事）
  - 不回落 → text/image 对齐是真实跨模态语义（去泄露叙事稳）

不改动原 text_embeddings.pt（写到新目录），保留 before/after 干净对比。
"""
# ---- env 必须在 import transformers/torch 前设 ----
import os
import argparse
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", type=int, default=0)
_args, _ = _pre.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_args.gpu)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import sys, json, re
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.run_newdomain_modality import _retrieve, cka          # 复用权威 retrieval/CKA
from benchmark.data.domain_loader import PROCESSED_DIR, DOMAIN_RAW, _abs

DOMAIN = "tmdb"
OUT_DIR = PROJECT_ROOT / "benchmark" / "processed" / "tmdb_genreword_strip"
RESULTS = PROJECT_ROOT / "benchmark" / "results" / "exp5a_tmdb_genre_attribution.json"

# 18 TMDB genres（小写；多词单独处理）+ 常见变体
GENRES = ["science fiction", "sci-fi", "sci fi", "action", "adventure", "animation",
          "comedy", "crime", "documentary", "drama", "family", "fantasy", "history",
          "horror", "music", "mystery", "romance", "thriller", "war", "western"]
GENRES_SORTED = sorted(set(GENRES), key=len, reverse=True)        # 长在前，避免子串抢先
GENRE_WORD_RE = r"\b(?:" + "|".join(re.escape(g) for g in GENRES_SORTED) + r")\b"

LINE_STRIP = r"^Genres?\s*:.*$"          # 现有去泄露（基线也有）
TREATMENT_PATTERNS = [LINE_STRIP, GENRE_WORD_RE]


def load_text_ids():
    tp = PROCESSED_DIR / DOMAIN / "text_embeddings.pt"
    obj = torch.load(tp, map_location="cpu")
    return obj, list(obj["ids"])


def encode_text(ids, paths, patterns, tag):
    from benchmark.encoders.text_encoder import TextEncoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = OUT_DIR if "GENRE" in tag else (PROJECT_ROOT / "benchmark" / "processed" / "tmdb_baseline_reverify")
    out.mkdir(parents=True, exist_ok=True)
    enc = TextEncoder("openai/clip-vit-large-patch14", device=device, strip_patterns=patterns)
    enc.encode_batch(paths, str(out), ids=ids)
    del enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.load(out / "text_embeddings.pt", map_location="cpu")


def align_image(nt_ids):
    ip = PROCESSED_DIR / DOMAIN / "image_embeddings.pt"
    iobj = torch.load(ip, map_location="cpu")
    i2i = {nid: i for i, nid in enumerate(iobj["ids"])}
    dim = iobj["embeddings"].size(1)
    img = torch.zeros(len(nt_ids), dim)
    found = 0
    for i, nid in enumerate(nt_ids):
        if nid in i2i:
            img[i] = iobj["embeddings"][i2i[nid]]; found += 1
    return img.float(), found, dim


def metrics(text_emb, img, device):
    rec, mrr = _retrieve(text_emb, img, device=device)
    return {"R@1": rec["R@1"], "R@5": rec["R@5"], "R@10": rec["R@10"],
            "MRR": mrr["MRR"], "CKA": cka(text_emb, img)}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw_root = _abs(DOMAIN_RAW[DOMAIN])
    text_obj, node_ids = load_text_ids()

    # 建 path（只保留有 .md 的）
    md_paths = {}
    for nid in node_ids:
        md = raw_root / nid / f"{nid}.md"
        if md.is_file():
            md_paths[nid] = str(md)
    ids = [i for i in node_ids if i in md_paths]
    paths = [md_paths[i] for i in ids]
    print(f"[exp5a] {len(ids)} tmdb nodes with .md (of {len(node_ids)} in orig text emb)")

    # 普查：只剥行后，多少正文仍含 genre 词
    line_re = re.compile(LINE_STRIP, re.MULTILINE | re.IGNORECASE)
    word_re = re.compile(GENRE_WORD_RE, re.IGNORECASE)
    contain = 0
    for p in paths:
        t = line_re.sub("", open(p, encoding="utf-8").read())
        if word_re.search(t):
            contain += 1
    census = round(contain / len(ids), 4)
    print(f"[exp5a] 剥 Genres: 行后仍含 genre 词的 overview: {contain}/{len(ids)} = {census}")

    # ---- 基线（只剥行；重编码以控变量，消除编码随机性）----
    print("[exp5a] 编码基线 (line-strip only) ...")
    base_obj = encode_text(ids, paths, [LINE_STRIP], "BASELINE")
    b_ids = list(base_obj["ids"]); b_text = base_obj["embeddings"].float()
    img, nimg, idim = align_image(b_ids)
    base = metrics(b_text, img, device)
    print(f"[exp5a] baseline R@1={base['R@1']:.4f} R@5={base['R@5']:.4f} R@10={base['R@10']:.4f} "
          f"MRR={base['MRR']:.4f} CKA={base['CKA']:.4f} (image align {nimg}/{len(b_ids)})")

    # ---- 处理（剥行 + 剥 genre 词）----
    print("[exp5a] 编码处理 (line + genre-word strip) ...")
    treat_obj = encode_text(ids, paths, TREATMENT_PATTERNS, "GENRE")
    t_ids = list(treat_obj["ids"]); t_text = treat_obj["embeddings"].float()
    img2, _, _ = align_image(t_ids)
    treat = metrics(t_text, img2, device)
    print(f"[exp5a] treatment R@1={treat['R@1']:.4f} R@5={treat['R@5']:.4f} R@10={treat['R@10']:.4f} "
          f"MRR={treat['MRR']:.4f} CKA={treat['CKA']:.4f}")

    d_r1 = treat["R@1"] - base["R@1"]
    floor = 1 / 18
    verdict = ("genre 词在做功（R@1 大跌→正文 genre 关键词驱动，natural correlation，非定义级硬泄露）"
               if d_r1 < -0.1 else
               "对齐稳健（R@1 基本不变→text/image 跨模态语义真实，去泄露叙事稳）")
    print(f"[exp5a] ΔR@1 = {d_r1:+.4f}  (random floor≈{floor:.3f})  → {verdict}")

    out = {
        "experiment": "exp5a_tmdb_genre_attribution",
        "domain": DOMAIN,
        "n_nodes": len(ids),
        "overview_genre_word_rate_after_line_strip": census,
        "patterns": {"baseline": [LINE_STRIP], "treatment": TREATMENT_PATTERNS},
        "baseline_line_strip_only": base,
        "treatment_line_plus_genre_word_strip": treat,
        "delta_R@1": round(d_r1, 4),
        "random_floor_1_over_18": round(floor, 4),
        "verdict": verdict,
        "note": "baseline 重编码以控变量；原 text_embeddings.pt 未改动。"
                " 若 baseline R@1≈0.761（与 docs §9 一致）则复现成功。",
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[exp5a] saved {RESULTS}")


if __name__ == "__main__":
    main()
