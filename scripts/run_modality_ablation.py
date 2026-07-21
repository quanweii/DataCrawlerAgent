#!/usr/bin/env python3
"""Modality ablation: text-only / image-only / text+image NC on all 4 domains.

Usage: python scripts/run_modality_ablation.py --gpu 3
"""
import os, sys, json, argparse, subprocess
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--epochs", type=int, default=300)
ap.add_argument("--models", default="mlp,gcn,gat,graphsage,gin,gcnii")
args = ap.parse_args()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOMAINS = ["steam", "tmdb", "wikiart", "arxiv"]
MODALITIES = ["text", "image", "text+image"]
GPUS = [args.gpu, args.gpu+1, args.gpu+2, args.gpu]  # distribute

runner = PROJECT_ROOT / "benchmark" / "run_newdomain_nc.py"

import threading
results = {}
lock = threading.Lock()

def run_one(domain, modality, gpu):
    cmd = [
        "python", str(runner),
        "--domain", domain, "--gpu", str(gpu),
        "--modalities", modality,
        "--models", args.models,
        "--init-seeds", str(args.seeds),
        "--epochs", str(args.epochs),
    ]
    print(f"[{domain}/{modality}] GPU={gpu} START")
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                          cwd=str(PROJECT_ROOT), env=env)
        with lock:
            results[f"{domain}/{modality}"] = {
                "rc": r.returncode, "stdout": r.stdout[-3000:], "stderr": r.stderr[-2000:]
            }
        print(f"[{domain}/{modality}] DONE rc={r.returncode}")
    except Exception as e:
        with lock:
            results[f"{domain}/{modality}"] = {"rc": -1, "error": str(e)}
        print(f"[{domain}/{modality}] FAIL: {e}")

# Run in parallel: each domain gets its own GPU
threads = []
for i, domain in enumerate(DOMAINS):
    gpu = args.gpu + i
    for modality in MODALITIES:
        t = threading.Thread(target=run_one, args=(domain, modality, gpu))
        t.start()
        threads.append(t)

for t in threads:
    t.join()

# Summary
print("\n" + "="*70)
print("MODALITY ABLATION SUMMARY")
print("="*70)
for domain in DOMAINS:
    for modality in MODALITIES:
        key = f"{domain}/{modality}"
        r = results.get(key, {})
        rc = r.get("rc", -1)
        print(f"  {key}: rc={rc}")
        if rc == 0:
            # Extract table from stdout
            for line in r.get("stdout", "").split("\n"):
                if "mlp" in line and ("acc" in line.lower() or "0." in line[:8]):
                    print(f"    {line.strip()[:120]}")

# Save results
out = PROJECT_ROOT / "benchmark" / "results" / "modality_ablation.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved -> {out}")
