"""
TRACK B driver: run run_steering_lp.py for models mlp,gcn,gat,graphsage,gin,
accumulate per-domain results into benchmark/results/steering/lp_<domain>.json
(each model nested under model key). Also writes lp_all_models.json.

Each model run writes steering_lp_<domain>.json which we read then tag with the
model name. We back up the existing steering_lp_<domain>.json first.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "benchmark" / "results" / "steering"
OUT.mkdir(parents=True, exist_ok=True)

DOMAINS = ["steam", "tmdb", "wikiart"]
MODELS = ["mlp", "gcn", "gat", "graphsage", "gin"]
GPU = "4"

# backup existing steering_lp_<domain>.json so we don't lose prior GCN run
for d in DOMAINS:
    p = OUT / f"steering_lp_{d}.json"
    if p.exists():
        shutil.copy(p, OUT / f"steering_lp_{d}.backup.json")

agg = {d: {} for d in DOMAINS}  # domain -> model -> {relation -> metrics}

for model in MODELS:
    print(f"\n##### MODEL {model} #####", flush=True)
    for domain in DOMAINS:
        cmd = [
            sys.executable, str(ROOT / "benchmark" / "run_steering_lp.py"),
            "--domains", domain,
            "--model", model,
            "--modalities", "text+image",
            "--seeds", "3",
            "--epochs", "300",
            "--gpu", GPU,
        ]
        print(f"  >> {model} / {domain}", flush=True)
        rc = subprocess.run(cmd, cwd=str(ROOT))
        if rc.returncode != 0:
            print(f"  !! {model}/{domain} FAILED rc={rc.returncode}", flush=True)
            continue
        f = OUT / f"steering_lp_{domain}.json"
        if not f.exists():
            print(f"  !! no output {f}", flush=True)
            continue
        data = json.loads(f.read_text())
        agg[domain][model] = data
        # also save per-model snapshot
        (OUT / f"steering_lp_{domain}_{model}.json").write_text(json.dumps(data, indent=2))

# write accumulated lp_<domain>.json (as requested by task)
for domain in DOMAINS:
    (OUT / f"lp_{domain}.json").write_text(json.dumps(agg[domain], indent=2))
(OUT / "lp_all_models.json").write_text(json.dumps(agg, indent=2))
print(f"\nDONE -> {OUT}/lp_{{steam,tmdb,wikiart}}.json", flush=True)
