#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
鲁棒版公开 benchmark 泄露画像 —— 修复 run_public_leakage_profile.py 的两个设计缺陷:
  缺陷1: dump-at-end  → 某个数据集挂起会丢掉前面所有结果
  缺陷2: 无单数据集超时 → 被墙镜像会无限等
修法: 每个数据集独立子进程 + 90s 超时 + 增量 dump(每个跑完立刻写盘).
测量口径不变(复用 profile_data / benchmark/data/splits.py).
"""
import sys, json, os, tempfile, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from benchmark.run_public_leakage_profile import PUBLIC, OUT, SPLIT_SEED

PER_DS_TIMEOUT = 90  # 单数据集下载+profile 上限

WORKER = r'''
import sys, json, importlib
name, kind, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
mod = importlib.import_module("benchmark.run_public_leakage_profile")
loader = None
for n, k, lf in mod.PUBLIC:
    if n == name:
        loader = lf; break
assert loader is not None, name
data = loader()
r = mod.profile_data(name, data, kind)
json.dump(r, open(out_path, "w"))
'''

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, failed = [], []
    t0 = time.time()
    for name, kind, _ in PUBLIC:
        fd, outp = tempfile.mkstemp(suffix=".json"); os.close(fd)
        try:
            subprocess.run([sys.executable, "-c", WORKER, name, kind, outp],
                           timeout=PER_DS_TIMEOUT, check=True, capture_output=True)
            rows.append(json.load(open(outp)))
            print(f"  OK  {name:<18} ({kind})", flush=True)
        except subprocess.TimeoutExpired:
            failed.append({"name": name, "kind": kind, "error": f"timeout(>{PER_DS_TIMEOUT}s, likely blocked/slow mirror)"})
            print(f"  TMO {name:<18} ({kind})  -> 镜像不可达/慢,跳过", flush=True)
        except subprocess.CalledProcessError as e:
            msg = (e.stderr.decode()[-200:] if e.stderr else str(e))[:160]
            failed.append({"name": name, "kind": kind, "error": msg})
            print(f"  ERR {name:<18} ({kind}): {msg[:80]}", flush=True)
        finally:
            try: os.unlink(outp)
            except OSError: pass
        # 增量 dump —— 任何时候挂掉都保住已有结果
        json.dump({"rows": rows, "failed": failed, "split_seed": SPLIT_SEED,
                   "measure": "benchmark/data/splits.py (label_propagation_ceiling, "
                              "stratified 60/20/20 seed=1, cross-split edge pruning)",
                   "per_dataset_timeout_s": PER_DS_TIMEOUT},
                  open(OUT / "leakage_profile_public.json", "w"), indent=2, ensure_ascii=False)
    print(f"\n[done in {time.time()-t0:.0f}s] rows={len(rows)} failed={len(failed)}")

if __name__ == "__main__":
    main()
