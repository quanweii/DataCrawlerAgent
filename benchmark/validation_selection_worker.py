import sys,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'benchmark/results/validation_selected_nc_20261014'
gpu=int(sys.argv[1]);jobs=json.loads((out/'jobs.json').read_text())
for i,(domain,modality) in enumerate(jobs):
    lock=out/f'job_{i:02d}.lock'
    try:
        with lock.open('x') as f: f.write(str(gpu))
    except FileExistsError: continue
    print('START',i,domain,modality,time.strftime('%H:%M:%S'),flush=True)
    with (out/f'job_{i:02d}.log').open('w') as log:
        p=subprocess.run([sys.executable,'-u',str(ROOT/'benchmark/run_validation_selected_nc.py'),
                          '--domain',domain,'--modality',modality,'--gpu',str(gpu)],
                         cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    (out/f'job_{i:02d}.status.json').write_text(json.dumps(dict(job=i,domain=domain,modality=modality,gpu=gpu,returncode=p.returncode)))
    print('DONE',i,'rc',p.returncode,time.strftime('%H:%M:%S'),flush=True)
    if p.returncode: raise SystemExit(p.returncode)
