#!/bin/bash
# 链接预测 7模态消融实验（修复后）
cd /home/dev_qw/DataCrawlerAgent

MODALITIES="text image formula text+image text+formula image+formula text+image+formula"
MODELS="mlp gcn gat graphsage"
OUTDIR="./benchmark/results/lp_ablation_fixed"
mkdir -p $OUTDIR

for MOD in $MODALITIES; do
  for MODEL in $MODELS; do
    OUTFILE="${OUTDIR}/lp_${MODEL}_${MOD//+/_}.json"
    if [ -f "$OUTFILE" ]; then
      echo "SKIP (exists): $MODEL + $MOD"
      continue
    fi
    echo "[$(date +%H:%M:%S)] Running: $MODEL + $MOD"
    python -u -c "
from benchmark.data.dataset import AMMAGDataset
from benchmark.tasks.link_prediction import run_link_prediction
ds = AMMAGDataset(root='.', domains=['cs_si','cs_lo','math_lo','physics_socph'],
                  modality='${MOD}', label_type='domain')
run_link_prediction(ds, '${MODEL}', '${MOD}', num_runs=5, num_epochs=500,
                    device='cuda', output_dir='${OUTDIR}', proj_dim=768)
" 2>&1 | grep -E "MRR|保存"
    echo "[$(date +%H:%M:%S)] Done: $MODEL + $MOD"
  done
done

echo ""
echo "===== 生成汇总表 ====="
python -u -c "
import json, os, glob
results = {}
for f in sorted(glob.glob('${OUTDIR}/lp_*.json')):
    with open(f) as fh:
        d = json.load(fh)
    key = d['model'] + '+' + d['modality']
    results[key] = {'mrr': d['mrr_mean'], 'std': d['mrr_std']}

models = ['mlp', 'gcn', 'gat', 'graphsage']
modalities = ['text', 'image', 'formula', 'text+image', 'text+formula', 'image+formula', 'text+image+formula']
print(f\"{'Model':<14}\" + ''.join(f'{m:<18}' for m in modalities))
for m in models:
    row = f'{m:<14}'
    for mod in modalities:
        k = m + '+' + mod
        if k in results:
            row += f\"{results[k]['mrr']:.4f}±{results[k]['std']:.3f}    \"
        else:
            row += f\"{'N/A':<18}\"
    print(row)
"
