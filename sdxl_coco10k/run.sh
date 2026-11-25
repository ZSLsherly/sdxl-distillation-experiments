#!/usr/bin/env bash
set -eu
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
out=outputs/coco10k_g160_20261020
python - <<'PY'
import json
from pathlib import Path
p=Path('outputs/coco10k_g160_20261020')
assert json.loads((p/'validation.json').read_text())['status']=='passed'
PY
pids=()
for rank in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$rank OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    python -u sdxl_coco10k/evaluate.py worker --rank "$rank" --world 4 --batch 8 --teacher-batch 4 \
    > "$out/worker_$rank.log" 2>&1 &
  pids+=("$!")
done
printf '%s\n' "${pids[@]}" > "$out/worker_pids.txt"
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if [ "$failed" != 0 ]; then echo "GENERATION_FAILED"; exit 1; fi
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  python -u sdxl_coco10k/evaluate.py metrics > "$out/metrics.log" 2>&1
python sdxl_coco10k/report.py
echo "COCO10K_ALL_COMPLETE"
