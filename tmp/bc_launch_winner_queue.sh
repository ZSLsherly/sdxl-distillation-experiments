set -eu
. /root/sdxl_research/env.sh
code="$SDXL_ROOT/tmp/bc_coco_winner_20261102/code"
mkdir -p "$code"
tar -xzf "$SDXL_ROOT/tmp/bc_coco_winner_code.tar.gz" -C "$code"
cd "$code"
export PYTHONPATH="$code"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest test_fake_tracking test_paired_stages test_interval_quota
CUDA_VISIBLE_DEVICES='' python - <<'PY'
import sys,json
from pathlib import Path
from launch import flags
from train import arguments
import evaluate
for arm in ('B','C'):
 cfg=json.loads(Path(f'configs/turning_{arm}_quota_a800_g160.json').read_text());cfg.update(fake_track_pilot=True,fake_student_mix=.01,fake_mix_start_g=1)
 sys.argv=['train','--run','validate',*flags(cfg)];a=arguments();assert a.fake_mix_start_g==1 and a.max_student_updates==160 and not a.benchmark and a.pair_fake_stages and a.match_fake_prior
 print('VALIDATED',arm,a.turning_profile)
PY
nohup python -u run_bc_coco_then_winner.py > "$SDXL_WORK/logs/bc_coco_winner_queue.log" 2>&1 < /dev/null &
echo QUEUE_PID=$!
