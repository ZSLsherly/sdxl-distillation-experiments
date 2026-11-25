set -eu
. /root/sdxl_research/env.sh
code="$SDXL_ROOT/tmp/fake_tracking_pilot_20261102/code"
cd "$code"
tar -xzf "$SDXL_ROOT/tmp/late_mix_review.tar.gz"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
export PYTHONPATH="$code"
python - <<'PY'
import sys,json
from pathlib import Path
from launch import flags
from train import arguments
p=Path('configs');base=json.loads((p/'turning_C_quota_a800_g160.json').read_text());late=json.loads((p/'tracking_C_late_mix_1pct_proposed_g160.json').read_text());assert late==dict(base,fake_track_pilot=True,fake_student_mix=.01,fake_mix_start_g=121)
sys.argv=['train.py','--run','parse_only',*flags(late)];a=arguments();assert (a.max_student_updates,a.fake_mix_start_g,a.fake_student_mix,a.benchmark,a.checkpoint_mode)==(160,121,.01,False,'final-weights');print('config OK: G1..120 zero, G121..160 1%, final-only storage')
PY
python -m unittest test_fake_tracking
