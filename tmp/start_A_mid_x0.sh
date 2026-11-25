set -eu
. /root/sdxl_research/env.sh
c="$SDXL_ROOT/tmp/A_mid_clean_x0/code";o="$SDXL_ROOT/outputs/A_mid_clean_x0"
mkdir -p "$c" "$o"; cd "$c"
exec 9>"$o/deploy.lock";flock -n 9
if test -s "$o/queue.pid" && kill -0 "$(cat "$o/queue.pid")";then echo ACTIVE;exit 0;fi
b="$SDXL_ROOT/outputs/turning_BC_paired_quota_20261102/uniform_control"
cp "$b/source/"*.py .
cp "$SDXL_ROOT/code/sdxl_research/"{launch,run_quota_bc}.py .
tar xzf "$SDXL_ROOT/tmp/A_mid_clean_x0.tgz"
python - <<'PY'
import os,hashlib,json
from pathlib import Path
p=Path(os.environ['SDXL_ROOT'])/'outputs/turning_BC_paired_quota_20261102/uniform_control'
for n,h in json.loads((p/'source_sha256.json').read_text()).items():
 assert hashlib.sha256(Path(n).read_bytes()).hexdigest()==h,n
PY
python prepare_A_mid_x0.py
export PYTHONPATH="$c:$SDXL_ROOT/tmp/dm1_mid4_uniform_mix1_g160/code:$SDXL_ROOT/tmp/bc_coco_winner_20261102/code:$SDXL_ROOT/code/sdxl_research" CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nohup python -u run_A_mid_x0.py 9>&- >"$o/queue.log" 2>&1 </dev/null &
echo $! >"$o/queue.pid"
for i in $(seq 1 20); do grep -q CPU_TARGET_GRADIENT_OK "$o/queue.log" && break; kill -0 "$(cat "$o/queue.pid")" 2>/dev/null || break;sleep 1;done
tail -8 "$o/queue.log"; kill -0 "$(cat "$o/queue.pid")";grep -q CPU_TARGET_GRADIENT_OK "$o/queue.log";cat "$o/queue.pid"

