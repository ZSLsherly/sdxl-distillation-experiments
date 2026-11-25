set -eu
. /root/sdxl_research/env.sh
c="$SDXL_ROOT/tmp/A_pcm10_terminal_patchgan/code";o="$SDXL_ROOT/outputs/A_pcm10_terminal_patchgan"
mkdir -p "$c" "$o"
cd "$c";exec 9>"$o/deploy.lock";flock -n 9
if test -s "$o/queue.pid" && kill -0 "$(cat "$o/queue.pid")";then echo ALREADY_ACTIVE;exit 0;fi
test ! -f "$o/train.log"
cp "$SDXL_ROOT/outputs/A_pcm10_g160/model/source/"*.py .
cp "$SDXL_ROOT/tmp/A_pcm10_g160/code/launch.py" .
tar xzf "$SDXL_ROOT/tmp/a10_patch_gan.tgz"
python - <<'PY'
import hashlib,json,os,shutil
from pathlib import Path
from prepare_a10_patch_gan import prepare
root=Path(os.environ['SDXL_ROOT'])
for name,h in json.loads((root/'outputs/A_pcm10_g160/model/source_sha256.json').read_text()).items():
 assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
for name,h in json.loads(Path('source.json').read_text())['sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h
d=root/'models/dinov2-small'
d.mkdir(exist_ok=True)
for name in ['config.json','model.safetensors','source.json']:shutil.copy2(name,d/name)
prepare(Path('train.py'),Path('train.py'))
PY
test "$(df -B1 --output=avail "$o" | tail -1)" -gt 5350000000
test "$(df -B1 --output=avail /root | tail -1)" -gt 1800000000
aux="/root/a10_patchgan_aux";mkdir -p "$aux/eval" "$aux/metrics" "$o/model"
ln -sfn "$aux/eval" "$o/model/eval";ln -sfn "$aux/metrics" "$o/A_pcm10_patchgan"
export PYTHONPATH="$c:$SDXL_ROOT/tmp/dm1_mid4_uniform_mix1_g160/code:$SDXL_ROOT/tmp/bc_coco_winner_20261102/code:$SDXL_ROOT/code/sdxl_research" CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nohup python -u run_a10_patch_gan.py 9>&- >"$o/queue.log" 2>&1 </dev/null &
echo $! >"$o/queue.pid";sleep 5;cat "$o/queue.log";kill -0 "$(cat "$o/queue.pid")"

