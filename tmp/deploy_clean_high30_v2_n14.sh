set -eu
. /root/sdxl_research/env.sh
n=A_clean_global_high30_g160_20261114_v2
stage="$SDXL_ROOT/tmp/${n}_deploy"
mkdir -p "$stage"
tar xzf "$SDXL_ROOT/tmp/clean_high30_v2_n14.tgz" -C "$stage"
python "$stage/prepare_clean_high30_v2_n14.py"
c="$SDXL_ROOT/tmp/$n/code";r="$SDXL_ROOT/outputs/$n"
cp "$stage/check_clean_high30_v2_n14.py" "$stage/run_clean_high30_v2_n14.py" "$c/"
cd "$c"
export PYTHONPATH="$c" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TORCHDYNAMO_DISABLE=1
CUDA_VISIBLE_DEVICES='' python check_clean_high30_v2_n14.py
python - <<'PY'
from pathlib import Path
import os,tarfile
r=Path(os.environ['SDXL_ROOT']);n='A_clean_global_high30_g160_20261114_v2'
with tarfile.open(r/'tmp/clean_high30_v2_preparation_n14.tgz','w:gz') as tar:
 for p in (r/'outputs'/n).glob('*.json'):tar.add(p,arcname=p.name)
 for p in (r/'tmp'/n/'code').glob('*.py'):tar.add(p,arcname='source/'+p.name)
 tar.add(r/'tmp'/n/'code/high30.json',arcname='high30.json')
PY
test ! -f "$r/queue.pid"
export CUDA_VISIBLE_DEVICES=0
nohup python -u run_clean_high30_v2_n14.py >"$r/queue.log" 2>&1 </dev/null &
echo $! >"$r/queue.pid"
sleep 2
cat "$r/queue.pid"
tail -8 "$r/queue.log"
kill -0 "$(cat "$r/queue.pid")"
