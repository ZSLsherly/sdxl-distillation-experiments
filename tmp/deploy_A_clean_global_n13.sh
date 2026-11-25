set -eu
. /root/sdxl_research/env.sh
name=A_clean_global_g160_20261113
stage="$SDXL_ROOT/tmp/${name}_deploy"
mkdir -p "$stage"
tar xzf "$SDXL_ROOT/tmp/A_clean_global_n13.tgz" -C "$stage"
python "$stage/prepare_A_clean_global_n13.py"
c="$SDXL_ROOT/tmp/$name/code"
cp "$stage/check_A_clean_global_n13.py" "$stage/run_A_clean_global_n13.py" "$c/"
cd "$c"
export PYTHONPATH="$c" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES='' python check_A_clean_global_n13.py
python - <<'PY'
from pathlib import Path
import os,tarfile
r=Path(os.environ['SDXL_ROOT']);n='A_clean_global_g160_20261113'
with tarfile.open(r/'tmp/A_clean_global_preparation_n13.tgz','w:gz') as tar:
 for p in (r/'outputs'/n).glob('*.json'):tar.add(p,arcname=p.name)
 for p in (r/'tmp'/n/'code').glob('*.py'):tar.add(p,arcname='source/'+p.name)
PY
export CUDA_VISIBLE_DEVICES=0
nohup python -u run_A_clean_global_n13.py >"$SDXL_ROOT/outputs/$name/queue.log" 2>&1 </dev/null &
echo $! >"$SDXL_ROOT/outputs/$name/queue.pid"
sleep 2
cat "$SDXL_ROOT/outputs/$name/queue.pid"
tail -5 "$SDXL_ROOT/outputs/$name/queue.log"
kill -0 "$(cat "$SDXL_ROOT/outputs/$name/queue.pid")"
