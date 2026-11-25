set -eu
. /root/sdxl_research/env.sh
name=A_pcm1_terminal_dm_g160_20261113
stage="$SDXL_ROOT/tmp/${name}_deploy"
mkdir -p "$stage"
tar xzf "$SDXL_ROOT/tmp/A_terminal_only_n13.tgz" -C "$stage"
python "$stage/prepare_A_terminal_only_n13.py"
c="$SDXL_ROOT/tmp/$name/code"
cp "$stage/check_A_terminal_only_n13.py" "$stage/run_A_terminal_only_n13.py" "$c/"
cd "$c"
export PYTHONPATH="$c" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES='' python check_A_terminal_only_n13.py
python - <<'PY'
from pathlib import Path
import os,tarfile
r=Path(os.environ['SDXL_ROOT']);n='A_pcm1_terminal_dm_g160_20261113'
with tarfile.open(r/'tmp/A_terminal_only_preparation_n13.tgz','w:gz') as tar:
 for p in (r/'outputs'/n).glob('*.json'):tar.add(p,arcname=p.name)
 for p in (r/'tmp'/n/'code').glob('*.py'):tar.add(p,arcname='source/'+p.name)
PY
# Queue starts separately after source diff review; never launch during preparation.
echo PREPARATION_AND_CPU_CHECKS_COMPLETE
