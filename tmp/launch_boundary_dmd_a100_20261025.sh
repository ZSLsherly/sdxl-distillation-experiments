set -euo pipefail
source /data5/shenglin.zeng/env.sh
root=/data5/shenglin.zeng/sdxl_crossdistill
out=$root/outputs/boundary_dmd_10k_a100_20261025_v1
mkdir "$out"
mkdir "$out/code"
tar -xzf "$root/boundary_dmd_a100_20261025.tar.gz" -C "$out/code"
cd "$out/code/sdxl_research"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 python -m unittest test_boundary -v > "$out/cpu_tests.log" 2>&1
python - <<'PY'
import hashlib,json
from pathlib import Path
m=json.loads(Path('source_manifest.json').read_text())
bad=[p for p,h in m.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
assert not bad,bad
print('source_verified',len(m))
PY
cd "$root"
nohup nice -n 10 python -u "$out/code/run_workflow.py" > "$out/workflow.log" 2>&1 < /dev/null &
echo $! > "$out/workflow.pid"
printf 'workflow_pid '; cat "$out/workflow.pid"
tail -5 "$out/cpu_tests.log"
