set -eu
R=/data1/shenglin.zeng/dmd2_official_recipe_4gpu_20261123
cd "$R/code"
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
"$R/env/bin/python" -u test_recipe.py
"$R/env/bin/python" -u test_cpu.py
export CUDA_VISIBLE_DEVICES=GPU-dee80cca-d29a-9086-d006-b76ba6ee093e,GPU-7328ece9-841e-484d-9e72-d5b762fdce95,GPU-c374184a-ab5a-681a-ed48-025bed2da39b,GPU-4fba4d7f-66e5-aa77-7210-3803fe7190df
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
"$R/env/bin/python" - <<'PY'
import json,hashlib
from pathlib import Path
from run_guard import check_idle
r=Path('..');c=json.loads((r/'config.json').read_text());check_idle(c['gpu_uuids'])
assert len(c['gpu_uuids'])==4 and c['batch']==64 and c['student_updates']==160
checks={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'code').rglob('*.py')}
checks['config.json']=hashlib.sha256((r/'config.json').read_bytes()).hexdigest()
(r/'source_manifest.json').write_text(json.dumps(checks,indent=2))
PY
"$R/env/bin/python" -m torch.distributed.run --standalone --nproc_per_node=4 test_fsdp_precision.py
nohup "$R/env/bin/python" -u "$R/code/run_guard.py" --root "$R" --run formal_g160 > "$R/logs/queue.log" 2>&1 < /dev/null &
echo "QUAD_QUEUE_PID=$!"
sleep 2
tail -n 4 "$R/logs/queue.log"
