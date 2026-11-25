set -eu
R=/data1/shenglin.zeng/dmd2_official_recipe_20261121
cd "$R/code"
export CUDA_VISIBLE_DEVICES=GPU-c374184a-ab5a-681a-ed48-025bed2da39b,GPU-df453630-a9d5-be2e-43db-e0abc1834da2
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$R/env/bin/python" - <<'PY'
import json
from pathlib import Path
from run_guard import check_idle
r=Path('..');check_idle(json.loads((r/'config.json').read_text())['gpu_uuids'])
assert (r/'source_manifest.json').is_file()
assert not (r/'runs/formal_g160').exists()
PY
"$R/env/bin/python" -m torch.distributed.run --standalone --nproc_per_node=2 test_fsdp_precision.py
nohup "$R/env/bin/python" -u "$R/code/run_guard.py" --root "$R" --run formal_g160 > "$R/logs/queue.log" 2>&1 < /dev/null &
echo "VALIDATED_SMOKE_QUEUE_PID=$!"
sleep 2
tail -n 5 "$R/logs/queue.log"

