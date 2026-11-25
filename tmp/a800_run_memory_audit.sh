set -eu
. /root/sdxl_research/env.sh
mkdir -p "$SDXL_ROOT/code/sdxl_memory_audit"
cp -a "$SDXL_ROOT/code/sdxl_research/." "$SDXL_ROOT/code/sdxl_memory_audit/"
cp "$SDXL_ROOT/tmp/train_memory_audit.py" "$SDXL_ROOT/code/sdxl_memory_audit/train.py"
python - <<'PY'
import json
from pathlib import Path
p=Path('/root/sdxl_research/code/sdxl_memory_audit/configs/turning_B_global_a800_probe.json')
c=json.loads(p.read_text());c.update(batch=4,benchmark_batch=4,micro_batch=4,rollout_batch=4,benchmark_cycles=2,benchmark_warmup=1)
p.write_text(json.dumps(c))
PY
python "$SDXL_ROOT/code/sdxl_memory_audit/launch.py" --config "$SDXL_ROOT/code/sdxl_memory_audit/configs/turning_B_global_a800_probe.json" --root "$SDXL_ROOT" --gpus 0 --run a800_memory_audit_20261101 --min-free-gib 76 --execute
