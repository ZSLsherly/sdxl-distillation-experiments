set -eu
. /root/sdxl_research/env.sh
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
tar -xzf "$SDXL_ROOT/tmp/quota_bc_code.tar.gz" -C "$SDXL_ROOT/code"
export PYTHONPATH="$SDXL_ROOT/code/sdxl_research"
python -m unittest test_interval_quota test_masked_turning test_turning_sampling test_boundary_global_noise test_boundary test_turning_batch64_workflow
python - <<'PY'
import json
from pathlib import Path
p=Path('/root/sdxl_research/outputs/turning_BC_masked_20261101')
(p/'superseded.json').write_text(json.dumps({'reason':'User changed to direct interval quotas: prior30%, complement70%; uniform inside each bucket','replacement':'turning_BC_interval_quota_20261101','old_queue_stopped':True},indent=2))
PY
nohup python -u "$SDXL_ROOT/code/sdxl_research/run_quota_bc.py" > "$SDXL_WORK/logs/quota_bc_queue.log" 2>&1 < /dev/null &
echo QUEUE_PID=$!
