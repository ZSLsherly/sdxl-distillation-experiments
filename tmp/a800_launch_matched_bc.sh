set -eu
. /root/sdxl_research/env.sh
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
tar -xzf "$SDXL_ROOT/tmp/matched_quota_code.tar.gz" -C "$SDXL_ROOT/code"
export PYTHONPATH="$SDXL_ROOT/code/sdxl_research"
python -m unittest test_matched_prior test_interval_quota test_masked_turning test_turning_sampling test_boundary_global_noise test_boundary test_turning_batch64_workflow
python - <<'PY'
import json
from pathlib import Path
p=Path('/root/sdxl_research/outputs/turning_BC_interval_quota_20261101')
(p/'superseded.json').write_text(json.dumps({'reason':'Align fake and student re-noise distributions as requested; independent RNG','replacement':'turning_BC_matched_quota_20261101','old_queue_stopped':True},indent=2))
PY
nohup python -u "$SDXL_ROOT/code/sdxl_research/run_quota_bc.py" > "$SDXL_WORK/logs/matched_quota_bc_queue.log" 2>&1 < /dev/null &
echo MATCHED_QUEUE_PID=$!
