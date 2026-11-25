set -eu
source /data5/shenglin.zeng/env.sh
cd "$CROSSDISTILL_ROOT"
python summarize_smoke.py > logs/smoke_summary.log
python - <<'PY'
import json
x=json.load(open('outputs/smoke_summary.json'))
print(json.dumps({k:x[k] for k in ['phases','training_compute_seconds','max_allocated_GiB','max_reserved_GiB','nvidia_smi_peak_GiB_per_gpu','process_exit_code']},indent=2))
PY
