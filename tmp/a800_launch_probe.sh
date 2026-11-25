set -eu
. /root/sdxl_research/env.sh
export PYTHONPATH="$SDXL_ROOT/code/sdxl_research"
python - <<'PY'
from turning_sampling import TurningSampler
s=TurningSampler('/root/sdxl_research/code/sdxl_research/turning_profiles/normalized_u_512_979.json',979)
for r in [(509,979),(259,979),(20,979)]:
 t,p=s.distribution(r);assert abs(sum(p)-1)<1e-12
 print('PRIOR_OK',r)
PY
nohup python -u "$SDXL_ROOT/tmp/a800_batch_probe.py" > "$SDXL_WORK/logs/a800_batch_probe.log" 2>&1 < /dev/null &
echo PROBE_PID=$!
