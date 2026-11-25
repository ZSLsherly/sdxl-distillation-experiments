set -euo pipefail
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
test "$(cat outputs/research_small_20261022/toy_state.exit)" = 0
python - <<'PY'
from pathlib import Path
import json
p=Path('sdxl_research/configs/small_checks');p.mkdir(exist_ok=True)
for name in ('legacy','bucket_cycle','block'):
 c=json.loads(Path(f'sdxl_research/configs/performance/{name}.json').read_text())
 c.update(batch=8,micro_batch=4,rollout_batch=4,benchmark_batch=8,benchmark_cycles=4,benchmark_warmup=1,hours=.5)
 (p/f'{name}.json').write_text(json.dumps(c,indent=2))
PY
nohup nice -n 10 ionice -c 2 -n 7 bash -c '
python sdxl_research/launch.py --root /data5/shenglin.zeng/sdxl_crossdistill --gpus 0,3 --config sdxl_research/configs/small_checks/legacy.json --run research_small_20261022/legacy_b8 --execute
rc=$?
echo "$rc" > outputs/research_small_20261022/legacy_b8.exit
exit "$rc"
' > logs/research_small_legacy_launcher.log 2>&1 </dev/null &
echo "$!" > outputs/research_small_20261022/legacy_launcher.pid
echo "STARTED_BASELINE $(cat outputs/research_small_20261022/legacy_launcher.pid)"
