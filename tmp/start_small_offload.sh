set -euo pipefail
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
from pathlib import Path
import json
for name in ('legacy','block'):
 p=Path(f'sdxl_research/configs/small_checks/{name}.json')
 c=json.loads(p.read_text());c['offload_student_optimizer']=True
 p.with_name(name+'_offload.json').write_text(json.dumps(c,indent=2))
PY
nohup nice -n 10 ionice -c 2 -n 7 bash -c '
python sdxl_research/launch.py --root /data5/shenglin.zeng/sdxl_crossdistill --gpus 0,3 --config sdxl_research/configs/small_checks/legacy_offload.json --run research_small_20261022/legacy_offload_b8 --execute
rc=$?
echo "$rc" > outputs/research_small_20261022/legacy_offload_b8.exit
exit "$rc"
' > logs/research_small_legacy_offload_launcher.log 2>&1 </dev/null &
echo "$!" > outputs/research_small_20261022/legacy_offload_launcher.pid
echo "STARTED_BASELINE_OFFLOAD $(cat outputs/research_small_20261022/legacy_offload_launcher.pid)"
