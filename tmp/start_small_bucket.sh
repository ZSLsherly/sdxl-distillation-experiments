set -euo pipefail
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
test "$(cat outputs/research_small_20261022/block_offload_b8.exit)" = 0
python - <<'PY'
from pathlib import Path
import json
p=Path('sdxl_research/configs/small_checks/legacy_offload.json')
c=json.loads(p.read_text());c.update(control_mode='cycle',d_grad_bucket_mb=16)
p.with_name('bucket_offload.json').write_text(json.dumps(c,indent=2))
PY
nohup nice -n 10 ionice -c 2 -n 7 bash -c '
python sdxl_research/launch.py --root /data5/shenglin.zeng/sdxl_crossdistill --gpus 0,3 --config sdxl_research/configs/small_checks/bucket_offload.json --run research_small_20261022/bucket_offload_b8 --execute
rc=$?
echo "$rc" > outputs/research_small_20261022/bucket_offload_b8.exit
exit "$rc"
' > logs/research_small_bucket_launcher.log 2>&1 </dev/null &
echo "$!" > outputs/research_small_20261022/bucket_launcher.pid
echo "STARTED_BUCKET_ONLY $(cat outputs/research_small_20261022/bucket_launcher.pid)"
