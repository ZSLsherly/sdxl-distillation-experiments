set -euo pipefail
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
test "$(cat outputs/research_small_20261022/legacy_offload_b8.exit)" = 0
python - <<'PY'
import json
from pathlib import Path
p=Path('outputs/research_small_20261022/legacy_offload_b8')
s=json.loads((p/'benchmark.json').read_text())
assert s['status']=='complete' and s['student_updates']==4 and s['fake_updates']==20
rows=[json.loads(x) for x in (p/'rank0.jsonl').read_text().splitlines()]
assert any(r.get('gan') for r in rows)
assert any(r.get('R1_active') for r in rows)
print('BASELINE_PASS',s['peak_allocated_GiB'],s['peak_reserved_GiB'])
PY
nohup nice -n 10 ionice -c 2 -n 7 bash -c '
python sdxl_research/launch.py --root /data5/shenglin.zeng/sdxl_crossdistill --gpus 0,3 --config sdxl_research/configs/small_checks/block_offload.json --run research_small_20261022/block_offload_b8 --execute
rc=$?
echo "$rc" > outputs/research_small_20261022/block_offload_b8.exit
exit "$rc"
' > logs/research_small_block_launcher.log 2>&1 </dev/null &
echo "$!" > outputs/research_small_20261022/block_launcher.pid
echo "STARTED_BLOCK $(cat outputs/research_small_20261022/block_launcher.pid)"
