set -euo pipefail
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
from pathlib import Path
import json
p=Path('outputs/joint4_uniform_10k_2gpu_20261023')
guard=json.loads((p/'guard_result.json').read_text())
status=json.loads((p/'train/status.json').read_text())
assert guard['returncode']==0 and status['status']=='stopped'
assert Path(status['training_state'],'complete.json').exists()
print('PRESERVED_CHECKPOINT',status['training_state'])
PY
test ! -e outputs/microbatch32_20261023
mkdir -p outputs/microbatch32_20261023
nohup nice -n 10 ionice -c 2 -n 7 python run_microbatch32_20261023.py > outputs/microbatch32_20261023/guard.log 2>&1 </dev/null &
echo "$!" > outputs/microbatch32_20261023/guard.pid
echo "MB32_PROBE_STARTED $(cat outputs/microbatch32_20261023/guard.pid)"
