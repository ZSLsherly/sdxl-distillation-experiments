set -eu
. /root/sdxl_research/env.sh
ps -p 88668 -o pid,ppid,etime,args
cat "$SDXL_WORK/logs/bc_coco_winner_queue.log"
cat "$SDXL_ROOT/outputs/turning_BC_paired_quota_20261102/uniform_control/status.json"
python - <<'PY'
from pathlib import Path
import json
p=Path('/root/sdxl_research/outputs/bc_winner_late_mix_20261102/coco10k/manifest.json');e=Path('/root/autodl-tmp/sdxl_runs/bc_texture_inputs/public_inputs_ready.json');print('public assets',e.is_file());print('manifest ready',p.is_file())
if p.exists():
 m=json.loads(p.read_text());print('protocol',m['n'],list(m['models']),len(m['visual_indices']))
PY
