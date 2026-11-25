set -eu
. /root/sdxl_research/env.sh
r="$SDXL_ROOT/outputs/A_clean_global_high30_g160_20261114_v2"
date -Iseconds
tail -8 "$r/queue.log"
test ! -f "$r/model/status.json" || cat "$r/model/status.json"
test ! -f "$r/first_update_validation.json" || cat "$r/first_update_validation.json"
tail -5 "$r/train.log"
python - <<'PY'
from pathlib import Path
import os,json
r=Path(os.environ['SDXL_ROOT'])/'outputs/A_clean_global_high30_g160_20261114_v2/model/rank0.jsonl'
if r.exists():
 text=r.read_text();rows=[json.loads(l) for l in text[:text.rfind('\n')].splitlines()]
 for x in [r for r in rows if r['phase']=='student_done'][-2:]:
  print('STUDENT',{k:x.get(k) for k in ['g','pcm_loss','middle_dm_loss','terminal_dm_loss','grad_norm','cycle_seconds','peak_allocated_GiB']})
PY
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader,nounits
