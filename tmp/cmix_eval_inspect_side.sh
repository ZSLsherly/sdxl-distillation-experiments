set -eu
. /root/sdxl_research/env.sh
date '+%F %T %z'
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader
ps -eo pid,comm,args | awk '$2 ~ /python/ && /train.py|evaluate.py|semantic|interval_D|clip|winner/ {print}' | head -20
find "$SDXL_WORK/outputs" -maxdepth 3 -type f \( -name 'scores.json' -o -name 'selection.json' -o -name 'training_complete.json' \) -printf '%P\n' | tail -30
python - <<'PY'
import json,os
from pathlib import Path
r=Path(os.environ['SDXL_ROOT']);w=Path(os.environ['SDXL_WORK']);p=r/'outputs/bc_winner_late_mix_20261102';j=p/'late_mix'
for f in ('status.json','config.json'):
 d=json.loads((j/f).read_text());print(f,{k:d[k] for k in ('variant','fake_student_mix','turning_profile','fake_mix_start_g','student_updates','fake_updates') if k in d})
print('weights',[(str(s),s.stat().st_size) for s in (j/'weights').glob('*/student_rank0.pt')])
print('D',json.loads((r/'outputs/interval_D_20261103/D/status.json').read_text()))
print('cap',os.statvfs(w).f_bavail*os.statvfs(w).f_frsize)
PY
