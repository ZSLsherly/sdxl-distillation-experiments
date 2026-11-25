set -eu
. /root/sdxl_research/env.sh
python - <<'PY'
from pathlib import Path
import os,json
r=Path(os.environ['SDXL_ROOT'])/'outputs'
for name in ['A_pcm10_terminal_low30_g160_20261113','A_clean_global_g160_20261113','A_pcm1_terminal_dm_g160_20261113','A_clean_global_high30_g160_20261114_v2']:
 p=r/name;print('GROUP',name)
 for f in ['complete.json','model/status.json']:
  q=p/f
  print(f,json.loads(q.read_text()) if q.exists() else 'missing')
 q=p/'complete.json'
 if q.exists():
  ck=Path(json.loads(q.read_text())['checkpoint'])
  print('WEIGHT_PRESENT',(ck/'student_rank0.pt').exists(),'CPU_VALIDATION', (ck/'cpu_validation.json').read_text() if (ck/'cpu_validation.json').exists() else 'missing')
PY
