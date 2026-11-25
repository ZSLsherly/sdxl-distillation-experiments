set -eu
. /root/sdxl_research/env.sh
date -Iseconds
cd "$SDXL_ROOT/outputs"
python - <<'PY'
from pathlib import Path
import json
for n in ('A_mid_clean_x0','A_pcm10_mix1_g160'):
 p=Path(n);print(n)
 for f in ('model/status.json','comparison.json','queue.log'):
  if not (p/f).exists():continue
  s=(p/f).read_text()
  if f.endswith('log'):print(s[-350:]);continue
  r=json.loads(s)
  print(f,r if f=='comparison.json' else {k:r.get(k) for k in ('status','student_updates','fake_updates','timestamp')})
 for f in p.glob('*/status_0.json'):
  r=json.loads(f.read_text());print(f.parent.name,r.get('done'),r.get('target'))
PY
