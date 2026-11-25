set -eu
. /root/sdxl_research/env.sh
date -Iseconds
cd "$SDXL_ROOT/outputs"
python - <<'PY'
import json
from pathlib import Path
for name in ['A_pcm10_terminal_patchgan','A_mid_clean_x0']:
 p=Path(name);print(name)
 for f in ['model/status.json','comparison.json','queue.log']:
  if (p/f).exists():
   s=(p/f).read_text()
   if f.endswith('log'):print(s[-700:]);continue
   r=json.loads(s)
   print(f,r if 'comparison' in f else {k:r.get(k) for k in ['status','student_updates','fake_updates','D_updates','timestamp']})
 for f in p.glob('*/status_0.json'):
  r=json.loads(f.read_text());print(f, {k:r.get(k) for k in ['done','target','seconds']})
PY
