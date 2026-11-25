set -eu
. /root/sdxl_research/env.sh
python - <<'PY'
import os,json
from pathlib import Path
o=Path(os.environ['SDXL_ROOT'])/'outputs'
for p in (o/'interval_D_20261103').rglob('STOP_REQUEST'):
 print(p.relative_to(o),p.read_text()[:400])
r=o/'bc_winner_late_mix_20261102'
m=json.loads((r/'coco10k/manifest.json').read_text())
d=r/'coco10k_C_mix1';t=json.loads((d/'manifest.json').read_text())
print('same samples/protocol/pictures',*[t.get(k)==m.get(k) for k in ('samples','protocol','visual_indices')])
print('mix complete',(d/'complete.json').read_text())
PY
