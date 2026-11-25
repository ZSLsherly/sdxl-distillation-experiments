set -eu
. /root/sdxl_research/env.sh
python - <<'PY'
from pathlib import Path
import json
r=Path('/root/sdxl_research/outputs')
for n in ['A_pcm10_mix1_g160','C_mix1_pcm_dm4_fake4_low30_g160']:
 p=r/n;print('ARM',n)
 for f in ['model/status.json','comparison.json','complete.json','queue.log']:
  if (p/f).is_file():print(f,(p/f).read_text()[-1500:])
PY
