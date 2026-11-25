set -eu
. /root/sdxl_research/env.sh
python - <<'PY'
import json,os
from pathlib import Path
base=Path('/root/sdxl_research/outputs/bc_winner_late_mix_20261102')
for name,dirname in [('A','coco10k_A'),('B','coco10k'),('C','coco10k')]:
    info=json.loads((base/dirname/'manifest.json').read_text())['models'][name]
    w=Path(info['checkpoint'])/'student_rank0.pt'
    print(name,w.stat().st_size,info['shards'][0]['sha256'],w.resolve())
PY
