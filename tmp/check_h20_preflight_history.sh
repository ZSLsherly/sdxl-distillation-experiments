set -eu
. /root/sdxl_research/env.sh
python - <<'PY'
import os
from pathlib import Path
d=Path(os.environ['SDXL_ROOT'])/'outputs/flash_gan_g160_20261102'
for p in sorted(d.glob('gan_micro*.log')):
 txt=p.read_text(errors='replace').splitlines()
 print(p.name)
 for i,l in enumerate(txt):
  if '[rank0]' in l and '/train.py"' in l and 'in main' in l:print(l.strip(),txt[i+1].strip())
  if '[rank0]' in l and 'OutOfMemoryError:' in l: print(l.split(' If reserved')[0])
PY
