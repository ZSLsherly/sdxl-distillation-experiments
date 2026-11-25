set -eu
p=/root/autodl-tmp/sdxl_runs/outputs
du -h -d 1 "$p" | sort -hr | head -18
du -xh -d 1 /root/sdxl_research | sort -hr | head -12
/root/sdxl_research/env/bin/python - <<'PY'
from pathlib import Path
from collections import Counter
r=Path('/root/autodl-tmp/sdxl_runs/outputs');a=Counter();n=Counter()
for p in r.rglob('*'):
 if not p.is_symlink() and p.is_file():
  a[p.suffix]+=p.stat().st_size;n[p.suffix]+=1
for k,v in a.most_common(8):print(k,n[k],round(v/2**30,2))
PY
