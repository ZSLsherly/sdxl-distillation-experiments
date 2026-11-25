python3 -u - <<'PY'
import time
from pathlib import Path
r=Path('/data5/shenglin.zeng/datasets/a50k_sources_20261123')
def sizes():
 return {p.name:sum(x.stat().st_size for x in p.iterdir() if x.is_file()) for p in r.glob('*.chunks')}
a=sizes();t=time.monotonic();print('SIZES_START',a,flush=True)
time.sleep(20)
b=sizes();dt=time.monotonic()-t
print('SIZES_END',b,'SECONDS',round(dt,2),'MB_PER_SECOND',round((sum(b.values())-sum(a.values()))/dt/1e6,3),flush=True)
print('COMPLETE_FILES',[(p.name,p.stat().st_size) for p in r.glob('*.pt')],flush=True)
PY
