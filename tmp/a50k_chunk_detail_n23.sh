python3 -u - <<'PY'
from pathlib import Path
r=Path('/data5/shenglin.zeng/datasets/a50k_sources_20261123')
for d in r.glob('*.chunks'):
 done=[p for p in d.iterdir() if p.name.isdigit()]
 pending=[(p.name,p.stat().st_size) for p in d.glob('*.tmp')]
 print(d.name,'COMPLETE_CHUNKS',len(done),'PENDING',pending,flush=True)
print('COMPLETE_SHARDS',[(p.name,p.stat().st_size) for p in r.glob('*.pt')],flush=True)
PY
