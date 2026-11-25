R=/data1/shenglin.zeng/dmd2_matched_10k_20261119/coco10k_official_20261121
python3 - <<'PY'
import json,datetime
from pathlib import Path
r=Path('/data1/shenglin.zeng/dmd2_matched_10k_20261119/coco10k_official_20261121')
blocks=list((r/'features/DMD2_matched_G160').glob('*.npz'))
done=sum(int(p.stem.split('_')[1])-int(p.stem.split('_')[0]) for p in blocks)
print('TIME',datetime.datetime.now().isoformat(),'CACHED',done,'/10000','BLOCKS',len(blocks))
for p in sorted(r.glob('status_*.json')):print(p.name,p.read_text())
for name in ['scores.json','clip_scores.json','complete.json']:
 p=r/name
 if p.exists():
  v=json.loads(p.read_text());print(name,{k:z for k,z in v.items() if k not in ['alignment','cases']})
PY
tail -n 4 "$R/logs/queue.log"
