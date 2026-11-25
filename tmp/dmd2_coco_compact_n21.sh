python3 - <<'PY'
import json,datetime
from pathlib import Path
r=Path('/data1/shenglin.zeng/dmd2_matched_10k_20261119/coco10k_official_20261121')
b=list((r/'features/DMD2_matched_G160').glob('*.npz'));n=sum(int(p.stem.split('_')[1])-int(p.stem.split('_')[0]) for p in b)
s=[json.loads(p.read_text()) for p in sorted(r.glob('status_*.json'))]
print(json.dumps({'time':datetime.datetime.now().isoformat(),'completed':n,'workers':[{'rank':x['rank'],'done':x['done'],'target':x['target'],'batch':x['batch'],'seconds':round(x['seconds']),'peak_GiB':round(x['peak_GiB'],2)} for x in s],'all_complete':(r/'complete.json').exists()}))
for f in ['scores.json','complete.json']:
 p=r/f
 if p.exists():print(f,p.read_text())
PY
