set -eu
r=/data1/shenglin.zeng/coco_missing_20261111
/data1/shenglin.zeng/teacher_turning_20261026/env/bin/python - <<'PY'
from pathlib import Path
import json,hashlib,tarfile
r=Path('/data1/shenglin.zeng/coco_missing_20261111')
for n in ['A_pcm10_mix1','normalized_u_g160','raw_angle_g160']:
 p=r/'results'/n;print(n)
 for f in ['complete.json','scores.json','status_0.json']:
  if (p/f).exists():print(f,json.loads((p/f).read_text()))
print('QUEUE', (r/'queue.log').read_text()[-1600:])
print('CODE_HASHES',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'code').glob('*.py')})
files=list((r/'results').glob('*/complete.json'))+list((r/'results').glob('*/scores.json'))+list((r/'results').glob('*/status_0.json'))+list((r/'results').glob('*/manifest.json'))+list((r/'checkpoints').glob('*/config.json'))+[r/'queue.log',r/'old_checkpoints_cpu_validation.json']
with tarfile.open(r/'audit_metadata_20261112.tar.gz','w:gz') as t:
 for p in files:
  if p.exists():t.add(p,arcname=str(p.relative_to(r)))
PY
nvidia-smi -i 4 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
