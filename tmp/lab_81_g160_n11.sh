python3 - <<'PY'
from pathlib import Path
import json
r=Path('/data3/shenglin.zeng/sdxl_turning_ablation_20261028/outputs')
for n in ['normalized_u_g160','raw_angle_g160']:
 p=r/'turning_batch64_g160_4gpu_teacher_shard_20261029'/n
 c=json.loads((p/'config.json').read_text());print(n,'schedule',c['schedule'])
 for q in sorted((p/'weights').glob('*/complete.json')):
  m=json.loads(q.read_text())
  if m.get('student_updates')==160:print('G160',q.parent,'world',m['world_size'],'shards',len(list(q.parent.glob('student_rank*.pt'))))
print('SCORES',list(r.rglob('scores.json')))
PY
