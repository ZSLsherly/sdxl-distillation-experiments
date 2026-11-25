set -euo pipefail
root=/data1/shenglin.zeng/teacher_turning_20261026
cd "$root"
python3 - <<'PY'
import json,zipfile
from pathlib import Path
p=Path('outputs/turning_boundary_ablation_20261027');run=p/'short_normalized_u_retry1'
rows=[json.loads(s) for s in (run/'rank0.jsonl').read_text().splitlines() if s.strip()]
fake=[r for r in rows if r.get('phase')=='fake'];student=[r for r in rows if r.get('phase')=='student_done']
result={'status':'bounded_GPU_probe_incomplete','reason':'20 minute wall limit reached',
 'fake_updates_completed':len(fake),'student_updates_completed':len(student),'raw_angle_started':(p/'short_raw_angle_retry1').exists(),
 'fake_update_records':fake,'formal_training_started':False,
 'resource_error':json.loads((p/'paired_GPU_probe_retry1_error.json').read_text())}
(p/'retry1_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
with zipfile.ZipFile('turning_probe_records_20261027.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in p.glob('*.json'):z.write(str(f),'workflow/'+f.name)
 for f in p.glob('*.log'):z.write(str(f),'workflow/'+f.name)
 for name in ['config.json','status.json','rank0.jsonl','source_sha256.json']:
  if (run/name).exists():z.write(str(run/name),'normalized_retry1/'+name)
PY
