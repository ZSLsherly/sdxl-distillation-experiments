set -eu
python3 - <<'PY'
import json,datetime
from pathlib import Path
root=Path('/data3/shenglin.zeng/sdxl_turning_ablation_20261028/outputs/turning_batch64_g160_4gpu_teacher_shard_20261029')
print('TIME',datetime.datetime.now().isoformat())
print('ROOT',root)
for p in sorted(root.glob('*/status.json')):
 print('RUN',p.parent.name,'STATUS',p.read_text())
 cp=p.parent/'config.json'
 if cp.exists():
  c=json.loads(cp.read_text());print('CONFIG',json.dumps({k:c.get(k) for k in ['objective','schedule','turning_profile','turning_sampling','student_dmd_t_range','fake_updates','batch','student_lr','fake_lr','gan_weight','data_dir','training_state']}))
 commits=[]
 for f in (p.parent/'training_state').glob('*/complete.json'):
  d=json.loads(f.read_text());commits.append({'path':str(f),'progress':d.get('progress'),'gcount':d.get('gcount')})
 print('FINAL_STATE',json.dumps(commits[-2:]))
 logs=p.parent/'rank0.jsonl'
 if logs.exists():
  rows=[]
  for l in logs.read_text().splitlines():
   try:r=json.loads(l)
   except ValueError:continue
   if r.get('phase')=='student_done':rows.append(r)
  print('STUDENT_ROWS',len(rows),'LAST',json.dumps(rows[-1] if rows else {}))
for f in ['workflow_complete.json','workflow.log','workflow_error.json']:
 p=root/f
 if p.exists():print(f,p.read_text()[-2000:])
PY
