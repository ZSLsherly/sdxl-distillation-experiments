source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import json,math
from pathlib import Path
base=Path('outputs/research_small_20261022')
def rows(name):
 p=base/name/'rank0.jsonl'
 return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
a=rows('legacy_offload_b8');b=rows('block_offload_b8')
keys=['fake_loss','D_hinge','R1_raw','D_real','D_fake','fake_grad_norm','D_grad_norm','dmd_loss','gan_loss','pcm_loss','grad_norm']
diffs={};pairs=0
for x,y in zip(a,b):
 assert x['phase']==y['phase']
 assert x.get('g')==y.get('g') and x.get('fake_update')==y.get('fake_update')
 assert x['original_indices_rank']==y['original_indices_rank']
 for k in keys:
  if k in x and k in y:
   assert math.isfinite(y[k]),(k,y[k])
   diffs[k]=max(diffs.get(k,0),abs(x[k]-y[k]))
 pairs+=1
print(json.dumps({'compared_rows':pairs,'candidate_completed_G':sum(x['phase']=='student_done' for x in b),'max_absolute_log_metric_difference':diffs},indent=2))
PY
