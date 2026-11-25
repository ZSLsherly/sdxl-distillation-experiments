set -euo pipefail
cd /data1/shenglin.zeng/sdxl_eval_uniform_20261025
runtime/bin/python - <<'PY'
from pathlib import Path
import json
p=Path('outputs/joint34_uniform_10k_2gpu_20261025')
x={'events':[json.loads(s) for s in (p/'workflow.jsonl').read_text().splitlines()][-3:]}
for phase in ['phase1','train']:
 q=p/phase
 for name in ['status.json','config.json']:
  if (q/name).exists():
   d=json.loads((q/name).read_text());x[phase+'/'+name]=d if name=='status.json' else {k:d[k] for k in ['variant','gan_stages','batch','micro_batch','world_size','train_count','max_student_updates']}
 if (q/'rank0.jsonl').exists():
  rows=[json.loads(s) for s in (q/'rank0.jsonl').read_text().splitlines()]
  x[phase+'/last_updates']=[{k:v for k,v in row.items() if k in ['phase','g','fake_update','stage','gan','grad_norm','seconds','cycle_seconds','gan_loss','pcm_loss','peak_allocated_GiB']} for row in rows[-3:]]
for log in p.glob('*_0.log'):x[log.name]=log.read_text(errors='replace').splitlines()[-4:]
if (p/'workflow_stopped.json').exists():x['stopped']=json.loads((p/'workflow_stopped.json').read_text())
print(json.dumps(x,indent=2))
PY
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
