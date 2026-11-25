set -eu
. /root/sdxl_research/env.sh
p="$SDXL_ROOT/outputs/interval_D_20261103"
cat "$p/queue.log"
tail -4 "$p/D.log"
python - <<'PY'
import os,json
from pathlib import Path
p=Path(os.environ['SDXL_ROOT'])/'outputs/interval_D_20261103/D'
for f in ['config.json','status.json']:
 if (p/f).exists():
  r=json.loads((p/f).read_text()); print(f,{k:r[k] for k in ('status','student_updates','fake_updates','phase','micro_finished','effective_batch','variant','match_fake_prior','turning_profile','max_student_updates') if k in r})
log=p/'rank0.jsonl'
if log.exists():
 rows=[json.loads(l) for l in log.read_text().splitlines() if l.strip()]
 print('completed G',sum(r['phase']=='student_done' for r in rows),'completed fake',sum(r['phase']=='fake' for r in rows))
 for phase in ('fake_turning_sample','boundary_turning_sample'):
  selected=[r for r in rows if r['phase']==phase]
  if selected:print(phase,'recorded stages',sorted(set(r['stage'] for r in selected)))
PY
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
