set -eu
. /root/sdxl_research/env.sh
export CUDA_VISIBLE_DEVICES=''
export PYTHONPATH="$SDXL_ROOT/code/sdxl_research"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
if test -f "$SDXL_ROOT/outputs/turning_BC_paired_quota_20261102/uniform_control/config.json"; then
 python "$SDXL_ROOT/tmp/a800_overnight_audit.py" uniform_control
else
 python "$SDXL_ROOT/tmp/a800_overnight_audit.py" C
fi
df -h /root/autodl-tmp /root
ps -p 12131,42356,61613 -o pid,ppid,etimes,args || true
python - <<'PY'
import json
from pathlib import Path
p=Path('/root/sdxl_research/outputs/turning_BC_paired_quota_20261102')
for arm in ['B','C','uniform_control']:
 q=p/arm
 if not (q/'status.json').exists():continue
 print('STATUS',arm,(q/'status.json').read_text())
 rows=[]
 for line in (q/'rank0.jsonl').read_text().splitlines():
  try:rows.append(json.loads(line))
  except json.JSONDecodeError:break
 for r in [x for x in rows if x['phase']=='student_done'][-3:]:
  print(arm,{k:r[k] for k in ['g','cycle_seconds','grad_norm','pcm_loss','middle_dm_loss','terminal_dm_loss','peak_allocated_GiB']})
 e=sorted((q/'eval').glob('*/rank0.json'))[-1]
 meta=json.loads(e.read_text())
 print('EVAL',arm,str(e),{'count':len(meta['samples']),'stds':[s['latent_std'] for s in meta['samples']]})
PY
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
