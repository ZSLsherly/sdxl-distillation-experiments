set -eu
. /root/sdxl_research/env.sh
date '+%F %T %z'
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
ps -eo pid,comm,args | awk '$2 ~ /python/ && /train.py|evaluate.py|run_interval_D|run_flash_after_winner/ {print substr($0,1,310)}'
python - <<'PY'
from pathlib import Path
import os,json
root=Path(os.environ['SDXL_ROOT']);work=Path(os.environ['SDXL_WORK']);o=root/'outputs'
print('recent',[(x.name) for x in sorted(o.iterdir(),key=lambda x:x.stat().st_mtime,reverse=True)[:10]])
for d in [o/'interval_D_20261103',o/'flash_gan_g160_20261102',o/'bc_winner_late_mix_20261102']:
 print('RUN',d.name)
 for s in sorted(d.glob('*/status.json')):
  j=json.loads(s.read_text());print(s.parent.name,{k:j[k] for k in ('status','student_updates','fake_updates','D_updates','timestamp','phase') if k in j})
 for f in [d/'complete.json',d/'training_complete.json',d/'D'/'STOP_REQUEST',d/'STOP_REQUEST',d/'USER_STOP']:
  if f.exists():print('MARKER',f.name)
 for f in [d/'queue.log',d/'D.log']:
  if f.is_file():print(f.name,f.read_bytes()[-1800:].decode(errors='replace').splitlines()[-3:])
 if d.name.startswith('interval_D'):
  for f in d.glob('D/weights/*/complete.json'):print('saved',f.parent.name,json.loads(f.read_text()).get('student_updates'))
for path in sorted(o.glob('*/**/scores.json')):
 if 'turning_BC_paired_quota' not in str(path) and path.stat().st_size<6000:
  print('scores',path.relative_to(o),path.read_text()[:350])
print('20k verified',(work/'datasets/parallel_20k_seed20261018/validation.json').is_file())
PY
