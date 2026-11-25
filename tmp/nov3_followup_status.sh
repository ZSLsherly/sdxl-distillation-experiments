set -eu
. /root/sdxl_research/env.sh
date '+%F %T %z'
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
ps -eo pid,comm,args | awk '$2 ~ /python/ && /run_bc_coco_then_winner|run_flash_after_winner|evaluate.py|train.py/ {print}'
python - <<'PY'
import os,json
from pathlib import Path
o=Path(os.environ['SDXL_ROOT'])/'outputs'; p=o/'bc_winner_late_mix_20261102'; q=o/'flash_gan_g160_20261102'
for name in ('coco10k','coco10k_A'):
 m=json.loads((p/name/'complete.json').read_text());print(name,m['scores'])
state=json.loads((p/'late_mix/status.json').read_text()); print('mix',state['status'],state['student_updates'],state['fake_updates'],(p/'training_complete.json').is_file())
r=json.loads((p/'training_complete.json').read_text()); checks=r['heldout_checks'];print('probe G', [x['g'] for x in checks]);print('g160',checks[-1]['results'])
print('checkpoint validation',[json.loads(f.read_text()) for f in (p/'late_mix/weights').glob('*/cpu_validation.json')])
for d in sorted(q.glob('gan_micro*')):
 if not d.is_dir(): continue
 rows=[json.loads(l) for l in (d/'rank0.jsonl').read_text().splitlines()];log=(q/(d.name+'.log')).read_text(errors='replace');print(d.name,'G',sum(x['phase']=='student_done' for x in rows),'OOM','out of memory' in log.lower())
print('gan complete',(q/'complete.json').exists());print('queue log last line',(q/'queue.log').read_text().splitlines()[-1])
print('recent output dirs',sorted([(str(d.name),int(d.stat().st_mtime)) for d in o.iterdir() if d.is_dir()],key=lambda x:x[1],reverse=True)[:5])
PY
