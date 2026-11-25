set -eu
. /root/sdxl_research/env.sh
cd "$SDXL_ROOT/outputs"
date -Iseconds
df -h .
python - <<'PY'
import json,statistics
from pathlib import Path
from datetime import datetime
for name in ('c_mix1_dm234_fake3_20261103','c_mix1_first_dm_fake3_g160','c_mix1_dm_pair_coco10k','c_mix1_pcm_dm4_matched_time'):
 p=Path(name)
 print(name)
 if (p/'queue.pid').exists(): print('queue_alive',Path('/proc', (p/'queue.pid').read_text().strip()).is_dir())
 for f in ('queue.log','complete.json','comparison.json'):
  if (p/f).is_file(): print(f,(p/f).read_text()[-1100:])
 for f in p.glob('*/status_0.json'):print(f,f.read_text()[-380:])
 m=p/'model'; log=m/'rank0.jsonl'
 if log.exists():
  gs=[];fs=0
  for l in log.open():
   try:r=json.loads(l)
   except ValueError:continue
   if r['phase']=='student_done':gs.append(r)
   if r['phase']=='fake':fs+=1
  if gs:
   def t(r):return datetime.strptime(r['timestamp'],'%Y%m%dT%H%M%S%z')
   print('G,F,last,wall_hours,avg_s_per_G',len(gs),fs,t(gs[-1]),round(((t(gs[-1])-t(gs[0])).total_seconds()+gs[0]['cycle_seconds'])/3600,3),round((t(gs[-1])-t(gs[0])).total_seconds()/max(1,len(gs)-1),1))
PY
