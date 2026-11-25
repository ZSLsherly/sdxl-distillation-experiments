source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import json,os,datetime
from pathlib import Path
print('NOW',datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
m=json.loads(Path('outputs/active_experiment.json').read_text())
for run in m['runs']:
 p=Path('outputs')/run['run'];print('RUN',run['run']);print('STATUS',(p/'status.json').read_text())
 for rank in range(m['world_size']):
  pid=int((p/f'pid_rank{rank}').read_text());print('RANK',rank,'PID',pid,'ALIVE',Path(f'/proc/{pid}').exists())
 f=p/'rank0.jsonl';rows=[json.loads(s) for s in f.read_text().splitlines()];gs=[r for r in rows if r['phase']=='student_done']
 print('LAST_ROW',json.dumps({k:rows[-1].get(k) for k in ['phase','g','fake_update','timestamp','seconds']}))
 if gs:print('LAST_STUDENT',json.dumps({k:gs[-1].get(k) for k in ['g','timestamp','cycle_seconds','peak_allocated_GiB']}))
 lines=Path(run['log']).read_text(errors='replace').splitlines()
 print('ERRORS',[s for s in lines if any(w in s.lower() for w in ['out of memory','traceback','childfailederror','nccl error'])][-8:])
 print('SAVED',[json.loads(q.read_text()).get('student_updates') for q in (p/'weights').glob('*/complete.json')])
PY
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
