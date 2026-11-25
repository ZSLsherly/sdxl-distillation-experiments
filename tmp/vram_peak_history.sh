source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import json
from pathlib import Path
m=json.loads(Path('outputs/active_experiment.json').read_text());p=Path('outputs')/m['runs'][0]['run']
print('STATUS',(p/'status.json').read_text())
for rank in range(4):
 rows=[json.loads(s) for s in (p/f'rank{rank}.jsonl').read_text().splitlines()]
 gs=[r for r in rows if r['phase']=='student_done'];fs=[r for r in rows if r['phase']=='fake_D']
 jumps=[];prev=0
 for r in gs:
  peak=r['peak_allocated_GiB']
  if peak>prev+0.01:jumps.append({'g':r['g'],'stage':r['stage'],'peak_GiB':peak});prev=peak
 print('RANK',rank,'G',gs[-1]['g'] if gs else 0,'STAGES',sorted({r['stage'] for r in gs}),'R1_UPDATES',sum(bool(r['R1_active']) for r in fs),'PEAK_JUMPS',jumps)
print('EVAL_DIRS',len(list((p/'eval').glob('*'))),'WEIGHT_COMPLETIONS',len(list((p/'weights').glob('*/complete.json'))))
lines=Path(m['runs'][0]['log']).read_text(errors='replace').splitlines()
print('ERRORS',[s for s in lines if 'out of memory' in s.lower() or 'Traceback' in s][-5:])
PY
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader -l 2 -f /dev/stdout &
sample_pid=$!
sleep 10
kill "$sample_pid"
