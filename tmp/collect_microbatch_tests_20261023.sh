set -euo pipefail
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import json,math,statistics
from pathlib import Path
records={}
for name in ('microbatch32_20261023','microbatch32_expand_20261023','microbatch16_20261023'):
 p=Path('outputs')/name;q=p/'probe'
 guard=json.loads((p/'guard_result.json').read_text())
 rows=[json.loads(x) for x in (q/'rank0.jsonl').read_text().splitlines()]
 monitor=[json.loads(x) for x in (p/'resource_monitor.jsonl').read_text().splitlines()]
 assert all(not x['foreign'] for x in monitor)
 log=Path('logs',name+'_probe.log').read_text()
 record={'guard':guard,'rows':rows,'OOM_lines':[x for x in log.splitlines() if 'OutOfMemoryError' in x],
         'allocator_warning':'expandable_segments not supported' in log,'foreign_processes_detected':False}
 if name=='microbatch16_20261023':
  status=json.loads((q/'benchmark.json').read_text())
  assert guard['returncode']==0 and status['status']=='complete'
  gs=[x for x in rows if x['phase']=='student_done'];fs=[x for x in rows if x['phase']=='fake_D']
  assert len(gs)==4 and len(fs)==20 and [x['stage'] for x in gs]==[2,3,4,2]
  assert [x['g'] for x in gs if x['gan']]==[3]
  assert [x['fake_update'] for x in fs if x['R1_active']]==[16]
  for rank in (0,1):
   rr=[json.loads(x) for x in (q/f'rank{rank}.jsonl').read_text().splitlines()]
   assert len(rr)==24
   for row in rr:assert all(math.isfinite(v) for v in row.values() if isinstance(v,float))
   assert not Path('/proc', (q/f'pid_rank{rank}').read_text().strip()).exists()
  record.update(status=status,mean_timed_cycle_s=statistics.mean([x['cycle_seconds'] for x in gs[1:]]),
                config=json.loads((q/'config.json').read_text()),source_sha256=json.loads((q/'source_sha256.json').read_text()))
  peaks={}
  for m in monitor:
   for line in m['gpus'].splitlines():
    idx,used,free,util=[int(x.strip()) for x in line.split(',')]
    x=peaks.setdefault(str(idx),{'max_used_MiB':0,'min_free_MiB':100000})
    x['max_used_MiB']=max(x['max_used_MiB'],used);x['min_free_MiB']=min(x['min_free_MiB'],free)
  record['device_peaks']=peaks
 records[name]=record
base=Path('outputs/joint4_uniform_10k_2gpu_20261023/train')
records['original_mb4']={'status':json.loads((base/'status.json').read_text()),
 'rows':[json.loads(x) for x in (base/'rank0.jsonl').read_text().splitlines()]}
out=Path('outputs/microbatch16_20261023/results.json')
out.write_text(json.dumps(records,indent=2))
for k,v in records.items():
 print(k,json.dumps({key:v[key] for key in ['status','mean_timed_cycle_s','device_peaks','OOM_lines','allocator_warning'] if key in v}))
PY
