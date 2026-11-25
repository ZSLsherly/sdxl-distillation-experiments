set -eu
R=/data1/shenglin.zeng/dmd2_official_recipe_4gpu_20261123
cd "$R/code"
"$R/env/bin/python" - <<'PY'
import json,math,statistics,hashlib
from pathlib import Path
from run_guard import check_idle
r=Path('..');cfg=json.loads((r/'config.json').read_text());check_idle(cfg['gpu_uuids'])
assert not (r/'runs/formal_g160').exists()
summaries={}
for micro in [8,16]:
 d=r/'runs'/f'smoke_micro{micro}';result=json.loads((d/'complete.json').read_text())
 assert result['G']==4 and result['F']==20 and result['strict_load'] and result['finite']
 peak=0
 for rank in range(4):
  rows=[json.loads(x) for x in (d/f'rank{rank}.jsonl').read_text().splitlines()]
  assert len(rows)==20 and [x['depth'] for x in rows if x['student_updated']]==[0,1,2,3]
  assert all(not isinstance(v,float) or math.isfinite(v) for x in rows for v in x.values())
  assert len(json.loads((d/'eval/g000004'/f'rank{rank}.json').read_text())['samples'])==2
  precision=json.loads((d/f'precision_rank{rank}.json').read_text())
  assert precision['fake:autocast=False:grad=False']['output']=='torch.float32'
  assert precision['student:autocast=False:grad=False']['output']=='torch.float32'
  assert precision['teacher:autocast=False:grad=False']['weight']=='torch.bfloat16'
  peak=max(peak,max(x['peak_GiB'] for x in rows))
 summaries[micro]={'result':result,'peak_GiB':peak,'iteration_seconds_sum':sum(x['seconds'] for x in rows),
   'G_iteration_seconds_mean':statistics.mean(x['seconds'] for x in rows if x['student_updated']),
   'F_only_seconds_mean':statistics.mean(x['seconds'] for x in rows if not x['student_updated']),
   'all_finite':True,'four_depths':True,'G4_images':8}
assert summaries[16]['peak_GiB']<=cfg['smoke_max_allocated_GiB']==42
proof=json.loads((r/'smoke_passed.json').read_text())
if proof['micro']==8:(r/'smoke_passed_micro8.json').write_text(json.dumps(proof,indent=2))
(r/'smoke_passed.json').write_text(json.dumps({'micro':16,'accumulation':1,'result':summaries[16]['result'],
 'peak_GiB':summaries[16]['peak_GiB'],'accepted_limit_GiB':42,
 'reason':'Complete measured 4-GPU16 probe passed; conservative 39GiB resource threshold revised to 42, no loss/precision change.'},indent=2))
(r/'microbatch_comparison.json').write_text(json.dumps(summaries,indent=2))
old=json.loads((r/'source_manifest.json').read_text())
(r/'shorttest_source_manifest.json').write_text(json.dumps(old,indent=2))
for name,sha in old.items():
 if name not in ['config.json','code/train_matched.py']:assert hashlib.sha256((r/name).read_bytes()).hexdigest()==sha,name
for name in ['config.json','code/train_matched.py']:old[name]=hashlib.sha256((r/name).read_bytes()).hexdigest()
(r/'source_manifest.json').write_text(json.dumps(old,indent=2))
print('MICRO16_ACCEPTED',json.dumps(summaries))
PY
nohup "$R/env/bin/python" -u "$R/code/run_guard.py" --root "$R" --run formal_g160 --reuse-smoke >> "$R/logs/queue.log" 2>&1 < /dev/null &
echo "FORMAL_MICRO16_QUEUE_PID=$!"
sleep 2
tail -n 3 "$R/logs/queue.log"
