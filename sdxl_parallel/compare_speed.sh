set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import json,statistics
from pathlib import Path
from datetime import datetime
root=Path.cwd();active=json.loads((root/'outputs/active_experiment.json').read_text())
names={'four_gpu_joint34':active['runs'][0]['run'],'two_gpu_joint4':'gan_stage_parallel/joint4_20261017T232317_mb16'}
result={}
for label,name in names.items():
 p=root/'outputs'/name
 rows=[]
 for line in (p/'rank0.jsonl').read_text().splitlines():
  try:rows.append(json.loads(line))
  except json.JSONDecodeError:pass
 gs=[r for r in rows if r['phase']=='student_done'];fs=[r for r in rows if r['phase']=='fake_D']
 result[label]={'run':name,'config':json.loads((p/'config.json').read_text()),'status':json.loads((p/'status.json').read_text()),'g':gs,'f':fs}
n=min(len(x['g']) for x in result.values())
def summarize(gs,fs,lo,hi):
 selected=[r for r in gs if lo<=r['g']<=hi]
 if not selected:return None
 vals=[r['cycle_seconds'] for r in selected]
 out={'first_g':selected[0]['g'],'last_g':selected[-1]['g'],'count':len(vals),'mean_train_seconds':statistics.mean(vals),'median_train_seconds':statistics.median(vals),'samples_per_second':64/statistics.mean(vals),'by_stage':{str(s):{'n':len(a),'mean_cycle_seconds':statistics.mean(r['cycle_seconds'] for r in a),'mean_G_seconds':statistics.mean(r['seconds'] for r in a)} for s in (2,3,4) if (a:=[r for r in selected if r['stage']==s])}}
 f=[r['seconds'] for r in fs if 5*(lo-1)<r['fake_update']<=5*hi]
 if f:out['mean_fake_D_seconds']=statistics.mean(f)
 first,last=selected[0],selected[-1]
 if last['g']>first['g']:
  dt=(datetime.strptime(last['timestamp'],'%Y%m%dT%H%M%S%z')-datetime.strptime(first['timestamp'],'%Y%m%dT%H%M%S%z')).total_seconds()
  out['wall_seconds_per_update_between_endpoints']=dt/(last['g']-first['g'])
 return out
report={'created_at':datetime.now().astimezone().isoformat(),'common_completed_updates':n,'runs':{}}
for label,x in result.items():
 c=x['config'];g=x['g'];f=x['f']
 report['runs'][label]={'run':x['run'],'status':x['status'],'settings':{k:c.get(k) for k in ['variant','world_size','effective_batch','microbatch_per_gpu','accumulation','student_lr','fake_lr','fake_updates','offload_student_optimizer','gan_stages','save_every_updates','eval_every_samples']},'total_completed_updates':len(g),'same_first_window':summarize(g,f,1,n),'same_warm_window':summarize(g,f,4,n),'latest_up_to_12':summarize(g,f,max(4,len(g)-11),len(g))}
print(json.dumps(report,indent=2))
dest=root/'outputs/gan_stage_joint34/speed_comparison.json';dest.write_text(json.dumps(report,indent=2))
PY
