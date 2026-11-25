set -euo pipefail
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import json,math,statistics,tarfile
from pathlib import Path
root=Path('outputs/joint4_uniform_10k_2gpu_mb16_20261023');p=root/'train'
old=Path('outputs/joint4_uniform_10k_2gpu_20261023/train')
status=json.loads((p/'status.json').read_text());guard=json.loads((root/'guard_result.json').read_text())
assert status['status']=='complete' and status['student_updates']==160 and guard['returncode']==0
merged={}
for rank in (0,1):
 rows=[]
 for q in (old,p):rows += [json.loads(x) for x in (q/f'rank{rank}.jsonl').read_text().splitlines()]
 gs=[x for x in rows if x['phase']=='student_done'];fs=[x for x in rows if x['phase']=='fake_D']
 assert [x['g'] for x in gs]==list(range(1,161))
 assert [x['fake_update'] for x in fs]==list(range(1,801))
 assert all(math.isfinite(v) for row in rows for v in row.values() if isinstance(v,float))
 merged[str(rank)]=rows
weights=[]
for q in sorted((p/'weights').glob('*')):
 m=json.loads((q/'complete.json').read_text())
 assert all((q/f'student_rank{rank}.pt').stat().st_size>0 for rank in (0,1))
 weights.append({'path':str(q),'metadata':m})
state=Path(status['training_state']);complete=json.loads((state/'complete.json').read_text())
assert complete['progress']=={'gcount':160,'fcount':800,'dcount':800}
for rank in (0,1):
 for prefix in ['student','fake','discriminator','optimizer_og','optimizer_of','optimizer_od','rng']:
  assert (state/f'{prefix}_rank{rank}.pt').stat().st_size>0
selected={}
for step,where in [(0,old),(40,p),(80,p),(120,p),(160,p)]:
 found=sorted((where/'eval').glob(f'*_g{step:06d}_*'))
 assert found,(step,where)
 selected[str(step)]=found[-1]
summaries={}
for step,d in selected.items():
 rows=[]
 for rank in (0,1):rows += json.loads((d/f'rank{rank}.json').read_text())['samples']
 assert len(rows)==8
 summaries[step]={'path':str(d),'samples':rows}
rows=merged['0'];gs=[x for x in rows if x['phase']=='student_done'];fs=[x for x in rows if x['phase']=='fake_D']
steady=[x['cycle_seconds'] for x in gs if x['g']>=10]
windows={}
for label,lo,hi in [('early',1,40),('middle',61,100),('late',121,160)]:
 gg=[x for x in gs if lo<=x['g']<=hi];ff=[x for x in fs if (lo-1)*5<x['fake_update']<=hi*5]
 windows[label]={'G_range':[lo,hi],'DMD_by_stage':{str(s):statistics.mean(x['dmd_loss'] for x in gg if x['stage']==s) for s in (2,3,4)},
 'GAN_loss':statistics.mean(x['gan_loss'] for x in gg if x['gan']),
 'PCM_loss':statistics.mean(x['pcm_loss'] for x in gg),'D_hinge':statistics.mean(x['D_hinge'] for x in ff),
 'D_real':statistics.mean(x['D_real'] for x in ff),'D_fake':statistics.mean(x['D_fake'] for x in ff)}
monitor=[json.loads(x) for x in (root/'resource_monitor.jsonl').read_text().splitlines()]
result={'status':status,'guard':guard,'config':json.loads((p/'config.json').read_text()),'rows':merged,
 'weights':weights,'final_state_complete':complete,'evals':summaries,'window_metrics':windows,
 'steady_mean_cycle_seconds':statistics.mean(steady),'steady_median_cycle_seconds':statistics.median(steady),
 'stage_counts':{str(s):sum(x['stage']==s for x in gs) for s in (2,3,4)},
 'GAN_updates':sum(x['gan'] for x in gs),'R1_updates':sum(x['R1_active'] for x in fs),
 'foreign_detected':any(x['foreign'] for x in monitor),
 'data_manifest':json.loads((p/'data_manifest.json').read_text()),
 'own_workers_exited':{str(rank):not Path('/proc',(p/f'pid_rank{rank}').read_text().strip()).exists() for rank in (0,1)}}
target=root/'results_snapshot_20261025.json';target.write_text(json.dumps(result,indent=2))
archive=root/'results_snapshot_20261025.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 tar.add(target,arcname='results.json')
 for step,d in selected.items():
  for f in d.iterdir():
   if f.suffix in ('.png','.json'):tar.add(f,arcname=f'images/g{int(step):03d}/{f.name}')
print(json.dumps({k:v for k,v in result.items() if k in ['status','steady_mean_cycle_seconds','steady_median_cycle_seconds','window_metrics','stage_counts','GAN_updates','R1_updates','own_workers_exited','foreign_detected']},indent=2))
print('WEIGHT_STEPS',[x['metadata']['student_updates'] for x in weights])
print('ARCHIVE',str(archive),archive.stat().st_size)
PY
