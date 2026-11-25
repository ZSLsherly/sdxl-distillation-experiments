set -euo pipefail
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import hashlib,json,math,statistics,subprocess
from pathlib import Path
root=Path('outputs/gpu_smoke_20261023');p=root/'default_b8_g4'
guard=json.loads((root/'guard_result.json').read_text())
status=json.loads((p/'benchmark.json').read_text())
config=json.loads((p/'config.json').read_text())
assert config['effective_batch']==8 and config['world_size']==2 and config['microbatch_per_gpu']==4
assert config['experiment_modules']=={'student_stages':[2,3,4],'dmd_stages':[2,3,4],
 'gan_stages':[4],'fake_policy':'crossdistill','gan_loss':'hinge','discriminator':'hybrid',
 'gan_view':'tile','gan_augment':'d_only'},config['experiment_modules']
assert guard['returncode']==0 and guard['stop_reason'] is None,guard
assert status['status']=='complete' and status['student_updates']==4 and status['fake_updates']==20 and status['D_updates']==20,status
allrows={}
numeric=['fake_loss','fake_grad_norm','D_hinge','R1_raw','D_real','D_fake','D_grad_norm',
         'dmd_loss','gan_loss','pcm_loss','grad_norm','cycle_seconds']
for rank in (0,1):
 rows=[json.loads(x) for x in (p/f'rank{rank}.jsonl').read_text().splitlines()]
 gs=[r for r in rows if r['phase']=='student_done'];fs=[r for r in rows if r['phase']=='fake_D']
 assert len(gs)==4 and len(fs)==20
 assert [r['stage'] for r in gs]==[2,3,4,2]
 assert [r['fake_stage'] for r in fs]==[1,2,3,4]*5
 assert all(r['D_fake_stage']==4 for r in fs)
 assert [r['g'] for r in gs if r['gan']]==[3]
 assert [r['fake_update'] for r in fs if r['R1_active']]==[16]
 for r in rows:
  assert len(r['original_indices_rank'])==4
  for k in numeric:
   if k in r:assert math.isfinite(r[k]),(rank,k,r)
 allrows[str(rank)]=rows
for a,b in zip(allrows['0'],allrows['1']):
 assert a['phase']==b['phase']
 assert len(set(a['original_indices_rank']+b['original_indices_rank']))==8
monitor=[json.loads(x) for x in (root/'resource_monitor.jsonl').read_text().splitlines()]
assert all(not r['foreign'] for r in monitor)
worker_pids=[int((p/f'pid_rank{r}').read_text()) for r in (0,1)]
assert all(not Path(f'/proc/{pid}').exists() for pid in worker_pids),worker_pids
peaks={}
for rec in monitor:
 for line in rec['gpus'].splitlines():
  index,used,free,util=[int(x.strip()) for x in line.split(',')]
  item=peaks.setdefault(str(index),{'max_used_MiB':0,'min_free_MiB':100000,'max_util_percent':0})
  item['max_used_MiB']=max(item['max_used_MiB'],used)
  item['min_free_MiB']=min(item['min_free_MiB'],free)
  item['max_util_percent']=max(item['max_util_percent'],util)
live=subprocess.check_output(['nvidia-smi','-i','0,2','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv'],text=True)
result={'guard':guard,'config':config,'status':status,'rows':allrows,
        'mean_timed_cycle_s':statistics.mean(status['cycle_seconds'][1:]),
        'median_timed_cycle_s':statistics.median(status['cycle_seconds'][1:]),
        'device_monitor_peaks':peaks,'foreign_processes_detected':False,
        'own_worker_pids_exited':worker_pids,'compute_apps_after':live,
        'source_sha256':json.loads((p/'source_sha256.json').read_text()),
        'limits':'batch8, 4G/20fake-D, 3 timed cycles; no evaluation images or checkpoints; not batch64 throughput or quality evidence'}
(root/'results.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ['rows','config','source_sha256','guard']},indent=2))
PY
du -sh outputs/gpu_smoke_20261023
