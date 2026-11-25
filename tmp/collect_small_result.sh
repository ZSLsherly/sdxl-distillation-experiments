set -euo pipefail
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import json,math,statistics
from pathlib import Path
base=Path('outputs/research_small_20261022')
names=['legacy_b8','legacy_offload_b8','block_offload_b8','bucket_offload_b8']
data={}
keys=['fake_loss','D_hinge','R1_raw','D_real','D_fake','fake_grad_norm','D_grad_norm','dmd_loss','gan_loss','pcm_loss','grad_norm']
for name in names:
 p=base/name
 cfg=json.loads((p/'config.json').read_text())
 done=(p/'benchmark.json').exists()
 status=json.loads((p/'benchmark.json').read_text()) if done else {'status':'failed','reason':'CUDA OOM after external PID 1899365 allocated GPU0; our workers exited'}
 if done:assert status['status']=='complete' and status['student_updates']==4 and status['fake_updates']==20 and status['D_updates']==20
 rows=[json.loads(x) for x in (p/'rank0.jsonl').read_text().splitlines()]
 other=[json.loads(x) for x in (p/'rank1.jsonl').read_text().splitlines()]
 if done:
  assert len(rows)==len(other)==24
  assert [r['stage'] for r in rows if r['phase']=='student_done']==[2,3,4,2]
 for r in rows+other:
  for k in keys:
   if k in r:assert math.isfinite(r[k]),(name,k)
 gs=[r for r in rows if r['phase']=='student_done' and r['g']>1]
 data[name]={'config':cfg,'status':status,'rows_rank0':rows,'source_sha256':json.loads((p/'source_sha256.json').read_text()),
  'mean_cycle_seconds':statistics.mean(r['cycle_seconds'] for r in gs) if done else None,
  'median_cycle_seconds':statistics.median(r['cycle_seconds'] for r in gs) if done else None,
  'R1_updates':[r['fake_update'] for r in rows if r.get('R1_active')],
  'GAN_G_updates':[r['g'] for r in rows if r.get('gan')],
  'D_collectives':sorted(set(r['D_collectives'] for r in rows if 'D_collectives' in r))}
def compare(left,right):
 a=data[left]['rows_rank0'];b=data[right]['rows_rank0'];diff={}
 for x,y in zip(a,b):
  assert x['phase']==y['phase'] and x.get('g')==y.get('g') and x.get('fake_update')==y.get('fake_update')
  assert x['original_indices_rank']==y['original_indices_rank']
  for k in keys:
   if k in x:
    r=diff.setdefault(k,{'max_abs':0.,'max_relative':0.,'reference_max_abs':0.})
    r['max_abs']=max(r['max_abs'],abs(x[k]-y[k]))
    r['max_relative']=max(r['max_relative'],abs(x[k]-y[k])/max(abs(x[k]),abs(y[k]),1e-6))
    r['reference_max_abs']=max(r['reference_max_abs'],abs(x[k]))
 return diff
result={'server':'172.18.33.60','physical_gpus':[0,3],'description':'SDXL 1024px, batch8, 2 ranks x micro4, fake:G5:1; first G warmup, next3 measured. One run/config; not batch64 performance or quality evidence.',
 'runs':data,'comparisons':{name:compare('legacy_offload_b8',name) for name in ['legacy_b8','block_offload_b8','bucket_offload_b8']}}
(base/'results.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:{'status':v['status']['status'],'mean_s':v['mean_cycle_seconds'],'allocated':v['status'].get('peak_allocated_GiB'),'reserved':v['status'].get('peak_reserved_GiB'),'R1':v['R1_updates'],'GAN':v['GAN_G_updates'],'D_collectives':v['D_collectives']} for k,v in data.items()},indent=2))
print('METRIC_DIFFERENCES',json.dumps(result['comparisons'],indent=2))
PY
