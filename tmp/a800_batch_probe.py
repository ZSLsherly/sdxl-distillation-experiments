import os,sys,json,time,subprocess,statistics,math
from pathlib import Path
ROOT=Path('/root/sdxl_research')
sys.path.insert(0,str(ROOT/'code/sdxl_research'))
from launch import command,gpu_check
from turning_batch64_workflow import descendants,foreign_on,stop_own
OUT=ROOT/'outputs/a800_B_batch_probe_20261101'
OUT.mkdir(parents=True,exist_ok=False)
base=json.loads((ROOT/'code/sdxl_research/configs/turning_B_global_a800_probe.json').read_text())
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TORCHDYNAMO_DISABLE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
results=[];deadline=time.monotonic()+55*60
for mb in (64,32,16,8,4,2,1):
 for retry in range(6):
  try:cards=gpu_check([0],76);break
  except RuntimeError:
   if retry==5:raise
   time.sleep(5)
 uuids={c[1] for c in cards}
 config=dict(base,micro_batch=mb,rollout_batch=min(mb,8))
 run=f'a800_B_batch_probe_20261101/mb{mb}'
 logpath=OUT/f'mb{mb}.log'
 (OUT/f'mb{mb}_requested.json').write_text(json.dumps(config,indent=2))
 print(json.dumps({'event':'starting','micro_batch':mb,'accumulation':64//mb,'rollout_batch':config['rollout_batch']}),flush=True)
 with logpath.open('x') as log:
  p=subprocess.Popen(command(config,ROOT,run,1),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   while p.poll() is None:
    if foreign_on(uuids,descendants(p.pid)):raise RuntimeError('Foreign GPU task detected')
    if time.monotonic()>deadline:raise RuntimeError('55 minute probe budget reached')
    if (OUT/'STOP').exists():raise RuntimeError('User stop')
    time.sleep(3)
  except BaseException:
   stop_own(p);raise
 text=logpath.read_text(errors='replace')
 row={'micro_batch':mb,'accumulation':64//mb,'rollout_batch':config['rollout_batch'],'returncode':p.returncode}
 if p.returncode:
  row['status']='oom' if ('CUDA out of memory' in text or 'torch.OutOfMemoryError' in text) else 'failed'
  row['tail']=text[-4000:]
 else:
  status=json.loads((ROOT/'outputs'/run/'status.json').read_text())
  entries=[json.loads(s) for s in (ROOT/'outputs'/run/'rank0.jsonl').read_text().splitlines()]
  gs=[s for s in entries if s.get('phase')=='student_done'];fs=[s for s in entries if s.get('phase')=='fake']
  assert len(gs)==4 and len(fs)==8 and status['effective_batch']==64
  assert all(math.isfinite(s[k]) for s in gs for k in ('grad_norm','pcm_loss','middle_dm_loss','terminal_dm_loss'))
  assert all(math.isfinite(s[k]) for s in fs for k in ('fake_grad_norm','fake_loss'))
  row.update(status='passed',result=status,stable_cycle_seconds=statistics.mean(s['cycle_seconds'] for s in gs[1:]),middle_stages=sorted({s['stage'] for s in gs}))
 results.append(row);(OUT/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(row),flush=True)
 if row['status']=='passed':break
 if row['status']!='oom':raise RuntimeError('Non-OOM failure, do not blindly lower batch')
(OUT/'finished.json').write_text(json.dumps({'results':results,'training_started':'benchmark updates only; no formal G160 run','selected_micro_batch':results[-1]['micro_batch'] if results[-1]['status']=='passed' else None},indent=2))
