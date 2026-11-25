import json,csv,statistics,math,subprocess
from pathlib import Path
root=Path('/data5/shenglin.zeng')
folder=root/'outputs/sparkdiffusion/CrossDistill_Smoke/wan1.3B_dense_pcm_dmd_smoke'
records=[]
for p in folder.glob('metrics_rank*.jsonl'):
 records.extend(json.loads(x) for x in p.read_text().splitlines() if x.strip())
assert len(records)==40,len(records)
assert all(math.isfinite(r['loss']) for r in records)
per_iteration=[]
for i in range(1,11):
 rows=[r for r in records if r['iteration']==i]
 assert sorted(r['rank'] for r in rows)==[0,1,2,3],(i,rows)
 per_iteration.append({'iteration':i,'phase':rows[0]['phase'],'seconds':max(r['seconds'] for r in rows),'peak_allocated_GiB':max(r['peak_allocated_GiB'] for r in rows),'peak_reserved_GiB':max(r['peak_reserved_GiB'] for r in rows),'parameter_delta_max':max(r['parameter_delta_max'] for r in rows)})
phases={}
for phase in ['student','critic']:
 rows=[r for r in per_iteration if r['phase']==phase]
 assert any(r['parameter_delta_max']>1e-10 for r in rows),phase
 phases[phase]={'updates':len(rows),'mean_seconds':statistics.mean(r['seconds'] for r in rows),'min_seconds':min(r['seconds'] for r in rows),'max_seconds':max(r['seconds'] for r in rows),'max_allocated_GiB':max(r['peak_allocated_GiB'] for r in rows)}
assert phases['student']['updates']==2 and phases['critic']['updates']==8
gpu_peak={}
with (root/'logs/training_gpu_samples.csv').open() as f:
 for row in csv.reader(f):
  if len(row)<3:continue
  try:gpu=int(row[1]);used=float(row[2].strip().split()[0])
  except ValueError:continue
  gpu_peak[gpu]=max(gpu_peak.get(gpu,0),used/1024)
summary={'server':'172.18.33.60','root':str(root),'source_commit':subprocess.check_output(['git','-C',str(root/'src/SparkDiffusion'),'rev-parse','HEAD'],text=True).strip(),'model':'Wan2.1-T2V-1.3B dense teacher/student/fake score','unique_samples':4,'teacher_sampling_steps':8,'resolution':[480,832],'frames':81,'gpus':4,'batch_per_gpu':1,'global_batch':4,'iterations':10,'precision':'BF16 compute / FP32 trainable parameters and optimizer states','optimizer':'PyTorch AdamW','ema':False,'gradient_accumulation':1,'warmup_steps':2,'nccl_p2p_disabled':True,'nccl_ib_disabled':True,'all_losses_finite':True,'student_and_critic_updated':True,'phases':phases,'training_compute_seconds':sum(r['seconds'] for r in per_iteration),'max_allocated_GiB':max(r['peak_allocated_GiB'] for r in records),'max_reserved_GiB':max(r['peak_reserved_GiB'] for r in records),'nvidia_smi_peak_GiB_per_gpu':gpu_peak,'disk_direct_sequential_MB_s':{'write':55.8,'read':60.1,'test_bytes':2147483648},'nccl':json.loads((root/'benchmarks/nccl.json').read_text()),'iterations_detail':per_iteration,'limitations':['4 unique teacher-generated samples repeated in shards; not a quality/convergence evaluation','Teacher samples use 8 steps; not the full paper dataset','Global batch 4 vs paper global batch 16; EMA disabled and short warmup','Only 2 student updates, so student timing is preliminary']}
exit_file=root/'logs/smoke_pipeline.exit'
summary['process_exit_code']=int(exit_file.read_text().strip()) if exit_file.exists() else None
summary['checkpoint_status']='complete' if summary['process_exit_code']==0 else 'saving'
summary['environment_path']=str(root/'envs/crossdistill')
summary['checkpoint_directory']=str(folder/'checkpoints/iter_000000010')
summary['exported_student_path']=str(folder/'checkpoints/iter_000000010/exported.pth')
summary['checkpoint_save_and_export_seconds']=469
from importlib.metadata import version
summary['package_versions']={name:version(name) for name in ['torch','torchvision','triton','transformers','diffusers']}
(root/'outputs/smoke_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
