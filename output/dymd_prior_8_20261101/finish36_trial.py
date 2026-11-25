"""Finish the current saved batch at 36, then one bounded two-by-two comparison."""
import copy,json,os,signal,subprocess,time,traceback,shutil
from pathlib import Path
import imageio_ffmpeg
B=Path('/data1/shenglin.zeng/dymd_prior_8_20261101')
PF=B/'official/pf_wan';TRIAL=B/'two_by_two_trial';TRIAL.mkdir(exist_ok=True)
PLAN=json.loads((B/'sample_plan.json').read_text())
GPUS=[1,4,5,7]
def write(path,obj):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
def status(stage,**extra):
 write(B/'status.json',dict(stage=stage,target_trajectories=36,updated_unix=time.time(),**extra))
def validated_video(path):
 reader=imageio_ffmpeg.read_frames(str(path),output_params=['-threads','1'])
 try:
  meta=next(reader);n=sum(1 for _ in reader)
 finally:reader.close()
 assert n==81 and meta['fps']==16,(path,n,meta['fps'])
def ready36():
 for i in range(36):
  if not (B/'profiles'/f'trajectory_{i:02d}.complete').exists():return False
  if not list((B/'videos').rglob(Path(PLAN[i]['filename']).stem+'.mp4')):return False
 try:
  for i in range(32,36):validated_video(next((B/'videos').rglob(Path(PLAN[i]['filename']).stem+'.mp4')))
 except Exception:return False
 return True
def stop_original():
 pid=int((B/'controller.pid').read_text())
 expected=(B/'controller.py').as_posix().encode()
 assert expected in Path(f'/proc/{pid}/cmdline').read_bytes(),'Unexpected controller'
 lines=subprocess.check_output(['ps','-eo','pid=,ppid='],text=True).splitlines()
 parents={int(p):int(pp) for p,pp in map(str.split,lines)}
 owned={pid}
 while True:
  extra={p for p,pp in parents.items() if pp in owned}-owned
  if not extra:break
  owned|=extra
 commands={p:Path(f'/proc/{p}/cmdline').read_bytes() for p in owned if Path(f'/proc/{p}/cmdline').exists()}
 os.kill(pid,signal.SIGTERM)
 for p in sorted(owned-{pid},reverse=True):
  try:
   if Path(f'/proc/{p}/cmdline').read_bytes()==commands[p]:os.kill(p,signal.SIGTERM)
  except (FileNotFoundError,ProcessLookupError):pass
 def alive(p):
  try:return bool(Path(f'/proc/{p}/cmdline').read_bytes())
  except FileNotFoundError:return False
 deadline=time.monotonic()+30
 while any(alive(p) for p in owned) and time.monotonic()<deadline:time.sleep(1)
 for p,cmd in commands.items():
  try:
   if Path(f'/proc/{p}/cmdline').read_bytes()==cmd:os.kill(p,signal.SIGKILL)
  except (FileNotFoundError,ProcessLookupError):pass
 write(TRIAL/'stopped_original_processes.json',{'controller':pid,'task_descendants':sorted(owned),'reason':'36 complete records and saved videos; user requested stopping here'})
 (B/'controller.pid').write_text(str(os.getpid()))
def idle():
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
 state={int(a):[int(b),int(c)] for a,b,c in (r.split(',') for r in rows)}
 return all(state[g][0]<500 and state[g][1]==0 for g in GPUS)
def durations(root,i):
 return float((root/'profiles'/f'trajectory_{i:02d}.complete').read_text())-(root/'profiles'/f'input_{i:02d}.json').stat().st_mtime
try:
 status('finishing_saved_batch_to_36',gpus=GPUS)
 deadline=time.monotonic()+2*3600
 while not ready36():
  if time.monotonic()>deadline:raise TimeoutError('36-video save boundary was not reached; original run left intact')
  time.sleep(2)
 for i in range(36):
  rows=[json.loads(x) for x in (B/'profiles'/f'trajectory_{i:02d}.jsonl').read_text().splitlines()]
  assert len(rows)==360 and {r['sample'] for r in rows}=={i}
 stop_original()
 status('preparing_two_by_two_trial',completed=36)
 archive=B/'stopped_after36';archive.mkdir(exist_ok=True)
 for p in list((B/'profiles').iterdir()):
  if p.name.startswith(('trajectory_','input_')) and int(p.stem.split('_')[-1])>=36:p.rename(archive/p.name)
 protocol=json.loads((B/'protocol.json').read_text());protocol.update(target_trajectories=36,stop_reason='User reduced scope to about 36 and requested one 2x2 timing trial',two_gpu_trial_samples=[32,33]);write(B/'protocol.json',protocol)
 status('analyzing_36_before_trial',completed=36)
 with (B/'analysis36.log').open('w') as log:
  for n in [32,36]:subprocess.run([str(B/'env/bin/python'),str(B/'analyze36.py'),str(B),str(n)],stdout=log,stderr=subprocess.STDOUT,check=True)
 shutil.copytree(B/'analysis_preview_032',B/'analysis/prefix32',dirs_exist_ok=True)
 subprocess.run([str(B/'env/bin/python'),str(B/'write_report36.py')],check=True)
 status('preparing_two_by_two_trial',completed=36)

 deadline=time.monotonic()+900
 while not idle():
  if time.monotonic()>deadline:raise TimeoutError('Allocated GPU set not fully idle after own processes stopped; no other jobs touched')
  time.sleep(5)
 jobs=[]
 for label,indices,sample,port in [('a',[1,4],32,29641),('b',[5,7],33,29642)]:
  g=TRIAL/f'group_{label}';g.mkdir(exist_ok=True)
  manifest=g/'manifest.txt';manifest.write_text(PLAN[sample]['prompt']+'@@'+PLAN[sample]['filename']+'\n')
  write(g/'sample_plan.json',[PLAN[sample]])
  cfg=json.loads((PF/'configs/generate/prior128.jsonc').read_text())
  cfg['meta_model']['hybrid_gpu_num']=2;cfg['meta_model']['ulysses_size']=2
  cfg['inference']['positive_prompt']=str(manifest);cfg['output_dir']=str(g/'videos');cfg['proj_name']='two_gpu_trial'
  write(g/'config.jsonc',cfg)
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,indices)),PYTHONPATH=str(PF),DYMD_PROFILE_DIR=str(g/'profiles'),DYMD_SAMPLE_PLAN=str(g/'sample_plan.json'),DYMD_TRIAL_DIR=str(g),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',NCCL_SOCKET_IFNAME='enp129s0f0',GLOO_SOCKET_IFNAME='enp129s0f0',NCCL_IB_DISABLE='1',NCCL_DEBUG='WARN',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
  log=(g/'run.log').open('w');started=time.time()
  cmd=[str(B/'env/bin/torchrun'),'--nproc-per-node=2','--nnodes=1',f'--rdzv-endpoint=127.0.0.1:{port}',str(B/'profile_trial.py'),'--config-file',str(g/'config.jsonc')]
  p=subprocess.Popen(cmd,cwd=PF,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  jobs.append({'label':label,'gpus':indices,'sample':sample,'dir':g,'process':p,'log':log,'started':started})
 write(TRIAL/'jobs.json',[{k:(str(v) if isinstance(v,Path) else v) for k,v in j.items() if k not in ['process','log']}|{'pid':j['process'].pid} for j in jobs])
 status('running_two_by_two_trial',completed=36,groups=[[1,4],[5,7]])
 deadline=time.monotonic()+45*60
 while any(j['process'].poll() is None for j in jobs):
  progress={'target':36,'completed':36,'stage':'two_by_two_trial','groups':{}}
  for j in jobs:
   if j['process'].poll() is not None and 'finished' not in j:j['finished']=time.time()
   f=j['dir']/'progress.json'
   if f.exists():
    try:progress['groups'][j['label']]=json.loads(f.read_text())
    except json.JSONDecodeError:pass
  write(B/'progress.json',progress)
  if time.monotonic()>deadline:
   for j in jobs:
    if j['process'].poll() is None:os.killpg(j['process'].pid,signal.SIGTERM)
   break
  time.sleep(10)
 result={'unique_original_trajectories':36,'extra_trial_reruns':2,'samples':[32,33],'groups':[],'note':'Same input/seed as 4-GPU baseline. Sampling span excludes model loading; full process wall time includes loading. Only one video per group; no claim of steady-state speedup.'}
 for j in jobs:
  code=j['process'].wait(timeout=60);j['log'].close();g=j['dir'];i=j['sample']
  entry={'group':j['label'],'gpus':j['gpus'],'sample':i,'exit_code':code,'process_wall_seconds':j.get('finished',time.time())-j['started'],'baseline_four_gpu_sampling_seconds':durations(B,i)}
  entry['peak_memory']=[json.loads(p.read_text()) for p in g.glob('rank*_memory.json')]
  if code==0:
   rows=[json.loads(x) for x in (g/'profiles'/f'trajectory_{i:02d}.jsonl').read_text().splitlines()];assert len(rows)==360
   videos=list((g/'videos').rglob('*.mp4'));assert len(videos)==1;validated_video(videos[0])
   entry.update(sampling_seconds=durations(g,i),sampling_start=(g/'profiles'/f'input_{i:02d}.json').stat().st_mtime,sampling_end=float((g/'profiles'/f'trajectory_{i:02d}.complete').read_text()),video_saved=json.loads((g/'video_saved.json').read_text()))
  else:entry['error_tail']=(g/'run.log').read_text(errors='replace')[-2500:]
  result['groups'].append(entry)
 result['both_succeeded']=all(x['exit_code']==0 for x in result['groups'])
 if result['both_succeeded']:
  serial=sum(x['baseline_four_gpu_sampling_seconds'] for x in result['groups'])
  span=max(x['sampling_end'] for x in result['groups'])-min(x['sampling_start'] for x in result['groups'])
  result.update(four_gpu_serial_sampling_seconds=serial,two_groups_parallel_sampling_span_seconds=span,measured_sampling_throughput_ratio=serial/span)
 write(TRIAL/'benchmark_summary.json',result)
 write(B/'analysis/benchmark_summary.json',result)
 subprocess.run([str(B/'env/bin/python'),str(B/'write_report36.py')],check=True)
 status('complete',trajectories=36,two_gpu_trial_succeeded=result['both_succeeded'])
except Exception as e:
 status('failed',error=str(e));traceback.print_exc();raise
