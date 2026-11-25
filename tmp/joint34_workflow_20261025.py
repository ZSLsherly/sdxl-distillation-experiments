"""One matched G160 training run followed by one fixed COCO10K evaluation."""
import os,sys,json,time,signal,subprocess,shutil,hashlib
from pathlib import Path
ROOT=Path('/data1/shenglin.zeng/sdxl_eval_uniform_20261025')
PREFIX='joint34_uniform_10k_2gpu_20261025'
OUT=ROOT/'outputs'/PREFIX
SOURCE=OUT/'code'
shutil.copytree(ROOT/'training_code',SOURCE,ignore=shutil.ignore_patterns('__pycache__'))
sys.path.insert(0,str(SOURCE))
from launch import gpu_check,command
GPUS=[2,3]
PY=ROOT/'env/bin/python'
env=os.environ.copy()
env.update(NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',TORCHDYNAMO_DISABLE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=str(ROOT/'cache'),TOKENIZERS_PARALLELISM='false',LD_LIBRARY_PATH=str(ROOT/'runtime/lib')+':'+env.get('LD_LIBRARY_PATH',''))
for key,folder in [('TMPDIR','tmp'),('TORCHINDUCTOR_CACHE_DIR','cache/inductor'),('TRITON_CACHE_DIR','cache/triton')]:
 path=ROOT/folder;path.mkdir(parents=True,exist_ok=True);env[key]=str(path)
def save(name,obj):
 (OUT/name).write_text(json.dumps(obj,indent=2))
def event(kind,**data):
 row={'time':time.time(),'event':kind,**data}
 with (OUT/'workflow.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 print(json.dumps(row),flush=True)
def descendants(parent):
 children={}
 for p in Path('/proc').glob('[0-9]*/stat'):
  try:row=p.read_text().rsplit(')',1)[1].split();children.setdefault(int(row[1]),[]).append(int(p.parent.name))
  except (OSError,ValueError):pass
 found={parent};todo=[parent]
 while todo:
  for pid in children.get(todo.pop(),[]):
   if pid not in found:found.add(pid);todo.append(pid)
 return found
def guarded(label,commands,individual=False):
 cards=gpu_check(GPUS,36);uuids=[x[1] for x in cards]
 processes=[];logs=[];begin=time.time()
 try:
  for i,cmd in enumerate(commands):
   log=(OUT/f'{label}_{i}.log').open('a');logs.append(log)
   visible=uuids[i] if individual else ','.join(uuids)
   p=subprocess.Popen(cmd,cwd=ROOT,env={**env,'CUDA_VISIBLE_DEVICES':visible},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   processes.append(p);event('process_started',phase=label,pid=p.pid,gpus=visible,command=cmd)
  while any(p.poll() is None for p in processes):
   owned=set().union(*(descendants(p.pid) for p in processes if p.poll() is None))
   active_uuids={uuids[i] for i,p in enumerate(processes) if p.poll() is None} if individual else set(uuids)
   apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=8)
   foreign=[]
   for line in apps.splitlines():
    row=[x.strip() for x in line.split(',')]
    if len(row)>=2 and row[0] in active_uuids and int(row[1]) not in owned:foreign.append({'gpu':row[0],'pid':int(row[1])})
   with (OUT/'resource_monitor.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'phase':label,'own_pids':sorted(owned),'foreign':foreign,'apps':apps})+'\n')
   if foreign:raise RuntimeError('Other task entered selected GPU: '+str(foreign))
   if any(p.poll() not in (None,0) for p in processes):raise RuntimeError('Child failed; inspect '+label+' logs')
   if time.time()-begin>50*3600:raise RuntimeError('50-hour phase timeout')
   time.sleep(10)
  if any(p.returncode for p in processes):raise RuntimeError('Child exit failure: '+label)
  event('phase_complete',phase=label,seconds=time.time()-begin)
 except BaseException:
  owned=set().union(*(descendants(p.pid) for p in processes if p.poll() is None)) if processes else set()
  for pid in sorted(owned,reverse=True):
   try:os.kill(pid,signal.SIGTERM)
   except ProcessLookupError:pass
  for p in processes:
   try:p.wait(timeout=20)
   except subprocess.TimeoutExpired:
    for pid in descendants(p.pid):
     try:os.kill(pid,signal.SIGKILL)
     except ProcessLookupError:pass
    p.wait()
  raise
 finally:
  for log in logs:log.close()
try:
 proof=json.loads((ROOT/'joint34_reference_provenance.json').read_text())
 for name,h in proof['source_sha256'].items():assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==h,name
 assert not (OUT/'phase1').exists() and not (OUT/'train').exists()
 cfg=json.loads((OUT/'phase1_config.json').read_text())
 guarded('phase1',[command(cfg,ROOT,PREFIX+'/phase1',2)])
 first=json.loads((OUT/'phase1/status.json').read_text());assert first['status']=='complete' and first['student_updates']==2
 intermediate=Path(first['training_state']);assert json.loads((intermediate/'complete.json').read_text())['progress']['gcount']==2
 cfg=json.loads((OUT/'train_config.json').read_text());cfg.update(resume=str(intermediate),resume_repack=True)
 save('train_config.json',cfg)
 guarded('train',[command(cfg,ROOT,PREFIX+'/train',2)])
 status=json.loads((OUT/'train/status.json').read_text())
 assert status['status']=='complete' and status['student_updates']==160 and status['fake_updates']==800 and status['D_updates']==800,status
 finalstate=Path(status['training_state']);assert json.loads((finalstate/'complete.json').read_text())['progress']['gcount']==160
 weights=[p.parent for p in (OUT/'train/weights').glob('*/complete.json') if json.loads(p.read_text())['student_updates']==160]
 assert len(weights)==1,weights
 checkpoint=weights[0];evaluation=OUT/'coco10k'
 save('models.json',{'joint34_uniform':{'checkpoint':str(checkpoint)}})
 cpu_env={**env,'CUDA_VISIBLE_DEVICES':''}
 with (OUT/'eval_prepare.log').open('w') as log:
  subprocess.run([str(PY),str(SOURCE/'evaluate.py'),'prepare','--root',str(ROOT),'--out',str(evaluation),'--models',str(OUT/'models.json'),'--n','10000'],env=cpu_env,stdout=log,stderr=subprocess.STDOUT,check=True)
 ref=json.loads((ROOT/'outputs/coco10k/manifest.json').read_text());manifest=json.loads((evaluation/'manifest.json').read_text())
 for key in ['n','source_zip_sha256','csv_sha256']:assert ref[key]==manifest[key],key
 fields=('index','prompt','real_file','seed')
 assert [tuple(r[k] for k in fields) for r in ref['samples']]==[tuple(r[k] for k in fields) for r in manifest['samples']]
 manifest['visual_indices']=ref['visual_indices'];(evaluation/'manifest.json').write_text(json.dumps(manifest,indent=2))
 (evaluation/'features').mkdir(exist_ok=True);(evaluation/'features/real').symlink_to(ROOT/'reference_features/real',target_is_directory=True)
 event('evaluation_protocol_verified',samples=10000,checkpoint=str(checkpoint))
 cmds=[[str(PY),'-u',str(SOURCE/'evaluate.py'),'worker','--root',str(ROOT),'--out',str(evaluation),'--methods','joint34_uniform','--rank',str(rank),'--world','2','--batch','4'] for rank in range(2)]
 guarded('evaluation',cmds,individual=True)
 with (OUT/'metrics.log').open('w') as log:
  subprocess.run(['nice','-n','19',str(PY),'-u',str(SOURCE/'evaluate.py'),'metrics','--root',str(ROOT),'--out',str(evaluation)],env=cpu_env,stdout=log,stderr=subprocess.STDOUT,check=True)
 scores=json.loads((ROOT/'outputs/coco10k/combined_scores.json').read_text());scores.update(json.loads((evaluation/'scores.json').read_text()))
 save('combined_scores.json',scores)
 # Only discard this run's temporary G2 handoff after the final state and evaluation succeed.
 assert intermediate.resolve().is_relative_to((OUT/'phase1/training_state').resolve())
 shutil.rmtree(intermediate)
 save('workflow_complete.json',{'time':time.time(),'training_state':str(finalstate),'checkpoint':str(checkpoint),'scores':scores,'temporary_G2_state_removed':True})
 event('complete',scores=scores)
except BaseException as exc:
 save('workflow_stopped.json',{'time':time.time(),'reason':str(exc)})
 event('stopped',reason=str(exc))
 raise
