from a50k_stage_assets_n23 import remote,ENV,RCLONE
from pathlib import Path
import hashlib,subprocess
root='/data/shenglin.zeng/a50k_20261123/eval_A_milestones_20261125'
old=Path('output/sdxl-crossdistill/new_models_eval_20261114/code')
local=Path('output/sdxl-crossdistill/a50k_train60_20261124/coco_milestones');local.mkdir(exist_ok=True)
worker=(old/'worker.py').read_text()
a=worker.index('def guard():');b=worker.index('def unit(',a)
adapter="""def load_student(pipe,arm,manifest):
 ck=Path(manifest['models'][arm]['checkpoint'])
 shards=[torch.load(ck/f'student_rank{i}.pt',map_location='cpu',mmap=True,weights_only=True) for i in range(4)]
 targets=pipe.unet.state_dict()
 assert all(s.keys()==targets.keys() for s in shards)
 for k,target in targets.items():
  pieces=[s[k] for s in shards]
  if pieces[0].shape==target.shape and all(t.shape==target.shape for t in pieces):
   assert all(torch.equal(t,pieces[0]) for t in pieces[1:]),k
   value=pieces[0]
  else:
   value=torch.cat(pieces,0)
  assert value.shape==target.shape and torch.isfinite(value).all(),k
  target.copy_(value)
 print('STRICT_STUDENT_LOADED',arm,len(targets),flush=True)
core.load_student=load_student
"""
worker=worker[:a]+adapter+worker[b:]
worker=worker.replace(" assert core.sha(ck/'student_rank0.pt')==record['shards'][0]['sha256']"," assert (ck/'complete.json').is_file() and record['world_size']==4")
worker=worker.replace('guard();images,info=','images,info=').replace(' guard();core.worker(args)',' core.worker(args)')
(local/'worker.py').write_text(worker,encoding='utf8')
remote('mkdir -p '+root+'/code')
for src,name in [(local/'worker.py','worker.py'),(Path('tmp/a50k_eval_four_n25.py'),'run_milestones.py'),(old/'evaluation_core.py','evaluation_core.py'),(old/'objectives.py','objectives.py')]:
 target=root+'/code/'+name
 subprocess.run([RCLONE,'copyto',str(src),'dest:'+target],env=ENV,check=True)
 h=hashlib.sha256(src.read_bytes()).hexdigest()
 assert remote('sha256sum '+target).split()[0]==h
 with (local/'部署记录.txt').open('a',encoding='utf8') as f:f.write(f'{src.resolve()} -> 172.18.32.81:{target} SHA256 {h}\n')
print(remote('cd '+root+' && (nohup /data3/shenglin.zeng/sdxl_turning_ablation_20261028/env/bin/python -u code/run_milestones.py >> queue.log 2>&1 </dev/null & echo PID=$!)'))
