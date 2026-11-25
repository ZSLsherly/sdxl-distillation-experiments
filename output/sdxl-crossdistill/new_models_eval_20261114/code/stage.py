"""Runs on the archive server; all file transfers use rclone, credentials stay in memory."""
import os,sys,json,hashlib,subprocess,shlex,time,shutil
from pathlib import Path
sys.path.insert(0,'/data5/shenglin.zeng/tools/eval_transfer_20261025')
import paramiko
AUTH=json.loads(sys.stdin.readline());HERE=Path(__file__).resolve().parent.parent
RC='/data5/shenglin.zeng/checkpoints/a800_archive_20261114/.tools/rclone'
pid=os.fork()
if pid:print('STAGING_PID',pid,flush=True);sys.exit(0)
os.setsid()
log=(HERE/'stage.log').open('a',buffering=1);os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
with open('/dev/null') as f:os.dup2(f.fileno(),0)
env=dict(os.environ,RCLONE_CONFIG='/dev/null')
for host,(user,password) in AUTH.items():
 key='H'+host.rsplit('.',1)[1];prefix='RCLONE_CONFIG_'+key+'_'
 env.update({prefix+'TYPE':'sftp',prefix+'HOST':host,prefix+'USER':user,prefix+'PORT':'22',prefix+'PASS':subprocess.check_output([RC,'obscure','-'],input=password+'\n',text=True).strip()})
def transfer(src,dest,folder=False):
 print('RCLONE',src,dest,flush=True)
 subprocess.run([RC,'copy' if folder else 'copyto',str(src),str(dest),'--transfers','3','--checkers','4','--retries','3','--stats','30s','--stats-one-line','--log-level','NOTICE'],env=env,check=True)
def connect(host):
 c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect(host,username=AUTH[host][0],password=AUTH[host][1],timeout=20,look_for_keys=False,allow_agent=False);return c
def execute(c,cmd):
 _,o,_=c.exec_command(cmd,timeout=600);o.channel.set_combine_stderr(True);s=o.read().decode();rc=o.channel.recv_exit_status()
 if rc:raise RuntimeError(s[-2000:])
 return s
def jsonwrite(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def dirsha(c,path):
 s=execute(c,'find '+shlex.quote(path)+' -type f -exec sha256sum {} +')
 return {line.split(maxsplit=1)[1].strip().removeprefix(path+'/'):line.split()[0] for line in s.splitlines()}
jobs=json.loads((HERE/'jobs.json').read_text());nodes=json.loads((HERE/'nodes.json').read_text())
transfer('H78:/data1/shenglin.zeng/coco_missing_20261111/reference/manifest.json',HERE/'reference.json')
reference=json.loads((HERE/'reference.json').read_text())
assert reference['n']==len(reference['samples'])==10000
for host,node in nodes.items():
 root=node['root'];remote='H'+host.rsplit('.',1)[1]+':';bundle=HERE/'bundles'/host
 bundle.mkdir(parents=True,exist_ok=True)
 shutil.copytree(HERE/'code',bundle/'code',dirs_exist_ok=True)
 shutil.copy2(HERE/'selection.json',bundle/'selection.json');jsonwrite(bundle/'node.json',node)
 for arm in node['arms']:
  job=jobs[arm];ck=root+'/checkpoints/'+arm
  shutil.copytree(HERE/'metadata'/arm,bundle/'checkpoints'/arm,dirs_exist_ok=True)
  rec=dict(config=job['config'],checkpoint=ck,world_size=1,student_updates=160,shards=[dict(name='student_rank0.pt',sha256=job['sha256'])])
  jsonwrite(bundle/'results'/arm/'manifest.json',dict(reference,models={arm:rec}))
 transfer(bundle,remote+root,True)
 with connect(host) as c:
  for p in bundle.rglob('*'):
   if p.is_file():
    digest=hashlib.sha256(p.read_bytes()).hexdigest();dest=root+'/'+str(p.relative_to(bundle))
    assert execute(c,'sha256sum '+shlex.quote(dest)).split()[0]==digest
  print(execute(c,shlex.quote(node['python'])+' '+shlex.quote(root+'/code/prepare_node.py')+' '+shlex.quote(root)),flush=True)
 # Missing public assets; base models and existing real features are linked locally.
 if host.endswith('.81'):
  transfer('H78:/data1/shenglin.zeng/coco_missing_20261111/reference/features/real',remote+root+'/reference/features/real',True)
  transfer('H78:/data1/shenglin.zeng/coco_missing_20261111/models/metrics/clean_fid/inception-2015-12-05.pt',remote+root+'/models/metrics/clean_fid/inception-2015-12-05.pt')
 else:
  transfer('H81:/data3/shenglin.zeng/clip_missing_20261113/models/clip',remote+root+'/models/clip',True)
 # Verify newly transferred assets with SHA-256 at their source and destination.
 records=[]
 with connect(host) as c:
  otherhost='172.18.32.78' if host.endswith('.81') else '172.18.32.81'
  assetpairs=[('/data1/shenglin.zeng/coco_missing_20261111/reference/features/real',root+'/reference/features/real'),('/data1/shenglin.zeng/coco_missing_20261111/models/metrics/clean_fid',root+'/models/metrics/clean_fid')] if host.endswith('.81') else [('/data3/shenglin.zeng/clip_missing_20261113/models/clip',root+'/models/clip')]
  with connect(otherhost) as other:
   for source,destination in assetpairs:
    a=dirsha(other,source);b=dirsha(c,destination);assert a==b,(source,destination)
    records.append(dict(source_server=otherhost,source=source,destination_server=host,destination=destination,sha256_files=a,verified=True))
  for arm in node['arms']:
   job=jobs[arm];src=Path(job['source'])
   with src.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==job['sha256']
   dest=root+'/checkpoints/'+arm+'/student_rank0.pt'
   transfer(src,remote+dest)
   assert execute(c,'sha256sum '+shlex.quote(dest)).split()[0]==job['sha256']
   records.append(dict(source_server='172.18.33.60',source=str(src),destination_server=host,destination=dest,sha256=job['sha256'],verified=True))
   marker=HERE/'markers'/arm/'ready.json';jsonwrite(marker,records[-1]);transfer(marker,remote+root+'/results/'+arm+'/ready.json')
  # The standard evaluator will check the exact 10K real feature indices and finiteness.
  marker=HERE/'markers'/host/'assets_ready.json';jsonwrite(marker,dict(ready=True));transfer(marker,remote+root+'/assets_ready.json')
  txt=HERE/'markers'/host/'传输记录.txt';txt.write_text(json.dumps(records,ensure_ascii=False,indent=2));transfer(txt,remote+root+'/传输记录.txt')
  print(execute(c,'cd '+shlex.quote(root)+' && (nohup env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 '+shlex.quote(node['python'])+' -u code/run_queue.py '+shlex.quote(root)+' >queue.log 2>&1 </dev/null & echo $! >queue.pid)'),flush=True)
  print('NODE_QUEUE_STARTED',host,node['arms'],flush=True)
jsonwrite(HERE/'staging_complete.json',dict(complete=True));print('STAGING_COMPLETE',flush=True)
