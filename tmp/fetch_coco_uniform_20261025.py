import sys,re,json,time
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'.image_runtime_pkgs'))
import paramiko
raw=Path(r'C:\Users\sherl\Desktop\login.txt').read_text(encoding='utf-8-sig')
u,p=next((u,p.strip()) for ip,u,p in re.findall(r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw) if ip=='172.18.132.41')
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect('172.18.132.41',username=u,password=p,timeout=20,look_for_keys=False,allow_agent=False)
base='/data1/shenglin.zeng/sdxl_eval_uniform_20261025';remote=base+'/outputs/coco10k'
out=Path('output/sdxl-crossdistill/coco_uniform_g160_20261025');status={'checked_unix':time.time(),'workers':{}}
with c.open_sftp() as s:
 names=s.listdir(remote)
 for name in names:
  if name.startswith('status_') and name.endswith('.json'):
   with s.open(remote+'/'+name) as f:status['workers'][name]=json.loads(f.read())
  if name in ('manifest.json','protocol_verification.json','scores.json','complete.json','combined_scores.json','guard.jsonl','metrics.log','resume_inventory_20261025.json'):
   s.get(remote+'/'+name,str(out/name))
 if (out/'guard.jsonl').exists():status['guard_events']=[json.loads(line) for line in (out/'guard.jsonl').read_text().splitlines()][-8:]
 status['worker_done_files']=[name for name in names if name.startswith('worker_') and name.endswith('_done.json')]
 s.get(base+'/environment_versions.json',str(out/'environment_versions.json'))
 if '--images' in sys.argv:
  folder=remote+'/images/joint4_uniform'
  try:images=sorted(s.listdir(folder))[:4]
  except FileNotFoundError:images=[]
  (out/'images/joint4_uniform').mkdir(parents=True,exist_ok=True)
  for name in images:s.get(folder+'/'+name,str(out/'images/joint4_uniform'/name))
c.close()
if (out/'resume_inventory_20261025.json').exists():
 inv=json.loads((out/'resume_inventory_20261025.json').read_text())
 for w in status['workers'].values():
  starts=[e['time'] for e in status.get('guard_events',[]) if e['event']=='worker_started' and e['rank']==w['rank']]
  fresh=bool(starts and w['updated_unix']>=max(starts) and max(starts)>=inv['checked_unix'])
  w['reported_images_per_second']=w['images_per_second']
  w['new_images_this_resume']=max(0,w['done']-inv['reused_by_rank'][w['rank']]) if fresh else 0
  w['images_per_second']=w['new_images_this_resume']/w['seconds'] if fresh else 0
(out/'remote_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8');print(json.dumps(status,indent=2))
