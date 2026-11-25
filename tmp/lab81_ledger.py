import json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'.image_runtime_pkgs'))
import paramiko
raw=Path('C:/Users/sherl/Desktop/login.txt').read_text(encoding='utf-8-sig')
creds={ip:(u,p.strip()) for ip,u,p in re.findall(r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)}
host='172.18.32.81';u,p=creds[host]
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(host,username=u,password=p,timeout=15,look_for_keys=False,allow_agent=False)
remote='/data3/shenglin.zeng/sdxl_turning_ablation_20261028'
from datetime import datetime
j={}
with c.open_sftp() as ftp:
 for arm in ['normalized_u','raw_angle']:
  groups=[]
  for phase in ['validation','g160']:
   d=remote+'/outputs/turning_batch64_g160_4gpu_teacher_shard_20261029/'+arm+'_'+phase
   with ftp.open(d+'/rank0.jsonl') as f:
    gs=[r for l in f if (r:=json.loads(l)).get('phase')=='student_done']
   if gs:groups.append(gs)
  j[arm]={'g':groups[-1][-1]['g'],'hours':sum((datetime.strptime(z[-1]['timestamp'],'%Y%m%dT%H%M%S%z')-datetime.strptime(z[0]['timestamp'],'%Y%m%dT%H%M%S%z')).total_seconds()+z[0]['cycle_seconds'] for z in groups)/3600}
print(j)
Path('output/sdxl-crossdistill/all_experiments_20261104/lab81.json').write_text(json.dumps(j))
