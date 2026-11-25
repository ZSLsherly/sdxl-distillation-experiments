"""Read-only GPU check on only the hosts in the user's supplied login file."""
import sys,re,json,csv,io,concurrent.futures
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'.image_runtime_pkgs'))
import paramiko
raw=Path(r'C:/Users/sherl/Desktop/login.txt').read_text(encoding='utf-8-sig')
creds={ip:(u,p.strip()) for ip,u,p in re.findall(r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)}
def check(ip):
 c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
 try:
  u,p=creds[ip];c.connect(ip,username=u,password=p,timeout=8,auth_timeout=8,banner_timeout=8,look_for_keys=False,allow_agent=False)
  _,out,_=c.exec_command('nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu,uuid --format=csv,noheader,nounits',timeout=15)
  rows=list(csv.reader(io.StringIO(out.read().decode())))
  _,out,_=c.exec_command('nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits',timeout=15)
  apps=list(csv.reader(io.StringIO(out.read().decode())))
  cards=[{'index':int(r[0]),'name':r[1].strip(),'total_GiB':round(int(r[2])/1024,1),'free_GiB':round(int(r[3])/1024,1),
          'util':r[4].strip(),'processes':sum(a[0].strip()==r[5].strip() for a in apps if len(a)>1)} for r in rows if len(r)==6]
  return {'host':ip,'cards':cards}
 except Exception as e:return {'host':ip,'error':type(e).__name__+': '+str(e)[:140]}
 finally:c.close()
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for r in pool.map(check,creds):results.append(r);print(json.dumps(r),flush=True)
Path('output/sdxl-crossdistill/semantic_old_uniform_20261104/gpu_inventory.json').write_text(json.dumps(results,indent=2),encoding='utf-8')

