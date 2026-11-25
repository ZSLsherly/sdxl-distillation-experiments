"""Copy the authorized lab checkpoint directly to the evaluation node; no local weight."""
from pathlib import Path
import sys,re,json,posixpath,shlex
sys.path.insert(0,str(Path('.image_runtime_pkgs').resolve()))
import paramiko
HERE=Path('output/sdxl-crossdistill/high30_parallel_eval_20261114')
ROOT='/data1/shenglin.zeng/high30_parallel_eval_20261114'
PY='/data1/shenglin.zeng/sdxl_eval_uniform_20261025/env/bin/python'
SOURCE='/data5/shenglin.zeng/checkpoints/a800_archive_20261114/outputs/A_clean_global_high30_g160_20261114_v2/model/weights/20261114T172912+0800_g000160_end/student_rank0.pt'
raw=Path('C:/Users/sherl/Desktop/login.txt').read_text(encoding='utf-8-sig')
creds={ip:(u,p.strip()) for ip,u,p in re.findall(r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)}
def connect(host):
 c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());u,p=creds[host]
 c.connect(host,username=u,password=p,timeout=20,look_for_keys=False,allow_agent=False);return c
def run(c,cmd,data=None):
 i,o,e=c.exec_command(cmd);o.channel.set_combine_stderr(True)
 if data is not None:i.write(data);i.flush()
 i.channel.shutdown_write()
 for line in o:print(line.rstrip(),flush=True)
 rc=o.channel.recv_exit_status()
 if rc:raise RuntimeError(f'remote exit {rc}')
gpus=['GPU-dee80cca-d29a-9086-d006-b76ba6ee093e','GPU-7328ece9-841e-484d-9e72-d5b762fdce95','GPU-c374184a-ab5a-681a-ed48-025bed2da39b','GPU-4fba4d7f-66e5-aa77-7210-3803fe7190df','GPU-baca962f-0eff-b198-1677-8c28d525ecc1','GPU-df453630-a9d5-be2e-43db-e0abc1834da2']
node=dict(root=ROOT,python=PY,gpus=gpus,physical_cards=[1,2,3,5,6,7])
(HERE/'node.json').write_text(json.dumps(node,indent=2))
for p in (HERE/'code').glob('*.py'):compile(p.read_text(),str(p),'exec')
assert (HERE/'code/evaluation_core.py').read_bytes()==Path('output/sdxl-crossdistill/new_models_eval_20261114/code/evaluation_core.py').read_bytes()
with connect('172.18.132.41') as dest:
 run(dest,'mkdir -p '+ROOT+'/code '+ROOT+'/checkpoint')
 with dest.open_sftp() as s:
  for p in list((HERE/'code').glob('*.py'))+[HERE/'selection.json',HERE/'training_effective_config.json',HERE/'node.json']:
   s.put(str(p),ROOT+'/'+p.relative_to(HERE).as_posix())
with connect('172.18.33.60') as source:
 script='''import sys,json,os,subprocess
a=json.loads(sys.stdin.readline());rc='/data5/shenglin.zeng/checkpoints/a800_archive_20261114/.tools/rclone'
env=dict(os.environ,RCLONE_CONFIG='/dev/null',RCLONE_CONFIG_H41_TYPE='sftp',RCLONE_CONFIG_H41_HOST='172.18.132.41',RCLONE_CONFIG_H41_USER=a['user'],RCLONE_CONFIG_H41_PORT='22')
env['RCLONE_CONFIG_H41_PASS']=subprocess.check_output([rc,'obscure','-'],input=a['password']+'\\n',text=True).strip()
subprocess.run([rc,'copyto',a['source'],'H41:'+a['destination'],'--retries','3','--stats','20s','--stats-one-line','--log-level','NOTICE'],env=env,check=True)
print('WEIGHT_COPIED',flush=True)
'''
 u,p=creds['172.18.132.41']
 run(source,'python3 -u -c '+shlex.quote(script),json.dumps(dict(user=u,password=p,source=SOURCE,destination=ROOT+'/checkpoint/student_rank0.pt'))+'\n')
with connect('172.18.132.41') as dest:
 run(dest,'env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 '+PY+' -u '+ROOT+'/code/prepare.py '+ROOT)
print('DEPLOYMENT_VERIFIED',flush=True)
