"""Fetch small evidence files, never model weights, from the selected test host."""
import json, re, sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'.image_runtime_pkgs'))
import paramiko
raw=Path('C:/Users/sherl/Desktop/login.txt').read_text(encoding='utf-8-sig')
creds={ip:(u,p.strip()) for ip,u,p in re.findall(r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)}
host='172.18.132.41';u,p=creds[host]
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(host,username=u,password=p,timeout=12,look_for_keys=False,allow_agent=False)
remote='/data1/shenglin.zeng/sdxl_eval_uniform_20261025/outputs/turning_boundary_no_offload_2gpu_20261027'
local=Path('output/sdxl-crossdistill/turning_ablation_20261027/no_offload_4090')
local.mkdir(parents=True,exist_ok=True)
files=['workflow.log','workflow.jsonl','results.json']
for arm in ('normalized_u','raw_angle'):
 files += [f'{arm}.log']+[f'{arm}/{f}' for f in ('benchmark.json','status.json','config.json','rank0.jsonl','rank1.jsonl','source_sha256.json')]
with c.open_sftp() as s:
 for f in files:
  dest=local/f;dest.parent.mkdir(parents=True,exist_ok=True)
  try:
   s.stat(remote+'/'+f)
   s.get(remote+'/'+f,str(dest))
  except FileNotFoundError:continue
c.close()
print('Evidence saved:',local)
for arm in ('normalized_u','raw_angle'):
 f=local/arm/'benchmark.json'
 if f.exists() and f.stat().st_size:print(arm,json.dumps(json.loads(f.read_text())))
