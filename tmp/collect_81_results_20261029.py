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
local=Path('output/sdxl-crossdistill/turning_ablation_20261027/four_gpu_20261029')
files=['storage_estimate.json','pipeline_offline_20261029.log','pipeline_expandable_retry1_20261029.log','pipeline_teacher_shard_20261029.log','outputs/batch64_teacher_shard_workflow.log']
for run in ['smoke_four_gpu_20261028','smoke_four_gpu_expandable_retry1_20261029','smoke_four_gpu_teacher_shard_20261029']:
 files += [f'outputs/{run}/workflow.log',f'outputs/{run}/normalized_u.log']
 files += [f'outputs/{run}/normalized_u/{f}' for f in ['config.json','status.json','benchmark.json','source_sha256.json','rank0.jsonl','rank1.jsonl','rank2.jsonl','rank3.jsonl']]
files += [f'outputs/smoke_four_gpu_teacher_shard_20261029/normalized_u_teacher_check/rank{rank}.json' for rank in range(4)]
for arm in ['normalized_u','raw_angle']:
 for phase in ['validation','g160']:
  files += [f'outputs/turning_batch64_g160_4gpu_teacher_shard_20261029/{arm}_{phase}/{f}' for f in ['status.json','config.json','rank0.jsonl']]
with c.open_sftp() as s:
 for name in files:
  try:s.stat(remote+'/'+name)
  except FileNotFoundError:continue
  path=local/name;path.parent.mkdir(parents=True,exist_ok=True);s.get(remote+'/'+name,str(path))
c.close()
print('Small evidence files saved:',local)
