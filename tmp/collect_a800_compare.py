import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'.image_runtime_pkgs'))
import paramiko
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('connect.nma1.seetacloud.com',port=14892,username='root',password=sys.stdin.readline().strip(),timeout=20,look_for_keys=False,allow_agent=False)
root='/root/autodl-tmp/sdxl_runs/outputs/a800_B_offload_compare_20261101'
out=Path('output/sdxl-crossdistill/a800_B_batch_probe_20261101/compare');out.mkdir(parents=True,exist_ok=True)
with c.open_sftp() as s:
 for name in ['results.json','finished.json','resources.csv','gpu4/config.json','gpu4/rank0.jsonl','gpu4/status.json','adam_cpu16/config.json','adam_cpu16/rank0.jsonl','adam_cpu16/status.json']:
  target=out/name;target.parent.mkdir(parents=True,exist_ok=True);s.get(root+'/'+name,str(target));print('collected',name)
c.close()
