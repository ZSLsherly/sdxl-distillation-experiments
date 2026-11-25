from a50k_stage_assets_n23 import AUTH,ENV,RCLONE,paramiko,remote
from pathlib import Path
import json,hashlib,subprocess
f=Path('tmp/a50k_prune_scored_n25.py');h=hashlib.sha256(f.read_bytes()).hexdigest()
for mode,node,remote_name,base in [('archive','172.18.32.81','dest','/data/shenglin.zeng/a50k_20261123/eval_A_milestones_20261125/code'),('training','172.18.33.60','source','/data5/shenglin.zeng/a50k_20261124/logs')]:
    dest=base+'/'+f.name
    subprocess.run([RCLONE,'copyto',str(f),remote_name+':'+dest],env=ENV,check=True)
    assert remote('sha256sum '+dest,node).split()[0]==h
    with Path('output/sdxl-crossdistill/a50k_train60_20261124/coco_milestones/部署记录.txt').open('a',encoding='utf8') as log:log.write(f'{f.resolve()} -> {node}:{dest} SHA256 {h}\n')
    c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect(node,username=AUTH[node][0],password=AUTH[node][1],timeout=20,look_for_keys=False,allow_agent=False)
    py='python3' if mode=='training' else '/data3/shenglin.zeng/sdxl_turning_ablation_20261028/env/bin/python'
    i,o,e=c.exec_command(py+' -u '+dest+' '+mode,timeout=30)
    if mode=='training':i.write(json.dumps(dict(u=AUTH['172.18.32.81'][0],p=AUTH['172.18.32.81'][1]))+'\n')
    i.flush();i.channel.shutdown_write()
    code=o.channel.recv_exit_status();c.close();assert not code,(node,code)
    print('ARMED',mode,flush=True)
