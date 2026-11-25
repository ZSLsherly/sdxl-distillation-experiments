"""Fetch only pilot reports and images; leave all checkpoints on /data5."""
import sys,re,json
from pathlib import Path
sys.path.insert(0,'.image_runtime_pkgs')
import paramiko
raw=Path(r'C:\Users\sherl\Desktop\login.txt').read_text(encoding='utf-8-sig')
creds={ip:(u,p.strip()) for ip,u,p in re.findall(r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)}
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());u,p=creds['172.18.33.60']
c.connect('172.18.33.60',username=u,password=p,timeout=15,look_for_keys=False,allow_agent=False)
remote='/data5/shenglin.zeng/sdxl_crossdistill/outputs/hparam_pilot';local=Path('output/sdxl-crossdistill/hparam_pilot');local.mkdir(parents=True,exist_ok=True)
trials=['lr5e7_pcm10','lr25e8_pcm10','lr5e7_pcm1','lr25e8_pcm1']
with c.open_sftp() as s:
    def fetch(name):
        dest=local/name;dest.parent.mkdir(parents=True,exist_ok=True);s.get(remote+'/'+name,str(dest))
    fetch('protocol.json')
    for name in ['audit.json','scheduling_incident.json']:fetch(name)
    for trial in trials:
        for name in s.listdir(remote+'/'+trial):
            if name.endswith(('.json','.jsonl','.sha256')):fetch(trial+'/'+name)
        print('TRIAL_REPORTS_SAVED',trial,flush=True)
    for name in s.listdir(remote+'/assessment'):
        if name.endswith(('.json','.png')):fetch('assessment/'+name)
    for name in s.listdir(remote+'/teacher_turning'):
        if name.endswith(('.json','.png')):fetch('teacher_turning/'+name)
c.close();print('PILOT_REPORTS_DOWNLOADED',local.resolve(),flush=True)
