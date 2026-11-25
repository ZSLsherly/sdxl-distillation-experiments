import sys,re
from pathlib import Path
sys.path.insert(0,'.image_runtime_pkgs')
import paramiko
raw=Path(r'C:\Users\sherl\Desktop\login.txt').read_text(encoding='utf-8-sig')
creds={ip:(u,p.strip()) for ip,u,p in re.findall(r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)}
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());u,p=creds['172.18.33.60']
c.connect('172.18.33.60',username=u,password=p,timeout=15,look_for_keys=False,allow_agent=False)
remote='/data5/shenglin.zeng/sdxl_crossdistill/outputs/gan_stage_pilot_v2'
local=Path('output/sdxl-crossdistill/gan_stage_pilot_v2');local.mkdir(parents=True,exist_ok=True)
with c.open_sftp() as s:
    def fetch(name):
        dest=local/name;dest.parent.mkdir(parents=True,exist_ok=True);s.get(remote+'/'+name,str(dest))
    for name in ['report.html','audit.json','completion.json','data_overlap_audit.json']:fetch(name)
    for trial in ['dmd','gan4','joint4','joint34_matched']:
        for name in s.listdir(remote+'/'+trial):
            if name.endswith(('.json','.jsonl')):fetch(trial+'/'+name)
    for i,name in enumerate(s.listdir(remote+'/assessment')):
        if name.endswith(('.png','.json')):fetch('assessment/'+name)
        if i%30==0:print('FETCHED_ASSESSMENT_FILES',i,flush=True)
c.close();print('SAVED_REPORT',local.resolve()/'report.html',flush=True)
