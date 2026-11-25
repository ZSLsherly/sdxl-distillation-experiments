import sys,re
from pathlib import Path
sys.path.insert(0,'.image_runtime_pkgs')
import paramiko
raw=Path(r'C:\Users\sherl\Desktop\login.txt').read_text(encoding='utf-8-sig')
creds={ip:(u,p.strip()) for ip,u,p in re.findall(r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)}
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());u,p=creds['172.18.33.60']
c.connect('172.18.33.60',username=u,password=p,timeout=15,look_for_keys=False,allow_agent=False)
root=Path('output/sdxl-crossdistill/hparam_pilot');remote='/data5/shenglin.zeng/sdxl_crossdistill/outputs/hparam_pilot'
patterns=['report.html','parameter_comparisons.json','training_curves.png','baseline_four_models.png','validation*_four_trials.png','teacher_stages/*_four_stages.png','teacher_stages/*_stages_and_seeds.png','teacher_stages/teacher_final_references.png','teacher_turning/turning_profile.png','teacher_turning/earlier_four_stages.png']
files=sorted({p for pattern in patterns for p in root.glob(pattern)})
assert (root/'report.html') in files
with c.open_sftp() as s:
    for path in files:s.put(str(path),remote+'/'+path.relative_to(root).as_posix())
c.close();print('REMOTE_REPORT_SAVED',remote+'/report.html','files',len(files),flush=True)
