from pathlib import Path
import sys,re,json
sys.path.insert(0,str(Path.cwd()/'.image_runtime_pkgs'))
import paramiko
raw=Path('C:/Users/sherl/Desktop/login.txt').read_text(encoding='utf-8-sig')
a={ip:(u,p.strip()) for ip,u,p in re.findall(r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)}
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect('172.18.33.60',username=a['172.18.33.60'][0],password=a['172.18.33.60'][1],look_for_keys=False,allow_agent=False,timeout=20)
p='/data5/shenglin.zeng/tools/eval_transfer_20261025/lab_to_3090_n11.py'
with c.open_sftp() as f:f.put('tmp/lab_to_3090_n11.py',p)
w,o,e=c.exec_command('/data5/shenglin.zeng/envs/python-runtime/bin/python3.12 -u '+p);o.channel.set_combine_stderr(True)
w.write(json.dumps(dict(zip(['user','password'],a['172.18.32.78'])))+'\n');w.flush();w.channel.shutdown_write()
for l in o:print(l.strip(),flush=True)
raise SystemExit(o.channel.recv_exit_status())
