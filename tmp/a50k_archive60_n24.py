"""Start the lab-side rclone archiver. Login material is passed via stdin only."""
import json, shlex, sys
from a50k_stage_assets_n23 import AUTH, paramiko
from pathlib import Path
host='172.18.33.60'
root='/data5/shenglin.zeng/a50k_20261124'
worker=r'''
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
request=json.loads(sys.stdin.readline())
root=Path('/data5/shenglin.zeng/a50k_20261124')
lock=(root/'archive.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
exe='/data5/shenglin.zeng/checkpoints/a800_archive_20261114/.tools/rclone'
env=dict(os.environ,RCLONE_CONFIG='/dev/null',RCLONE_CONFIG_LAB_TYPE='sftp',RCLONE_CONFIG_LAB_HOST='172.18.32.81',RCLONE_CONFIG_LAB_USER=request['u'],RCLONE_CONFIG_LAB_PASS=subprocess.check_output([exe,'obscure','-'],input=request.pop('p')+'\n',universal_newlines=True).strip())
if os.fork():sys.exit(0)
os.setsid()
with open(root/'logs/archive.log','a',1) as log:
 os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
 with open('/dev/null') as f:os.dup2(f.fileno(),0)
 for g in (313,470,627,784):
  out=root/'outputs'/'A50K_singlepass'
  while True:
   paths=list((out/'training_state').glob('*_g%06d/complete.json'%g))
   if paths:break
   time.sleep(60)
  src=paths[0].parent
  meta=json.loads(paths[0].read_text());assert meta['progress']['gcount']==g and meta['world_size']==4
  dest='/data/shenglin.zeng/a50k_20261123/checkpoints/singlepass/'+src.name
  # Publish complete.json only once every payload has a matching SHA-256.
  subprocess.run([exe,'copy',str(src),'lab:'+dest,'--exclude','complete.json','--transfers','2','--retries','4'],env=env,check=True)
  for f in sorted(src.iterdir()):
   if f.name=='complete.json':continue
   actual=subprocess.check_output([exe,'hashsum','SHA-256','--download','lab:'+dest+'/'+f.name],env=env).decode().split()[0]
   expected=subprocess.check_output(['sha256sum',str(f)]).decode().split()[0]
   assert expected==actual,(f,expected,actual)
   print('172.18.33.60:'+str(f),'->','172.18.32.81:'+dest+'/'+f.name,actual,flush=True)
  f=src/'complete.json'
  subprocess.run([exe,'copyto',str(f),'lab:'+dest+'/complete.json'],env=env,check=True)
  actual=subprocess.check_output([exe,'hashsum','SHA-256','--download','lab:'+dest+'/complete.json'],env=env).decode().split()[0]
  assert actual==subprocess.check_output(['sha256sum',str(f)]).decode().split()[0]
  with (root/'归档校验.txt').open('a') as report:report.write('G%d 172.18.33.60:%s -> 172.18.32.81:%s all file SHA256 verified; complete.json %s\n'%(g,src,dest,actual))
'''
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(host,username=AUTH[host][0],password=AUTH[host][1],timeout=20,look_for_keys=False,allow_agent=False)
i,o,e=c.exec_command("python3 -u -c "+shlex.quote(worker),timeout=30)
i.write(json.dumps(dict(u=AUTH['172.18.32.81'][0],p=AUTH['172.18.32.81'][1]))+'\n');i.flush();i.channel.shutdown_write()
rc=o.channel.recv_exit_status();print('ARCHIVE_START_RC',rc);c.close();sys.exit(rc)
