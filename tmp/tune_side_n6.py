from backup_a800_nov5 import lab,run
from abc_direct_grant import store
from pathlib import Path
LAB='/data5/shenglin.zeng/checkpoints/a800_archive_20261106';private=LAB+'/.pull6'
s=Path('tmp/pull_backup_n5.py').read_text()
s=s.replace('import os,sys,tarfile,hashlib,json,subprocess,time,concurrent.futures','import os,sys,tarfile,hashlib,json,subprocess,time,concurrent.futures,threading')
s=s.replace('def call(cmd,sink):','launch_lock=threading.Lock()\nlast_launch=[0.0]\ndef call(cmd,sink):')
s=s.replace('if subprocess.run(ssh+[cmd],stdout=sink).returncode==0:return',"with launch_lock:\n   time.sleep(max(0,1.0-(time.monotonic()-last_launch[0])))\n   child=subprocess.Popen(ssh+[cmd],stdout=sink)\n   last_launch[0]=time.monotonic()\n  if child.wait()==0:return")
s=s.replace('ThreadPoolExecutor(8)','ThreadPoolExecutor(64)')
with lab() as d:
 # Stop only this transfer controller and its immediate SSH children.
 stop="""import os,signal,time
from pathlib import Path
matches=[]
for p in Path('/proc').glob('[0-9]*'):
 try:c=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
 except (OSError,UnicodeDecodeError):continue
 if c.startswith('python3 -u ') and TARGET in c:matches.append(int(p.name))
for pid in matches:
 children=[]
 for p in Path('/proc').glob('[0-9]*'):
  try:
   s=(p/'stat').read_text().rsplit(')',1)[1].split()
   if int(s[1])==pid:children.append(int(p.name))
  except (OSError,ValueError):pass
 os.kill(pid,signal.SIGTERM)
 for child in children:
  try:os.kill(child,signal.SIGTERM)
  except ProcessLookupError:pass
 print('stopped',pid,children)
""".replace('TARGET',repr(private+'/pull.py'))
 store(d,private+'/stop_own.py',stop,0o700)
 print(run(d,'python3 '+private+'/stop_own.py'))
 store(d,private+'/pull.py',s,0o700)
 print(run(d,f'nohup python3 -u {private}/pull.py </dev/null >>{private}/log 2>&1 & echo $!'))
