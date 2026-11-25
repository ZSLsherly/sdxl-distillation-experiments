from backup_a800_nov5 import cloud,lab,run,pw
from abc_direct_grant import store
from pathlib import Path
import json,hashlib,base64
stage='/root/sdxl_research/tmp/side_archive_n6';dest='/data5/shenglin.zeng/checkpoints/a800_archive_20261106'
with cloud() as s,lab() as d:
 pack="""import json,subprocess,shutil
from pathlib import Path
s=Path(STAGE);r=Path('/root/autodl-tmp/sdxl_runs');m=json.loads((s/'manifest.json').read_text())
need=sum(m['records'][k][0] for g in m['groups'] for k in g)
assert shutil.disk_usage(r).free>need+7*2**30
(s/'groups').mkdir(exist_ok=True)
for i in range(len(m['groups'])):
 p=s/'groups'/f'part_{i:03d}.tar'
 if not p.exists():subprocess.run(['tar','-C',str(r),'-cf',str(p),'--no-recursion','--null','-T',str(s/f'list{i}')],check=True)
print('PACKED',len(m['groups']),need)
""".replace('STAGE',repr(stage))
 store(s,stage+'/pack_fz.py',pack,0o700)
 print(run(s,'nice -n 15 /root/sdxl_research/env/bin/python '+stage+'/pack_fz.py'),flush=True)
 key=s.get_transport().get_remote_server_key().asbytes()
 c={'password':pw,'sha256':base64.b64encode(hashlib.sha256(key).digest()).decode().rstrip('='),'md5':':'.join(format(x,'02x') for x in hashlib.md5(key).digest())}
 store(d,dest+'/.pull6/fz_backup.py',Path('tmp/fz_n6_backup_remote.py').read_text(),0o700)
 w,r,e=d.exec_command('python3 '+dest+'/.pull6/fz_backup.py')
 w.channel.sendall((json.dumps(c)+'\n').encode());w.channel.shutdown_write()
 print(r.read().decode());print(e.read().decode());assert r.channel.recv_exit_status()==0
 print(run(s,'/root/sdxl_research/env/bin/python '+stage+'/revoke.py'))
