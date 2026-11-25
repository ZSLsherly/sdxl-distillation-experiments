from backup_a800_nov5 import cloud,lab,report,LAB,run,q
from abc_direct_grant import store
from pathlib import Path
import time,re,subprocess,json
token='backup_n5_20261105'
stage='/root/sdxl_research/tmp/'+token
private=LAB+'/.pull'
with cloud() as s,lab() as d:
 run(s,'mkdir -p -m 700 '+stage);run(d,'umask 077;mkdir -p '+private)
 for n in ('build_backup_n5','read_backup_n5'):
  store(s,stage+'/'+n+'.py',Path('tmp/'+n+'.py').read_text(),0o700)
 print('Building source manifest',flush=True)
 run(s,'nice -n 12 python3 '+stage+'/build_backup_n5.py')
 run(d,'test -f '+private+'/key || ssh-keygen -q -t ed25519 -N "" -C '+token+' -f '+private+'/key')
 pub=run(d,'cat '+private+'/key.pub');assert re.fullmatch(r'ssh-ed25519 [A-Za-z0-9+/=]+ '+token,pub)
 hk=s.get_transport().get_remote_server_key()
 store(d,private+'/hosts','[connect.nma1.seetacloud.com]:14892 '+hk.get_name()+' '+hk.get_base64()+'\n',0o600)
 expiry=int(time.time())+24*3600
 store(s,stage+'/expiry',str(expiry),0o600)
 keyfile='/root/.ssh/authorized_keys2'
 cmd='command="/usr/bin/python3 '+stage+'/read_backup_n5.py",restrict,expiry-time="'+time.strftime('%Y%m%d%H%M%SZ',time.gmtime(expiry))+'" '+pub
 revoke="""from pathlib import Path
import fcntl
p=Path('/root/.ssh/authorized_keys2')
with p.open('r+b') as f:
 fcntl.flock(f,fcntl.LOCK_EX);ls=f.readlines();f.seek(0);f.writelines([x for x in ls if x.rstrip()!=LINE]);f.truncate()
""".replace('LINE',repr(cmd.encode()))
 store(s,stage+'/revoke.py',revoke,0o700)
 run(s,'test -f '+keyfile+' && ! grep -q '+token+' '+keyfile+' && printf "%s\\n" '+q(cmd)+' >> '+keyfile)
 store(d,private+'/pull.py',Path('tmp/pull_backup_n5.py').read_text(),0o700)
 print(run(d,f'nohup python3 -u {private}/pull.py </dev/null >{private}/log 2>&1 & echo $!'),flush=True)

