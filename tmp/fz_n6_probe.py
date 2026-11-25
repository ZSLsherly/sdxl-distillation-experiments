from backup_a800_nov5 import cloud,lab,run,pw
from abc_direct_grant import store
from pathlib import Path
import hashlib,base64,json
with cloud() as s,lab() as d:
 k=s.get_transport().get_remote_server_key().asbytes()
 c={'password':pw,'sha256':base64.b64encode(hashlib.sha256(k).digest()).decode().rstrip('='),'md5':':'.join(format(x,'02x') for x in hashlib.md5(k).digest())}
 dest='/data5/shenglin.zeng/checkpoints/a800_archive_20261106/.pull6/fz_probe.py'
 store(d,dest,Path('tmp/fz_n6_probe_remote.py').read_text(),0o700)
 w,r,e=d.exec_command('python3 '+dest)
 w.channel.sendall((json.dumps(c)+'\n').encode());w.channel.shutdown_write()
 print(r.read().decode());print(e.read().decode());assert r.channel.recv_exit_status()==0
