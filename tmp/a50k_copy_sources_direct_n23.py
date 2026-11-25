"""Run rclone on source lab host, forwarding only destination credentials in stdin."""
import json, shlex, sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'tmp'))
from a50k_stage_assets_n23 import AUTH, OUT, DEST, remote, record, paramiko
code=r'''
import json,sys,subprocess,os,hashlib
r=json.loads(sys.stdin.readline());rc='/data5/shenglin.zeng/checkpoints/a800_archive_20261114/.tools/rclone'
env=dict(os.environ,RCLONE_CONFIG='/dev/null',RCLONE_CONFIG_DEST_TYPE='sftp',
    RCLONE_CONFIG_DEST_HOST='172.18.32.81',RCLONE_CONFIG_DEST_PORT='22',RCLONE_CONFIG_DEST_USER=r['user'])
env['RCLONE_CONFIG_DEST_PASS']=subprocess.check_output([rc,'obscure','-'],input=r.pop('password')+'\n',universal_newlines=True).strip()
for spec in r['specs']:
    src='/data5/shenglin.zeng/checkpoints/a800_archive_20261106/datasets/20k_sources/'+spec['name']
    dst=r['dest']+'/datasets/sources/'+spec['name']
    subprocess.check_call([rc,'copyto',src,'dest:'+dst,'--stats','15s','--stats-one-line',
        '--log-level','NOTICE','--retries','2','--contimeout','15s'],env=env)
    print('DIRECT_COPY_COMPLETE',spec['name'],flush=True)
'''
host='172.18.33.60'
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(host,username=AUTH[host][0],password=AUTH[host][1],timeout=20,look_for_keys=False,allow_agent=False)
c.get_transport().set_keepalive(15)
specs=[s for s in json.loads((OUT/'sources.json').read_text())['files'] if s['shard'] in (2,3)]
i,o,e=c.exec_command('python3 -u -c '+shlex.quote(code),timeout=3600);o.channel.set_combine_stderr(True)
i.write(json.dumps(dict(user=AUTH['172.18.32.81'][0],password=AUTH['172.18.32.81'][1],specs=specs,dest=DEST))+'\n')
i.flush();i.channel.shutdown_write()
for line in o:print(line.rstrip(),flush=True)
assert o.channel.recv_exit_status()==0;c.close()
for spec in specs:
    dest=DEST+'/datasets/sources/'+spec['name']
    got=remote('sha256sum '+shlex.quote(dest)).split()[0]
    assert got==spec['sha256']
    record(dict(source_host=host,source='/data5/shenglin.zeng/checkpoints/a800_archive_20261106/datasets/20k_sources/'+spec['name'],
                destination_host='172.18.32.81',destination=dest,sha256=got,verified=True,method='server-side rclone'))
    print('DIRECT_HASH_VERIFIED',spec['name'],flush=True)
