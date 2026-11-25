"""Short-lived read-only SSH grant for exactly the three approved student files."""
import base64, json, os, re, shlex, sys, time
from pathlib import Path
from archive_abc_to_lab import arms, base, connect, local, run
token='abc_g160_transfer_20261102'
source_dir='/root/sdxl_research/tmp/'+token
lab_dir=base+'/.'+token
def store(c,path,data,mode):
    with c.open_sftp() as sf:
        with sf.file(path,'w') as f:f.write(data.encode())
        sf.chmod(path,mode)
def main():
    pwd=sys.stdin.readline().strip()
    raw=Path('C:/Users/sherl/Desktop/login.txt').read_text(encoding='utf-8-sig')
    cred=re.search(r'IP:\s*172\.18\.33\.60\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)
    cloud=connect('connect.nma1.seetacloud.com',14892,'root',pwd)
    lab=connect('172.18.33.60',22,cred[1],cred[2].strip())
    try:
        if sys.argv[1]=='cleanup':
            run(cloud,f'/root/sdxl_research/env/bin/python {source_dir}/revoke.py')
            run(lab,f'rm -f -- {lab_dir}/id {lab_dir}/id.pub')
            print('temporary grant revoked and transfer identity removed');return
        if sys.argv[1]!='setup':raise ValueError('unknown operation')
        run(lab,f'umask 077; mkdir -p {base}; mkdir -p -m 700 {lab_dir}; test -s {lab_dir}/id || ssh-keygen -q -t ed25519 -N "" -C {token} -f {lab_dir}/id')
        pub=run(lab,f'cat {lab_dir}/id.pub')
        assert re.fullmatch(r'ssh-ed25519 [A-Za-z0-9+/=]+ '+token,pub)
        hostkey=cloud.get_transport().get_remote_server_key()
        store(lab,lab_dir+'/known_hosts',f'[connect.nma1.seetacloud.com]:14892 {hostkey.get_name()} {hostkey.get_base64()}\n',0o600)
        paths={a:json.loads((local/'completed'/f'checkpoint_validation_{arm}.json').read_text())['file'] for a,(arm,h) in arms.items()}
        run(cloud,f'umask 077; mkdir -p -m 700 {source_dir}')
        expiry=int(run(cloud,'date +%s'))+3*3600
        reader=f"""#!/root/sdxl_research/env/bin/python
import os,re,time
files={paths!r}
remaining={expiry}-int(time.time())
if remaining<=0:raise SystemExit(77)
cmd=os.environ.get('SSH_ORIGINAL_COMMAND','')
if cmd=='probe':
    args=['head','-c','4194304','--',files['A']]
else:
    m=re.fullmatch(r'([ABC]) ([0-9]{{1,12}})',cmd)
    if not m:raise SystemExit(77)
    f=files[m[1]];off=int(m[2])
    if off>os.stat(f).st_size:raise SystemExit(77)
    args=['tail','-c','+'+str(off+1),'--',f]
os.execvp('timeout',['timeout',str(remaining),*args])
"""
        store(cloud,source_dir+'/read.py',reader,0o700)
        entry=f'restrict,command="{source_dir}/read.py" {pub}'
        grant='/root/.ssh/authorized_keys2'
        maint=f"""import os,pathlib,time,sys,fcntl
p=pathlib.Path({str(grant)!r}); line={entry.encode()!r}
assert p.resolve() in (pathlib.Path('/root/.ssh/authorized_keys'),pathlib.Path('/root/.ssh/authorized_keys2')) and p.is_file()
mode=sys.argv[1] if len(sys.argv)>1 else 'remove'
with p.open('r+b') as f:
    fcntl.flock(f,fcntl.LOCK_EX)
    data=f.read(); matches=[s for s in data.splitlines() if s==line]
    if mode=='add':
        assert not matches and time.time()<{expiry}
        f.seek(0,2);f.write((b'' if not data or data.endswith(b'\\n') else b'\\n')+line+b'\\n')
    elif mode=='remove' and matches:
        assert len(matches)==1
        f.seek(0);f.write(b''.join(s for s in data.splitlines(keepends=True) if s.rstrip(b'\\r\\n')!=line));f.truncate()
    f.flush();os.fsync(f.fileno())
"""
        store(cloud,source_dir+'/revoke.py',maint,0o700)
        run(cloud,f'/root/sdxl_research/env/bin/python {source_dir}/revoke.py add')
        # Arm expiration independently of workstation / VPN availability.
        run(cloud,f'nohup sh -c "sleep 10801; /root/sdxl_research/env/bin/python {source_dir}/revoke.py" </dev/null >/dev/null 2>&1 &')
        ssh=f'ssh -oBatchMode=yes -oIdentitiesOnly=yes -oStrictHostKeyChecking=yes -oUserKnownHostsFile={lab_dir}/known_hosts -i {lab_dir}/id -p 14892 root@connect.nma1.seetacloud.com'
        begin=time.monotonic()
        print('probe bytes',run(lab,f'bash -o pipefail -c {shlex.quote(ssh+" probe | wc -c")}'),'seconds',round(time.monotonic()-begin,2),flush=True)
        print('limited grant, expiry unix',expiry)
    finally:
        lab.close();cloud.close()
if __name__=='__main__':main()
