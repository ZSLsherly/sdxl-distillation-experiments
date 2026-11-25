from pathlib import Path
import json
import re
import shlex
import sys
import time
sys.path.insert(0,str(Path.cwd()/'.image_runtime_pkgs'))
import paramiko

base='/data5/shenglin.zeng/checkpoints/a800_abc_g160_20261102'
local=Path('output/sdxl-crossdistill/a800_BC_paired_quota_20261102')
record=Path('output/sdxl-crossdistill/实验记录.txt')
arms={'A':('uniform_control','00e8abc9e66f5ebeca6e4733e3b981c0fcc34efe3583100ac565d6a8def8e695'),
      'B':('B','78045ea389b6c781ae81b2b096756ab8104dfe02911cd7cfc920bfc3e9d1aa46'),
      'C':('C','9b117072217a8825ba7a2978f244e3cbfa251aff6b9267c49d9cae18e4c3294b')}
def connect(host,port,user,password):
    c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host,port,username=user,password=password,timeout=20,look_for_keys=False,allow_agent=False)
    c.get_transport().set_keepalive(30)
    return c
def run(c,cmd):
    _,out,err=c.exec_command(cmd); data=out.read().decode().strip()
    if out.channel.recv_exit_status(): raise RuntimeError(err.read().decode() or cmd)
    return data

def transfer(src,dst,name,arm,sha):
    source=json.loads((local/'completed'/f'checkpoint_validation_{arm}.json').read_text())
    size=source['bytes'];origin=shlex.quote(source['file']);folder=f'{base}/{name}';target=f'{folder}/student_rank0.pt'
    partial=target+'.partial'
    run(dst,f'umask 077; mkdir -p {folder}')
    if run(dst,f'test ! -e {target} || echo exists'):
        if run(dst,f'sha256sum {target}').split()[0]!=sha:raise ValueError('Unexpected existing '+target)
    else:
        offset=int(run(dst,f'if test -e {partial}; then stat -c %s {partial}; else echo 0; fi'))
        if offset>size:raise ValueError('Unexpected partial size')
        print('BEGIN',name,offset,size,flush=True)
        if offset<size:
            _,sout,serr=src.exec_command(f'tail -c +{offset+1} -- {origin}')
            writer,ack,err=dst.exec_command(f'umask 077; cat >> {partial}')
            begin=last=time.monotonic(); sent=offset
            while block:=sout.read(1024*1024):
                writer.channel.sendall(block);sent+=len(block)
                if time.monotonic()-last>=20:
                    print('COPY',name,sent,size,round((sent-offset)/1048576/(time.monotonic()-begin),2),'MiB/s',flush=True)
                    last=time.monotonic()
            writer.channel.shutdown_write()
            if ack.channel.recv_exit_status() or sout.channel.recv_exit_status():raise RuntimeError('stream failed')
        result=run(dst,f'stat -c %s {partial}; sha256sum {partial}').split()
        if int(result[0])!=size or result[1]!=sha:raise ValueError('Size/hash mismatch '+name)
        run(dst,f'mv -T --no-clobber {partial} {target}; test ! -e {partial}')
    with dst.open_sftp() as sf:
        for file in ['config.json','status.json']:
            sf.put(str(local/'completed'/arm/file),folder+'/'+file)
        sf.put(str(local/'completed'/f'checkpoint_validation_{arm}.json'),folder+'/source_validation.json')
    print('VERIFIED',name,size,sha,flush=True)

def main():
    pw=sys.stdin.readline().strip()
    raw=Path('C:/Users/sherl/Desktop/login.txt').read_text(encoding='utf-8-sig')
    match=re.search(r'IP:\s*172\.18\.33\.60\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)
    src=connect('connect.nma1.seetacloud.com',14892,'root',pw)
    dst=connect('172.18.33.60',22,match[1],match[2].strip())
    try:
        run(dst,f'umask 077; mkdir -p {base}')
        locker,reply,_=dst.exec_command(f'flock -n {base}/.copy.lock sh -c "echo OWNED; cat >/dev/null"')
        if reply.readline().strip()!='OWNED':raise RuntimeError('another transfer holds the lock')
        def sync_note():
            with dst.open_sftp() as sf:
                sf.put(str(record),base+'/EXPERIMENTS.txt.new')
            run(dst,f'mv -f {base}/EXPERIMENTS.txt.new {base}/EXPERIMENTS.txt')
        sync_note()
        for name,(arm,sha) in arms.items(): transfer(src,dst,name,arm,sha)
        msg=f'三组备份均已按文件大小及 SHA-256 校验通过。目的服务器 172.18.33.60，目录 {base}。'
        if msg not in record.read_text(encoding='utf8'):
            with record.open('a',encoding='utf8') as f: f.write(time.strftime('%Y-%m-%d %H:%M:%S')+' '+msg+'\n')
        sync_note()
    finally:
        dst.close();src.close()
if __name__=='__main__':main()
