"""Copy only the two requested student checkpoints, no credential forwarding."""
import concurrent.futures,hashlib,json,os,re,shlex,sys,threading,time
from pathlib import Path
from archive_abc_to_lab import connect,run
from contextlib import closing
BASE='/data5/shenglin.zeng/checkpoints/a800_20261103'
ROOT='/root/autodl-tmp/sdxl_runs/outputs/'
FILES={
'C_mix1_G160':('bc_winner_late_mix_20261102/late_mix','20261103T074022+0800_g000160_end','391337f02679006b7cb76c292eb39236e84496fda40183fadb88fad361e42aed'),
'D_G9':('interval_D_20261103/D','20261103T105931+0800_g000009_stopped','60de01cc02225bbe7f6b336d4947e3744b68578df1b6145e120c5beafbf9d3a5')}
size=5135506134; chunk=4*1024**2; nworker=16
q=shlex.quote; local=threading.local(); initlock=threading.Lock()
password=sys.stdin.readline().strip()
raw=Path('C:/Users/sherl/Desktop/login.txt').read_text(encoding='utf-8-sig')
m=re.search(r'IP:\s*172\.18\.33\.60\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw);lu,lp=m[1],m[2].strip();del raw,m
def pair():
    s=connect('connect.nma1.seetacloud.com',14892,'root',password)
    try:return s,connect('172.18.33.60',22,lu,lp)
    except Exception:s.close();raise
def connections():
    if not hasattr(local,'p'):
        with initlock: local.p=pair()
    return local.p
def chunk_copy(entry):
    origin,folder,start,length=entry
    final=folder+f'/{start:012d}'
    for retry in range(5):
        s,d=connections()
        try:
            _,f,e=s.exec_command(f'dd if={q(origin)} bs=1M iflag=skip_bytes,count_bytes,fullblock skip={start} count={length} status=none',timeout=300)
            b=f.read()
            if len(b)!=length or f.channel.recv_exit_status():raise OSError('short source block')
            w,r,err=d.exec_command(f'umask 077; cat > {q(final+".part")} && test "$(stat -c %s {q(final+".part")})" = {length} && mv {q(final+".part")} {q(final)}',timeout=300)
            w.channel.sendall(b);w.channel.shutdown_write()
            if r.channel.recv_exit_status():raise OSError(err.read().decode())
            return length
        except Exception:
            s.close();d.close();del local.p
            if retry==4:raise
            time.sleep(1+retry)
def transfer(arm,cloud,lab,pool):
    rd,stamp,digest=FILES[arm];src=ROOT+rd; parent=BASE+'/'+arm
    checkpoint=parent+'/weights/'+stamp; target=checkpoint+'/student_rank0.pt'
    origin=src+'/weights/'+stamp+'/student_rank0.pt'
    stage=parent+'/.transfer_parts'
    print('BEGIN',arm,flush=True)
    run(lab,f'umask 077; mkdir -p {q(checkpoint)} {q(stage)}')
    with cloud.open_sftp() as s,lab.open_sftp() as d:
        for name in ('config.json','status.json','weights/'+stamp+'/complete.json','weights/'+stamp+'/shapes_rank0.json'):
            with s.file(src+'/'+name,'rb') as f: b=f.read()
            json.loads(b)
            path=parent+'/'+name
            with d.file(path+'.new','wb') as f:f.write(b)
            h=hashlib.sha256(b).hexdigest()
            assert run(lab,f'sha256sum {q(path+".new")}').split()[0]==h
            run(lab,f'mv -f {q(path+".new")} {q(path)}')
    if run(lab,f'test ! -e {q(target)} || echo exists')!='exists':
        present=run(lab,f'find {q(stage)} -maxdepth 1 -type f -printf "%f %s\\n"')
        old={n:int(v) for n,v in (a.split() for a in present.splitlines()) if re.fullmatch(r'[0-9]{12}',n)}
        todo=[];done=0
        for off in range(0,size,chunk):
            length=min(chunk,size-off)
            if old.get(f'{off:012d}')==length:done+=length;continue
            todo.append((origin,stage,off,length))
        start=time.monotonic();initial=done;last=start
        for f in concurrent.futures.as_completed([pool.submit(chunk_copy,e) for e in todo]):
            done+=f.result();now=time.monotonic()
            if now-last>=30 or done==size:
                print('BYTES',arm,done,size,'MiB/s',round((done-initial)/2**20/(now-start),2),flush=True);last=now
        script="""import hashlib,os,sys
from pathlib import Path
stage,out=map(Path,sys.argv[1:3]); size=int(sys.argv[3]); chunk=int(sys.argv[4]); digest=sys.argv[5]
tmp=out.with_suffix('.assembling'); h=hashlib.sha256()
with tmp.open('wb') as dest:
    for off in range(0,size,chunk):
        b=(stage/f'{off:012d}').read_bytes();assert len(b)==min(chunk,size-off);h.update(b);dest.write(b)
    dest.flush();os.fsync(dest.fileno())
assert tmp.stat().st_size==size and h.hexdigest()==digest
os.link(tmp,out);tmp.unlink()
for off in range(0,size,chunk):(stage/f'{off:012d}').unlink()
"""
        run(lab,'python3 -c '+q(script)+' '+' '.join(map(q,[stage,target,str(size),str(chunk),digest])))
    proof=run(lab,f'stat -c %s {q(target)}; sha256sum {q(target)}').split()
    assert int(proof[0])==size and proof[1]==digest
    receipt=f'{arm} {size} {digest} {target}'
    run(lab,f'printf "%s\\n" {q(receipt)} > {q(parent+"/VERIFIED.txt")}')
    print('VERIFIED',receipt,flush=True)
with concurrent.futures.ThreadPoolExecutor(nworker) as pool:
    cloud,lab=pair()
    try:
        run(lab,f'umask 077; mkdir -p {BASE}')
        locker,ack,_=lab.exec_command(f'flock -n {BASE}/.transfer.lock sh -c "echo OWNED; cat >/dev/null"')
        assert ack.readline().strip()=='OWNED'
        for name in FILES:transfer(name,cloud,lab,pool)
    finally:cloud.close();lab.close()
