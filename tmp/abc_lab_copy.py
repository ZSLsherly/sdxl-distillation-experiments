import json, os, re, subprocess, concurrent.futures, datetime, time, fcntl, hashlib, shutil
from pathlib import Path
base=Path('/data5/shenglin.zeng/checkpoints/a800_abc_g160_20261102')
work=base/'.abc_g160_transfer_20261102'
records=json.loads((work/'files.json').read_text())
CHUNK=64*1024**2
ssh=['ssh','-oBatchMode=yes','-oIdentitiesOnly=yes','-oStrictHostKeyChecking=yes',
     '-oConnectTimeout=10','-oServerAliveInterval=15','-oServerAliveCountMax=3',
     '-oUserKnownHostsFile='+str(work/'known_hosts'),'-i',str(work/'id'),'-p','14892','root@connect.nma1.seetacloud.com']
def emit(msg):
    print(datetime.datetime.now().astimezone().isoformat(timespec='seconds'),msg,flush=True)
def get(piece):
    path,arm,start,length=piece
    if path.exists() and path.stat().st_size==length: return length
    stage=path.with_suffix('.part')
    for attempt in range(4):
        if time.time()>int((work/'expiry').read_text())-20: raise RuntimeError('Transfer grant expired')
        offset = stage.stat().st_size if stage.exists() else 0
        if offset > length: raise ValueError('Invalid piece size')
        with open(stage,'ab') as f:
            result=subprocess.run(ssh+[f'{arm} {start+offset} {length-offset}'],stdout=f,timeout=1200)
        if result.returncode==0 and stage.stat().st_size==length:
            os.replace(stage,path);return length
        time.sleep(2**attempt)
    raise ValueError('could not read '+str(piece))
def check(path,want):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    assert h.hexdigest()==want,(path,h.hexdigest(),want)
def main():
    os.umask(0o077)
    with (work/'direct.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for arm,rec in records.items():
            size,digest=rec['size'],rec['sha256']
            folder=base/arm;folder.mkdir(exist_ok=True)
            out=folder/'student_rank0.pt';cache=work/arm;cache.mkdir(exist_ok=True)
            if not out.exists():
                pieces=[(cache/f'{s:012d}.chunk',arm,s,min(CHUNK,size-s)) for s in range(0,size,CHUNK)]
                begin=time.monotonic();done=0
                with concurrent.futures.ThreadPoolExecutor(max_workers=14) as pool:
                    futures=[pool.submit(get,p) for p in pieces]
                    for f in concurrent.futures.as_completed(futures):
                        done+=f.result();emit(f'{arm} {done}/{size} chunk bytes')
                tmp=folder/'student_rank0.pt.assembling'
                with tmp.open('wb') as target:
                    for piece,*_ in pieces:
                        with piece.open('rb') as src:shutil.copyfileobj(src,target,4*1024**2)
                assert tmp.stat().st_size==size
                check(tmp,digest)
                os.link(tmp,out);tmp.unlink()
                for p,*_ in pieces:p.unlink()
            else:
                assert out.stat().st_size==size;check(out,digest)
            receipt=f'{arm} {size} {digest} {out}'
            (folder/'VERIFIED.txt').write_text(receipt+'\n')
            emit('VERIFIED '+receipt)
        emit('ALL_THREE_VERIFIED')
if __name__=='__main__':main()
