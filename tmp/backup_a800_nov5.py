"""Backup completed SDXL research artifacts using two independent SSH sessions."""
from pathlib import Path,PurePosixPath
import re,sys,json,hashlib,shlex,time,threading
from concurrent.futures import ThreadPoolExecutor
from archive_abc_to_lab import connect,run
q=shlex.quote
ROOT='/root/autodl-tmp/sdxl_runs'
LAB='/data5/shenglin.zeng/checkpoints/a800_archive_20261105'
report=Path('output/sdxl-crossdistill/a800_archive_20261105')
report.mkdir(exist_ok=True)
pw=sys.stdin.readline().strip()
auth=re.search(r'IP:\s*172\.18\.33\.60\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',Path('C:/Users/sherl/Desktop/login.txt').read_text(encoding='utf-8-sig'))
def cloud():return connect('connect.nma1.seetacloud.com',14892,'root',pw)
def lab():return connect('172.18.33.60',22,auth[1],auth[2].strip())
arms=['c_mix1_dm234_fake3_20261103','c_mix1_first_dm_fake3_g160','c_mix1_pcm_dm4_matched_time','A_pcm10_g160','dm1_mid4_uniform_mix1_g160']
tl=threading.local()
def clients():
    if not hasattr(tl,'s'):tl.s,tl.d=cloud(),lab()
    return tl.s,tl.d
def sendpart(task):
    src,dest,off,n=task
    s,d=clients()
    for attempt in range(3):
        try:
            _,r,e=s.exec_command(f'dd if={q(src)} bs=1M iflag=skip_bytes,count_bytes,fullblock skip={off} count={n} status=none',timeout=180)
            w,ack,de=d.exec_command(f'dd of={q(dest)} bs=1M oflag=seek_bytes conv=notrunc seek={off} status=none',timeout=180)
            count=0
            while b:=r.read(1<<20):
                w.channel.sendall(b);count+=len(b)
            w.channel.shutdown_write()
            assert count==n and r.channel.recv_exit_status()==0 and ack.channel.recv_exit_status()==0
            return n
        except Exception:
            if attempt>=2:raise
            time.sleep(2)
def main():
    with cloud() as s,lab() as d,ThreadPoolExecutor(8) as pool:
        run(d,f'umask 077;mkdir -p {LAB}')
        proof={}
        for arm in arms:
            parent=ROOT+'/outputs/'+arm+'/model'
            src=run(s,'find -L '+q(parent+'/weights')+' -name student_rank0.pt -type f')
            if not src:continue
            assert len(src.splitlines())==1 and src.startswith(parent+'/weights/')
            size=int(run(s,'stat -Lc %s '+q(src))); assert size==5135506134
            h=run(s,'sha256sum '+q(src)).split()[0]
            dest=LAB+'/'+src.removeprefix(ROOT+'/')
            run(d,'umask 077;mkdir -p '+q(str(PurePosixPath(dest).parent)))
            if run(d,'test ! -e '+q(dest)+' || echo exists')!='exists':
                tmp=dest+'.partial';run(d,f'umask 077;truncate -s {size} {q(tmp)}')
                chunk=32<<20
                done=0;start=time.monotonic()
                for n in pool.map(sendpart,[(src,tmp,i,min(chunk,size-i)) for i in range(0,size,chunk)]):
                    done+=n
                    if done%(512<<20)==0: print(arm,done,round(done/(1<<20)/(time.monotonic()-start),1),'MiB/s',flush=True)
                assert run(d,'sha256sum '+q(tmp)).split()[0]==h
                run(d,'ln '+q(tmp)+' '+q(dest)+' && unlink '+q(tmp))
            assert int(run(d,'stat -c %s '+q(dest)))==size and run(d,'sha256sum '+q(dest)).split()[0]==h
            with s.open_sftp() as sf,d.open_sftp() as df:
                for name in ('config.json','status.json','data_manifest.json','source_sha256.json','rank0.jsonl'):
                    origin=parent+'/'+name; target=LAB+'/'+origin.removeprefix(ROOT+'/')
                    with sf.file(origin,'rb') as f:data=f.read()
                    with df.file(target+'.tmp','wb') as f:f.write(data)
                    assert run(d,'sha256sum '+q(target+'.tmp')).split()[0]==hashlib.sha256(data).hexdigest()
                    df.posix_rename(target+'.tmp',target)
                for name in ['complete.json','shapes_rank0.json']:
                    origin=str(PurePosixPath(src).parent/name);target=str(PurePosixPath(dest).parent/name)
                    with sf.file(origin,'rb') as f:data=f.read()
                    with df.file(target+'.tmp','wb') as f:f.write(data)
                    assert run(d,'sha256sum '+q(target+'.tmp')).split()[0]==hashlib.sha256(data).hexdigest()
                    df.posix_rename(target+'.tmp',target)
            proof[arm]=dict(src=src,dest=dest,sha256=h,size=size)
            (report/'weights.json').write_text(json.dumps(proof,indent=2))
            print('VERIFIED',arm,h,flush=True)
if __name__=='__main__':main()

