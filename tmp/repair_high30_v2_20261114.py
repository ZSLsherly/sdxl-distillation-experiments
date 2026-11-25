"""Drive the installed FileZilla SFTP handler; the weight stays server-to-server."""
from pathlib import Path
import base64,datetime,fcntl,hashlib,json,os,subprocess,sys,time,threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_EXCEPTION

ROOT=Path('/data5/shenglin.zeng/checkpoints/a800_archive_20261114')
GROUP='A_clean_global_high30_g160_20261114_v2'
REL=Path('outputs')/GROUP/'model/weights/20261114T172912+0800_g000160_end/student_rank0.pt'
SOURCE='/root/sdxl_research/'+str(REL)
EXPECTED='95cf20b296bdc5d5c2fffce717c969f624aca7e5c81bb6c7de3a6ee2245be437'
SIZE=5135506134
TOOL=Path('/data5/shenglin.zeng/tools/filezilla/app/usr/bin/fzsftp')
PW=sys.stdin.readline().rstrip('\r\n')

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(8<<20),b''):h.update(data)
    return h.hexdigest()

def download(part, start, end):
    env=os.environ.copy()
    env['LD_LIBRARY_PATH']=str(TOOL.parent.parent/'lib/x86_64-linux-gnu')
    proc=subprocess.Popen([str(TOOL)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,env=env)
    auth_timer=threading.Timer(30,proc.terminate);auth_timer.daemon=True;auth_timer.start()
    def send(line):proc.stdin.write((line+'\n').encode());proc.stdin.flush()
    def line():
        data=proc.stdout.readline()
        if not data:raise RuntimeError('FileZilla disconnected mid-event')
        return data.decode('utf8',errors='replace').rstrip('\r\n')
    expected_fingerprints=[]
    for row in (ROOT/'.known_hosts').read_text().splitlines():
        fields=row.split()
        if len(fields)>=3 and 'connect.nma1.seetacloud.com' in fields[0]:
            key=base64.b64decode(fields[2]);md5=hashlib.md5(key).hexdigest()
            expected_fingerprints += ['SHA256:'+base64.b64encode(hashlib.sha256(key).digest()).decode().rstrip('='),
                                      ':'.join(md5[i:i+2] for i in range(0,32,2))]
    assert expected_fingerprints,'Missing known source host key'
    state='start';verified=False;auth_count=0;last_quota=0.;last_report=time.monotonic();started=last_report
    last_size=part.stat().st_size if part.exists() else 0
    try:
        while True:
            b=proc.stdout.read(1)
            if not b:raise RuntimeError('FileZilla exited before transfer completion')
            event=b[0]-ord('0')
            if event in (6,7):continue  # Socket activity, no payload or newline.
            if event==16:
                delay=last_quota+2.-time.monotonic()
                if delay>0:time.sleep(delay)
                send('-02097152,1048576');last_quota=time.monotonic();continue
            if event==17:send('-1-');continue
            value=line()
            if event==25:
                verified=any(fp in value for fp in expected_fingerprints)
                if not verified:raise RuntimeError('Source host key does not match recorded key')
            elif event in (9,10):
                port=line()
                if not verified or value!='connect.nma1.seetacloud.com' or port!='14892':
                    raise RuntimeError('Unexpected SFTP host identity')
                send('n')  # FileZilla protocol: accept once; only y stores the key.
            elif event==12:
                auth_count+=1
                if not verified or auth_count>1:raise RuntimeError('Unexpected authentication retry')
                send(PW)
            elif event==0 and state=='start':
                if 'protocol_version=9' not in value:raise RuntimeError('Unsupported FileZilla handler protocol')
                send('open "root@connect.nma1.seetacloud.com" 14892');state='opening'
            elif event==1:
                if value!='1':raise RuntimeError('FileZilla command failed: '+value)
                if state=='opening':
                    auth_timer.cancel()
                    command='reget' if part.exists() and part.stat().st_size else 'get'
                    send(f'{command} "{SOURCE}" "{part}"');state='copying'
                    print('FILEZILLA_TRANSFER_STARTED',GROUP,flush=True)
                elif state=='copying':
                    send('quit');proc.wait(timeout=20)
                    print('FILEZILLA_TRANSFER_FINISHED',round(time.monotonic()-started,1),flush=True)
                    return
            elif event==2:raise RuntimeError('FileZilla error: '+value.replace(PW,'[redacted]'))
            if event==13 and state=='copying' and part.stat().st_size>=end:
                return
            if False and time.monotonic()-last_report>=25:
                size=part.stat().st_size if part.exists() else 0;now=time.monotonic()
                print('PROGRESS',json.dumps(dict(bytes=size,total=SIZE,percent=round(100*size/SIZE,2),
                    recent_MiB_s=round((size-last_size)/2**20/(now-last_report),3))),flush=True)
                last_report=now;last_size=size
    finally:
        auth_timer.cancel()
        if proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'clean_high30_v2_filezilla.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        final=ROOT/REL;part=final.with_name(final.name+'.filezilla.partial')
        final.parent.mkdir(parents=True,exist_ok=True)
        if final.exists():
            assert final.stat().st_size==SIZE and digest(final)==EXPECTED
            print('EXISTING_WEIGHT_VERIFIED',flush=True)
        else:
            spans=[(SIZE*i//16,SIZE*(i+1)//16) for i in range(16)]
            parts=[final.with_name(final.name+f'.fzrange16_{i}') for i in range(16)]
            for i in range(8):
                old=final.with_name(final.name+f'.fzrange{i}')
                if old.exists():
                    assert old.stat().st_size<=spans[2*i][1] and not parts[2*i].exists()
                    os.rename(old,parts[2*i])
            if part.exists() and not parts[0].exists():os.rename(part,parts[0])
            def worker(i):
                start,end=spans[i];piece=parts[i]
                if not piece.exists():
                    with piece.open('wb') as f:f.truncate(start)
                assert piece.stat().st_size>=start
                time.sleep(i*.5)
                for attempt in range(3):
                    try:
                        if piece.stat().st_size<end:download(piece,start,end)
                        assert piece.stat().st_size>=end
                        return
                    except Exception:
                        if attempt==2:raise
                        time.sleep(2)
            last_report=time.monotonic();last_bytes=0
            with ThreadPoolExecutor(max_workers=16) as pool:
                jobs=[pool.submit(worker,i) for i in range(16)]
                while True:
                    done,pending=wait(jobs,timeout=25,return_when=FIRST_EXCEPTION)
                    for f in done:f.result()
                    current=sum(min(end-start,max(0,p.stat().st_size-start)) if p.exists() else 0 for p,(start,end) in zip(parts,spans))
                    now=time.monotonic()
                    print('PROGRESS',json.dumps(dict(bytes=current,total=SIZE,percent=round(100*current/SIZE,2),recent_MiB_s=round((current-last_bytes)/2**20/max(.001,now-last_report),3),active=len(pending))),flush=True)
                    last_report=now;last_bytes=current
                    if not pending:break
            print('MERGING_ON_LAB_SERVER',flush=True)
            with part.open('wb') as out:
                for piece,(start,end) in zip(parts,spans):
                    with piece.open('rb') as inp:
                        inp.seek(start);left=end-start
                        while left:
                            data=inp.read(min(8<<20,left))
                            assert data,'Unexpected end of range'
                            out.write(data);left-=len(data)
            assert part.stat().st_size==SIZE
            print('VERIFYING_SHA256',flush=True)
            assert digest(part)==EXPECTED,'SHA256 mismatch; partial retained'
            os.rename(part,final)
            for piece in parts:piece.unlink()
        now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
        result=dict(status='verified',completed_at=now,group=GROUP,G=160,fake=320,
          source_server='connect.nma1.seetacloud.com:14892',source=SOURCE,
          destination_server='172.18.33.60',destination=str(final),bytes=SIZE,sha256=EXPECTED,
          all_sha256_verified=True,tool='FileZilla 3.46.3 fzsftp (protocol 9), sixteen parallel range downloads',
          scope='one final BF16 student inference weight only; no configs, logs, images or optimizer state',
          source_deleted=False,local_weight_copy=False)
        note='clean全局高噪声先验v2最终学生权重备份记录\n'+now+'\n'
        note+='实验：'+GROUP+'；G160/F320\n源服务器：'+result['source_server']+'\n源路径：'+SOURCE+'\n'
        note+='目标服务器：172.18.33.60\n目标路径：'+str(final)+'\n'
        note+='工具：服务器现有FileZilla的fzsftp组件；服务器直传，不经过本机。\n'
        note+='字节数：'+str(SIZE)+'\nSHA256：'+EXPECTED+'\n源/目标SHA256一致，已核对。\n'
        note+='范围：仅最终BF16学生推理权重；未传配置、源码、训练日志、样图或优化器。\n源权重保留；未操作训练进程。\n'
        (ROOT/'clean_high30_v2_备份记录.txt').write_text(note,encoding='utf8')
        (ROOT/'clean_high30_v2_COMPLETE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
        print('BACKUP_COMPLETE',json.dumps(result,ensure_ascii=False),flush=True)

def repair_stalled_range():
    final=ROOT/REL
    if final.exists():return
    spans=[(SIZE*i//16,SIZE*(i+1)//16) for i in range(16)]
    pieces=[final.with_name(final.name+f'.fzrange16_{i}') for i in range(16)]
    missing=[i for i,(p,(a,b)) in enumerate(zip(pieces,spans)) if p.stat().st_size<b]
    if not missing:return
    assert missing==[14],missing
    a,b=spans[14];part=pieces[14]
    assert part.stat().st_size==a
    subranges=[(a+(b-a)*i//8,a+(b-a)*(i+1)//8) for i in range(8)]
    parts=[final.with_name(final.name+f'.fzrepair14_{i}') for i in range(8)]
    def worker(i):
        lo,hi=subranges[i];p=parts[i]
        if not p.exists():
            with p.open('wb') as f:f.truncate(lo)
        time.sleep(i*.5)
        for attempt in range(3):
            try:
                if p.stat().st_size<hi:download(p,lo,hi)
                assert p.stat().st_size>=hi
                return
            except Exception:
                if attempt==2:raise
                time.sleep(2)
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs=[pool.submit(worker,i) for i in range(8)]
        while True:
            done,pending=wait(jobs,timeout=25,return_when=FIRST_EXCEPTION)
            for f in done:f.result()
            completed=sum(min(hi-lo,max(0,p.stat().st_size-lo)) if p.exists() else 0 for p,(lo,hi) in zip(parts,subranges))
            print('REPAIR_PROGRESS',json.dumps(dict(percent=round(100*completed/(b-a),2),bytes=completed,active=len(pending))),flush=True)
            if not pending:break
    merged=part.with_name(part.name+'.repair_complete')
    with merged.open('wb') as out:
        out.truncate(a);out.seek(a)
        for p,(lo,hi) in zip(parts,subranges):
            with p.open('rb') as inp:
                inp.seek(lo);left=hi-lo
                while left:
                    data=inp.read(min(8<<20,left));assert data
                    out.write(data);left-=len(data)
    assert merged.stat().st_size==b
    os.replace(merged,part)
    for p in parts:p.unlink()
    print('REPAIRED_RANGE_14',flush=True)

if __name__=='__main__':
    repair_stalled_range()
    main()

