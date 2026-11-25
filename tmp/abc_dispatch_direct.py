import sys,json,shlex,re,time
from pathlib import Path
from abc_direct_grant import source_dir,lab_dir,store
from archive_abc_to_lab import arms,base,connect,local,run
pwd=sys.stdin.readline().strip()
raw=Path('C:/Users/sherl/Desktop/login.txt').read_text(encoding='utf-8-sig')
cred=re.search(r'IP:\s*172\.18\.33\.60\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)
cloud=connect('connect.nma1.seetacloud.com',14892,'root',pwd);lab=connect('172.18.33.60',22,cred[1],cred[2].strip())
try:
    with cloud.open_sftp() as sf:
        script=sf.file(source_dir+'/read.py','r').read().decode()
    old="""    m=re.fullmatch(r'([ABC]) ([0-9]{1,12})',cmd)
    if not m:raise SystemExit(77)
    f=files[m[1]];off=int(m[2])
    if off>os.stat(f).st_size:raise SystemExit(77)
    args=['tail','-c','+'+str(off+1),'--',f]"""
    new="""    m=re.fullmatch(r'([ABC]) ([0-9]{1,12}) ([0-9]{1,9})',cmd)
    if not m:raise SystemExit(77)
    f=files[m[1]];off=int(m[2]);length=int(m[3])
    if length>67108864 or off+length>os.stat(f).st_size:raise SystemExit(77)
    args=['dd','if='+f,'iflag=count_bytes,skip_bytes,fullblock','bs=1M','skip='+str(off),'count='+str(length),'status=none']"""
    if old in script:script=script.replace(old,new)
    assert new in script
    store(cloud,source_dir+'/read.next',script,0o700)
    run(cloud,f'mv {source_dir}/read.next {source_dir}/read.py')
    expiry=re.search(r'remaining=(\d+)-',script)[1]
    store(lab,lab_dir+'/expiry',expiry,0o600)
    manifest={}
    for a,(arm,sha) in arms.items():
        run(lab,f'umask 077; mkdir -p {base}/{a}')
        val=local/'completed'/f'checkpoint_validation_{arm}.json'
        for p,name in [(val,'source_validation.json'),*((local/'completed'/arm/n,n) for n in ('config.json','status.json'))]:
            store(lab,base+'/'+a+'/'+name,p.read_text(encoding='utf8'),0o600)
        manifest[a]={'size':json.loads(val.read_text())['bytes'],'sha256':sha}
    store(lab,lab_dir+'/files.json',json.dumps(manifest),0o600)
    store(lab,lab_dir+'/copy.py',Path('tmp/abc_lab_copy.py').read_text(encoding='utf-8-sig'),0o700)
    print(run(lab,f'nohup python3 -u {lab_dir}/copy.py </dev/null >>{lab_dir}/direct.log 2>&1 & echo DIRECT_PID=$!'),flush=True)
finally:
    cloud.close();lab.close()
