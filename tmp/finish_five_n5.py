from backup_a800_nov5 import cloud,lab,report,ROOT,LAB,run,q,arms
from pathlib import PurePosixPath
import json,time,hashlib
private=LAB+'/.pull5'
with cloud() as src,lab() as dst:
    m=json.loads(run(dst,'cat '+q(LAB+'/manifest.json')))
    proof={}
    while len(proof)<5:
        verified=run(dst,'test ! -f '+LAB+'/verified_large.txt || cat '+LAB+'/verified_large.txt')
        for line in verified.splitlines():
            h,rel=line.split('  ',1);arm=rel.split('/')[1]
            if arm in proof:continue
            assert arm in arms and m['records'][rel]==[5135506134,h]
            # Make the associated small files usable in the same directory as the weight.
            with src.open_sftp() as s,dst.open_sftp() as d:
                for k,(n,digest) in m['records'].items():
                    if k.split('/')[1]!=arm or k==rel:continue
                    with s.file(ROOT+'/'+k,'rb') as f:b=f.read()
                    assert len(b)==n and hashlib.sha256(b).hexdigest()==digest
                    with d.file(LAB+'/'+k+'.tmp','wb') as f:f.write(b)
                    assert run(dst,'sha256sum '+q(LAB+'/'+k+'.tmp')).split()[0]==digest
                    d.posix_rename(LAB+'/'+k+'.tmp',LAB+'/'+k)
            proof[arm]=dict(src=ROOT+'/'+rel,dest=LAB+'/'+rel,sha256=h,size=5135506134)
            tmp=report/'weights.tmp';tmp.write_text(json.dumps(proof,indent=2));tmp.replace(report/'weights.json')
            print('RECEIPT',arm,flush=True)
        if len(proof)<5: time.sleep(30)
    while not run(dst,'test ! -s '+LAB+'/COMPLETE.json || echo complete'):time.sleep(10)
    run(src,'/root/sdxl_research/env/bin/python /root/sdxl_research/tmp/five_checkpoints_n5/revoke.py')
    run(dst,'unlink '+private+'/key')
    # Known superseded relay's partial file only; complete weight has already passed SHA.
    old=proof[arms[0]]['dest']+'.partial'
    run(dst,'test ! -f '+q(old)+' || unlink '+q(old))
    print('COMPLETE; temporary transfer access revoked',flush=True)
