import hashlib,json,re,sys
from pathlib import Path,PurePosixPath
from archive_abc_to_lab import arms,base,connect,local,record,run
password=sys.stdin.readline().strip()
raw=Path('C:/Users/sherl/Desktop/login.txt').read_text(encoding='utf-8-sig')
c=re.search(r'IP:\s*172\.18\.33\.60\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)
src=connect('connect.nma1.seetacloud.com',14892,'root',password)
dst=connect('172.18.33.60',22,c[1],c[2].strip())
try:
    with src.open_sftp() as cloud,dst.open_sftp() as lab:
        for arm,(runname,_) in arms.items():
            origin=PurePosixPath(json.loads((local/'completed'/f'checkpoint_validation_{runname}.json').read_text())['file']).parent
            recv=f'{base}/{arm}/weights/{origin.name}'
            run(dst,f'umask 077; mkdir -p {recv}')
            for n in ['shapes_rank0.json','complete.json']:
                with cloud.file(str(origin/n),'rb') as f:data=f.read()
                obj=json.loads(data)
                if n=='shapes_rank0.json':assert len(obj)==1680
                else:assert obj['student_updates']==160 and obj['world_size']==1
                with lab.file(recv+'/'+n+'.tmp','wb') as f:f.write(data)
                digest=hashlib.sha256(data).hexdigest()
                assert run(dst,f'mv {recv}/{n}.tmp {recv}/{n}; sha256sum {recv}/{n}').split()[0]==digest
            run(dst,f'cd {recv}; if ! test -L student_rank0.pt; then ln -s ../../student_rank0.pt student_rank0.pt; fi; test "$(readlink student_rank0.pt)" = ../../student_rank0.pt')
            print('CHECKPOINT METADATA AND RELATIVE LINK VERIFIED',arm,recv)
    msg='实验室评测程序使用的 complete.json 和 shapes_rank0.json 已逐文件校验。可传给评测程序的 checkpoint 目录为各组 <组>/weights/*_g000160_end/，其中学生文件引用各组唯一的 student_rank0.pt。'
    if msg not in record.read_text(encoding='utf8'):
        import time
        with record.open('a',encoding='utf8') as f:f.write(time.strftime('%Y-%m-%d %H:%M:%S')+'  '+msg+'\n')
finally:dst.close();src.close()
