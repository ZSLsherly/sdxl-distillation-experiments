import os,re,subprocess,sys,time
from pathlib import Path
from archive_abc_to_lab import connect,run
pw=sys.stdin.readline().strip()
raw=Path('C:/Users/sherl/Desktop/login.txt').read_text(encoding='utf-8-sig')
m=re.search(r'IP:\s*172\.18\.33\.60\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw);lu,lp=m[1],m[2].strip();del raw,m
for arm in ('C_mix1_G160','D_G9'):
    receipt=f'/data5/shenglin.zeng/checkpoints/a800_20261103/{arm}/VERIFIED.txt'
    while True:
        lab=None
        try:
            lab=connect('172.18.33.60',22,lu,lp)
            if run(lab,f'if test -f {receipt}; then echo found; fi')=='found':break
        except (OSError,EOFError) as e:print('WAIT',arm,type(e).__name__,flush=True)
        finally:
            if lab:lab.close()
        time.sleep(90)
    subprocess.run([sys.executable,'-u','tmp/remove_new_weights_side.py',arm],input=(pw+'\n').encode(),check=True)
    print('ARCHIVED_AND_SOURCE_REMOVED',arm,time.strftime('%Y-%m-%d %H:%M:%S'),flush=True)
