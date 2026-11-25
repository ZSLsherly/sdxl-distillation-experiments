import sys,time,shlex,re
from pathlib import Path
from archive_abc_to_lab import arms,base,record,connect,run
from abc_direct_grant import lab_dir,source_dir
def main():
    password=sys.stdin.readline().strip()
    raw=Path('C:/Users/sherl/Desktop/login.txt').read_text(encoding='utf-8-sig')
    creds=re.search(r'IP:\s*172\.18\.33\.60\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)
    while True:
        lab=None; cloud=None
        try:
            lab=connect('172.18.33.60',22,creds[1],creds[2].strip())
            results=[]
            for arm,(_,sha) in arms.items():
                target=f'{base}/{arm}/student_rank0.pt'
                proof=run(lab,f'if test -f {base}/{arm}/VERIFIED.txt; then cat {base}/{arm}/VERIFIED.txt; fi')
                if proof:
                    if proof!=f'{arm} 5135506134 {sha} {target}': raise ValueError('Unexpected receipt: '+arm)
                    results.append(arm)
                    append(f'{arm} 目标端文件大小及 SHA-256 校验通过，{target}，SHA-256 {sha}。')
            if len(results)==3:
                cloud=connect('connect.nma1.seetacloud.com',14892,'root',password)
                run(cloud,f'/root/sdxl_research/env/bin/python {source_dir}/revoke.py')
                run(lab,f'rm -f -- {lab_dir}/id {lab_dir}/id.pub')
                append('A、B、C 三份备份均已验证，临时直传授权已撤销。')
            # Sync even if a previous attempt appended locally but did not publish the TXT.
            with lab.open_sftp() as sf:
                sf.put(str(record),base+'/EXPERIMENTS.txt.new')
            run(lab,f'mv -f {base}/EXPERIMENTS.txt.new {base}/EXPERIMENTS.txt')
            print('VERIFIED',results,flush=True)
            if len(results)==3:return
            state=run(lab,f'tail -3 {lab_dir}/direct.log')
            print(state,flush=True)
            if 'Traceback' in state or 'Error:' in state: raise ValueError('direct job failed: '+state)
            if int(run(lab,'date +%s'))>=int(run(lab,f'cat {lab_dir}/expiry')):raise ValueError('grant expired before completion')
        except (OSError,EOFError) as e:print('Transient connection failure:',type(e).__name__,flush=True)
        finally:
            if lab:lab.close()
            if cloud:cloud.close()
        time.sleep(180)
def append(entry):
    if entry not in record.read_text(encoding='utf8'):
        with record.open('a',encoding='utf8') as f:
            f.write(time.strftime('%Y-%m-%d %H:%M:%S')+'  '+entry+'\n')
if __name__=='__main__':main()
