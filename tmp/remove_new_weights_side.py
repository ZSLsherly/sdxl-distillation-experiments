"""Remove exactly one transferred source, only after re-reading proof from the lab host."""
import os,sys,re,json,shlex
from pathlib import Path
from archive_abc_to_lab import connect,run
sources={
'C_mix1_G160':('bc_winner_late_mix_20261102/late_mix','20261103T074022+0800_g000160_end','391337f02679006b7cb76c292eb39236e84496fda40183fadb88fad361e42aed'),
'D_G9':('interval_D_20261103/D','20261103T105931+0800_g000009_stopped','60de01cc02225bbe7f6b336d4947e3744b68578df1b6145e120c5beafbf9d3a5')}
arm=sys.argv[1];d,stamp,sha=sources[arm]
suffix='/weights/'+stamp+'/student_rank0.pt'
dst='/data5/shenglin.zeng/checkpoints/a800_20261103/'+arm
src='/root/autodl-tmp/sdxl_runs/outputs/'+d
pw=sys.stdin.readline().strip()
raw=Path('C:/Users/sherl/Desktop/login.txt').read_text(encoding='utf-8-sig')
m=re.search(r'IP:\s*172\.18\.33\.60\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)
cloud=connect('connect.nma1.seetacloud.com',14892,'root',pw)
lab=connect('172.18.33.60',22,m[1],m[2].strip())
try:
    q=shlex.quote
    assert run(lab,'cat '+q(dst+'/VERIFIED.txt'))==f'{arm} 5135506134 {sha} {dst+suffix}'
    parts=run(lab,'stat -c %s '+q(dst+suffix)+'; sha256sum '+q(dst+suffix)).split()
    assert parts[:2]==['5135506134',sha]
    for n in ['/config.json','/status.json','/weights/'+stamp+'/shapes_rank0.json','/weights/'+stamp+'/complete.json']:
        assert run(lab,'sha256sum '+q(dst+n)).split()[0]==run(cloud,'sha256sum '+q(src+n)).split()[0]
    cmd="""import os,sys,hashlib
from pathlib import Path
p=Path(sys.argv[1]); assert p.resolve(strict=True)==p and p.is_relative_to('/root/autodl-tmp/sdxl_runs/outputs') and p.name=='student_rank0.pt'
with p.open('rb') as f:
 s=os.fstat(f.fileno());assert s.st_size==5135506134 and hashlib.file_digest(f,'sha256').hexdigest()==sys.argv[2]
 t=p.stat();assert (s.st_dev,s.st_ino,s.st_mtime_ns)==(t.st_dev,t.st_ino,t.st_mtime_ns)
 p.unlink()
assert not p.exists()
"""
    run(cloud,'/root/sdxl_research/env/bin/python -c '+q(cmd)+' '+q(src+suffix)+' '+q(sha))
    print('VERIFIED LAB, REMOVED SOURCE',arm,sha,flush=True)
finally:cloud.close();lab.close()
