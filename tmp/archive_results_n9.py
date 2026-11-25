from backup_a800_nov5 import cloud,lab,run,q
from pathlib import Path,PurePosixPath
import json,hashlib,time,threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
out=Path('output/sdxl-crossdistill/a800_archive_20261109');out.mkdir(exist_ok=True)
D='/data5/shenglin.zeng/checkpoints/'
N=D+'a800_archive_20261109'
root='/root/autodl-tmp/sdxl_runs/'
lock=threading.Lock()
def note(msg):
 with lock:
  with (out/'清理记录.txt').open('a',encoding='utf-8') as f:f.write(datetime.now().isoformat(timespec='seconds')+' '+msg+'\n')
  print(msg,flush=True)
def digest(c,p):return run(c,'sha256sum -- '+q(p)).split()[0]
def clean(c,p,h,dst):
 assert p.startswith((root+'outputs/',root+'tmp/side_archive_n6/groups/','/root/dm1_mid4_uniform_mix1_model/','/root/a10_patchgan_aux/'))
 assert run(c,'realpath -- '+q(p))==p
 cmd='test -f '+q(p)+' && test "$(sha256sum -- '+q(p)+' | cut -c1-64)" = '+q(h)+' && unlink -- '+q(p)
 run(c,cmd);note('已删除 '+p+' | SHA256 '+h+' | 实验室 '+dst)
def weight(src):
 with cloud() as s,lab() as d:
  h=digest(s,src); name=src.split('/weights/')[0].split('/')[-2]
  rel=src.split('/outputs/')[-1] if '/outputs/' in src else 'dm1_mid4_uniform_mix1_g160/model/weights/'+src.split('/weights/')[1]
  targets=[D+'a800_archive_20261105/outputs/'+rel,D+'a800_archive_20261106/outputs/'+rel,N+'/outputs/'+rel]
  target=next((x for x in targets if run(d,'test ! -f '+q(x)+' || echo yes')=='yes' and digest(d,x)==h),None)
  if not target:
   partial=targets[1]+'.fzpart'
   target=targets[2];run(d,'mkdir -p '+q(str(PurePosixPath(target).parent)))
   for candidate in (targets[1].replace('.pt','.direct-part'),partial):
    if run(d,'test ! -f '+q(candidate)+' || echo yes')=='yes' and digest(d,candidate)==h:
     run(d,'ln -- '+q(candidate)+' '+q(target));break
   else:
    tmp=target+'.partial'
    if run(d,'test ! -f '+q(tmp)+' || echo yes')!='yes' and run(d,'test ! -f '+q(partial)+' || echo yes')=='yes':
     run(d,'cp -- '+q(partial)+' '+q(tmp))
    n=int(run(d,'stat -c %s '+q(tmp)+' 2>/dev/null || echo 0'))
    if n and run(s,'head -c '+str(n)+' -- '+q(src)+' | sha256sum').split()[0]!=digest(d,tmp):raise RuntimeError('partial mismatch: '+tmp)
    note('开始续传 '+src+' 已有字节 '+str(n))
    transfer(s,d,'tail -c +'+str(n+1)+' -- '+q(src),'cat >> '+q(tmp))
    assert digest(d,tmp)==h;run(d,'mv -T -- '+q(tmp)+' '+q(target))
  assert digest(d,target)==h
  clean(s,src,h,target)
def transfer(s,d,src,dst):
 _,a,e=s.exec_command(src);w,b,f=d.exec_command('umask 077; '+dst)
 while chunk:=a.read(1<<20):w.channel.sendall(chunk)
 w.channel.shutdown_write()
 if a.channel.recv_exit_status() or b.channel.recv_exit_status():raise IOError('transfer failed')
def archives():
 with cloud() as s,lab() as d:
  # Previous staging files correspond to experiment-results tar archives only.
  groups=root+'tmp/side_archive_n6/groups/'
  for p in run(s,'find '+q(groups)+' -maxdepth 1 -type f -name "part_*.tar"').splitlines():
   dest=D+'a800_archive_20261106/evidence/'+p.rsplit('/',1)[1]
   if run(d,'test ! -f '+q(dest)+' || echo yes')=='yes':
    h=digest(d,dest)
    if digest(s,p)==h:clean(s,p,h,dest)
  dirs=run(s,'find '+q(root+'outputs')+' -mindepth 1 -maxdepth 1 -type d').splitlines()+['/root/dm1_mid4_uniform_mix1_model','/root/a10_patchgan_aux']
  for path in dirs:
   arm=path.rsplit('/',1)[1]
   if arm=='C_mix1_pcm_dm4_fake4_low30_g160':continue
   # Include provenance and results in the backup; release bulky generated artifacts only.
   rows=run(s,'cd '+q(path)+' && find . -type f ! -name student_rank0.pt -print').splitlines()
   bulk=[r for r in rows if r.endswith(('.npz','.npy','.png','.jpg','.jpeg','.webp'))]
   if not bulk:continue
   assert all(r.startswith('./') and '\n' not in r for r in rows)
   dest=N+'/results/'+arm+'.tar';run(d,'mkdir -p '+q(N+'/results'))
   cmd='cd '+q(path)+' && find . -type f ! -name student_rank0.pt -print0 | tar --null -T - -cf -'
   exists=run(d,'test ! -f '+q(dest)+' || echo yes')=='yes'
   if not exists:transfer(s,d,cmd,'cat > '+q(dest+'.partial'))
   # Verify each result against archive contents before releasing its source.
   script="import sys,tarfile,hashlib,json; t=tarfile.open(sys.argv[1]); print(json.dumps({a.name:hashlib.sha256(t.extractfile(a).read()).hexdigest() for a in t if a.isfile()}))"
   hashes=json.loads(run(d,'python3 -c '+q(script)+' '+q(dest if exists else dest+'.partial')))
   assert set(rows)<=set(hashes)
   if not exists:run(d,'mv -- '+q(dest+'.partial')+' '+q(dest))
   # Recheck source on-host; write full per-file receipts to the laboratory.
   plan={r:hashes[r] for r in bulk}; body=json.dumps(plan)
   now='import pathlib,json,sys,hashlib; root=pathlib.Path(sys.argv[1]).resolve(); m=json.load(sys.stdin); exec("for r,h in m.items():\\n p=root/r; assert p.resolve()==p and root in p.parents and p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==h; p.unlink()")'
   w,b,e=s.exec_command('/root/sdxl_research/env/bin/python -c '+q(now)+' '+q(path));w.channel.sendall(body.encode());w.channel.shutdown_write()
   if b.channel.recv_exit_status():raise RuntimeError(e.read().decode())
   (out/(arm+'.json')).write_text(body,encoding='utf-8')
   note(f'已归档 {path}，删除其中 {len(bulk)} 个已校验图像/特征文件，逐文件清单 {arm}.json | {dest}')
if __name__=='__main__':
 with cloud() as s:
  src=run(s,'find /root -type f -name student_rank0.pt').splitlines()
  assert all(p.startswith((root+'outputs/','/root/dm1_mid4_uniform_mix1_model/')) for p in src)
 # Free the system disk before attempting new transfers.
 for system_weight in (p for p in src if '/autodl-tmp/' not in p):weight(system_weight)
 with ThreadPoolExecutor(4) as pool:
  jobs=[pool.submit(weight,p) for p in src if '/autodl-tmp/' in p]+[pool.submit(archives)]
  for j in jobs:j.result()
 with cloud() as s:note('完成 '+run(s,'df -h /root /root/autodl-tmp'))
