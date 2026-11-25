from pathlib import Path
import hashlib,json,tarfile,os,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from archive_results_n9 import cloud,lab,run,q,root,N,transfer
local=Path('output/sdxl-crossdistill/a800_archive_20261109');local.mkdir(exist_ok=True)
root=root+'outputs';dest=N+'/progress_selected'
with cloud() as s:
 plan=json.loads(run(s,'/root/sdxl_research/env/bin/python -c '+q(Path('tmp/select_progress_n9.py').read_text(encoding='utf-8-sig'))))
(local/'training_progress_plan.json').write_text(json.dumps(plan,indent=1))
ns=sum(len(v['sources']) for v in plan.values());nk=sum(sum(p.endswith('.png') for p in v['save']) for v in plan.values())
print('planned',len(plan),ns,nk,flush=True)
def backup(item):
 group,rec=item;tar=dest+'/'+group.replace('/','__')+'.tar'
 with cloud() as s,lab() as d:
  run(d,'mkdir -p '+q(dest))
  if run(d,'test -f '+q(tar)+' && echo yes || true')!='yes':
   cmd='cd '+q(root)+' && tar -cf - -- '+' '.join(map(q,rec['save']))
   transfer(s,d,cmd,'umask 077; cat > '+q(tar+'.partial'))
   tmp=tar+'.partial'
  else: tmp=tar
  reader='import sys,tarfile,hashlib,json;t=tarfile.open(sys.argv[1]);print(json.dumps({m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t if m.isfile()}))'
  assert json.loads(run(d,'python3 -c '+q(reader)+' '+q(tmp)))==rec['save'],group
  if tmp!=tar:run(d,'mv -T -- '+q(tmp)+' '+q(tar))
  print('verified',group,len(rec['save']),flush=True)
with ThreadPoolExecutor(8) as pool:
 for f in as_completed([pool.submit(backup,i) for i in plan.items()]):f.result()
# Remove only the enumerated evaluation PNGs. Abort on any changed file before deleting.
src_cleanup="""from pathlib import Path
import sys,json,hashlib
root=Path('/root/autodl-tmp/sdxl_runs/outputs')
groups=json.load(sys.stdin);paths=[]
for group,r in groups.items():
 for rel,stat in r['sources'].items():
  p=root/rel
  assert p.resolve()==p and root in p.parents and p.suffix=='.png' and p.parent.parent==root/group and p.parent.parent.name=='eval'
  s=p.stat();assert [s.st_size,s.st_mtime_ns]==stat
  if rel in r['save']:assert hashlib.sha256(p.read_bytes()).hexdigest()==r['save'][rel]
  paths.append(p)
for p in paths:p.unlink()
assert all(not p.exists() for p in paths)
print(len(paths))
"""
with cloud() as s,lab() as d:
 body=json.dumps(plan).encode()
 with d.open_sftp() as ftp:
  with ftp.open(dest+'/manifest.json.tmp','wb') as w:w.write(body)
 run(d,'mv -T -- '+q(dest+'/manifest.json.tmp')+' '+q(dest+'/manifest.json'))
 assert run(d,'sha256sum -- '+q(dest+'/manifest.json')).split()[0]==hashlib.sha256(body).hexdigest()
 w,b,e=s.exec_command('/root/sdxl_research/env/bin/python -c '+q(src_cleanup));w.channel.sendall(body);w.channel.shutdown_write()
 count=b.read().decode().strip();err=e.read().decode();assert b.channel.recv_exit_status()==0,err;assert int(count)==ns
 check=run(s,'df -h /root/autodl-tmp; find '+q(root)+' -type f -name "*.png" | wc -l')
 receipt={'progress_images_removed':ns,'images_backed_up':nk,'intermediate_discarded':ns-nk,'freed_bytes':sum(st[0] for r in plan.values() for st in r['sources'].values()),'groups':len(plan),'lab':dest,'final_check':check}
 (local/'进度图清理结果.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False),encoding='utf-8')
 print('COMPLETE',receipt,flush=True)
