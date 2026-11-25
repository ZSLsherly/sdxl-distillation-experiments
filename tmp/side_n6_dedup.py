from backup_a800_nov5 import cloud,lab,run
from abc_direct_grant import store
from pathlib import Path
import json
stage='/root/sdxl_research/tmp/side_archive_n6';dest='/data5/shenglin.zeng/checkpoints/a800_archive_20261106'
with cloud() as s,lab() as d:
 src="""import json,hashlib
from pathlib import Path
r=Path('/root/autodl-tmp/sdxl_runs');m=json.loads(Path(STAGE+'/manifest.json').read_text());out={}
for rel in m['large']:
 p=r/rel;n,h=m['records'][rel]
 with p.open('rb') as f:
  head=f.read(1<<20);f.seek(max(0,n-(1<<20)));tail=f.read()
 out[rel]={'size':n,'sha256':h,'edge':hashlib.sha256(head+tail).hexdigest()}
print(json.dumps(out))
""".replace('STAGE',repr(stage))
 store(s,stage+'/edge_inventory.py',src,0o700)
 targets=json.loads(run(s,'/root/sdxl_research/env/bin/python '+stage+'/edge_inventory.py'))
 scan="""import os,json,hashlib,tarfile
from pathlib import Path
targets=TARGETS;root=Path('/data5/shenglin.zeng');exclude=Path(EXCLUDE);matches={k:[] for k in targets};candidates=0;archives=[]
def edge(f,n):
 a=f.read(1<<20);f.seek(max(0,n-(1<<20)));return hashlib.sha256(a+f.read()).hexdigest()
for base,dirs,files in os.walk(root):
 dirs[:]=[x for x in dirs if Path(base,x)!=exclude and x not in ['.git','__pycache__']]
 for name in files:
  p=Path(base,name)
  if p.is_symlink():continue
  try:n=p.stat().st_size
  except OSError:continue
  if name.endswith('.tar') and n>=min(v['size'] for v in targets.values()):archives.append(p)
  ks=[k for k,v in targets.items() if v['size']==n]
  if not ks:continue
  candidates+=1
  with p.open('rb') as f:e=edge(f,n)
  for k in ks:
   if targets[k]['edge']!=e:continue
   h=hashlib.sha256()
   with p.open('rb') as f:
    for b in iter(lambda:f.read(4<<20),b''):h.update(b)
   if h.hexdigest()==targets[k]['sha256']:matches[k].append(str(p))
for p in archives:
 with tarfile.open(p,'r:') as tar:
  for member in tar:
   ks=[k for k,v in targets.items() if v['size']==member.size]
   if not member.isfile() or not ks:continue
   candidates+=1
   with tar.extractfile(member) as f:e=edge(f,member.size)
   for k in ks:
    if targets[k]['edge']!=e:continue
    h=hashlib.sha256()
    with tar.extractfile(member) as f:
     for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    if h.hexdigest()==targets[k]['sha256']:matches[k].append(str(p)+'::'+member.name)
out={'scope':str(root),'excluded_current_transfer':str(exclude),'same_size_candidates_checked':candidates,'tar_archives_checked':list(map(str,archives)),'matches':matches,'targets':targets}
Path(EXCLUDE+'/dedup_checked.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))
""".replace('TARGETS',repr(targets)).replace('EXCLUDE',repr(dest))
 store(d,dest+'/.pull6/dedup.py',scan,0o700)
 result=run(d,'nice -n 15 python3 '+dest+'/.pull6/dedup.py')
 Path('output/sdxl-crossdistill/a800_archive_20261106/dedup_checked.json').write_text(result,encoding='utf-8')
 print(result)
 print(run(d,'cat /data5/shenglin.zeng/tools/filezilla/src/filezilla-3.46.3/src/putty/fzprintf.h'))
