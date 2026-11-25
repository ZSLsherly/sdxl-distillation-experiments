from pathlib import Path
from collections import defaultdict
import re,json,hashlib
root=Path('/root/autodl-tmp/sdxl_runs/outputs')
groups=defaultdict(list)
for p in root.rglob('*.png'):
 rel=p.relative_to(root)
 if 'eval' not in rel.parts:continue
 assert p.parent.parent.name=='eval' and p.resolve()==p and not p.is_symlink(),rel
 m=re.fullmatch(r'.*_g(\d+)_.*',p.parent.name);assert m,rel
 groups[str(p.parent.parent.relative_to(root))].append((int(m[1]),p))
out={}
for group,items in groups.items():
 last=max(g for g,p in items)
 saved={};sources={}
 for g,p in sorted(items):
  k=str(p.relative_to(root));s=p.stat();sources[k]=[s.st_size,s.st_mtime_ns]
  if g in (0,8,40,80,118,120,160,last):
   saved[k]=hashlib.sha256(p.read_bytes()).hexdigest()
   meta=p.parent/'rank0.json';mk=str(meta.relative_to(root))
   if mk not in saved:saved[mk]=hashlib.sha256(meta.read_bytes()).hexdigest()
   rec=json.loads(meta.read_text());assert any(x['image']==p.name and 'seed' in x and 'prompt' in x for x in rec['samples'])
 out[group]={'save':saved,'sources':sources}
print(json.dumps(out))
