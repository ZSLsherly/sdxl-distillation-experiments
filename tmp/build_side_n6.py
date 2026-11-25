from pathlib import Path
import os,hashlib,json
root=Path('/root/autodl-tmp/sdxl_runs');stage=Path(__file__).parent
folders=['outputs/A_pcm10_terminal_patchgan','outputs/A_mid_clean_x0','datasets/parallel_20k_seed20261018','datasets/20k_sources']
folders += ['tmp/'+x+'/code' for x in ['A_pcm10_terminal_patchgan','A_mid_clean_x0'] if (root/'tmp'/x/'code').is_dir()]
items={}
for folder in folders:
 for base,dirs,files in os.walk(root/folder,followlinks=True):
  for name in files:
   p=Path(base)/name
   resolved=p.resolve()
   assert resolved.is_relative_to(root) or resolved.is_relative_to('/root/a10_patchgan_aux'),str(p)
   assert p.is_file(),str(p)
   items[str(p.relative_to(root))]=p.stat().st_size
assert len([k for k in items if k.endswith('student_rank0.pt')])==2
large=[];groups=[];batch=[];size=0
for path,n in sorted(items.items()):
 if n>128*2**20:large.append(path);continue
 if size+n>128*2**20:groups.append(batch);batch=[];size=0
 batch.append(path);size+=n
if batch:groups.append(batch)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as fp:
  for b in iter(lambda:fp.read(4*2**20),b''):h.update(b)
 return h.hexdigest()
records={k:[n,digest(root/k)] for k,n in items.items()}
for i,group in enumerate(groups):
 (stage/f'list{i}').write_bytes(b'\0'.join(k.encode() for k in group)+b'\0')
(stage/'manifest.json').write_text(json.dumps(dict(records=records,large=large,groups=groups)))
print(json.dumps(dict(files=len(records),bytes=sum(items.values()),large=large)))
