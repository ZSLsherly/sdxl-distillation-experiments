from pathlib import Path
import os,hashlib,json
root=Path('/root/autodl-tmp/sdxl_runs');stage=Path(__file__).parent
live={'A_pcm10_terminal_patchgan','A_mid_clean_x0'}
folders=[p for p in (root/'outputs').iterdir() if p.is_dir() and p.name not in live and not p.is_symlink()]
folders +=[root/'outputs/dm1_mid4_uniform_mix1_g160/model']
folders += [root/'datasets',*root.glob('semantic_*'),root/'bc_texture_inputs/reference_features']
items={}
for folder in folders:
 for dp,sub,files in os.walk(folder,followlinks=False):
  for f in files:
   p=Path(dp,f)
   if p.is_symlink() or not p.is_file(): continue
   rel=str(p.relative_to(root))
   if '\n' in rel or rel.startswith('-'):raise ValueError(rel)
   items[rel]=p.stat().st_size
large=[];groups=[];batch=[];size=0
for path,n in sorted(items.items()):
 if n>128*2**20:large.append((path,n));continue
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
(stage/'manifest.json').write_text(json.dumps(dict(records=records,large=[k for k,n in large],groups=groups)))

