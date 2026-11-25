from pathlib import Path
import os,hashlib,json
root=Path('/root/autodl-tmp/sdxl_runs');stage=Path(__file__).parent
arms=['c_mix1_dm234_fake3_20261103','c_mix1_first_dm_fake3_g160','c_mix1_pcm_dm4_matched_time','A_pcm10_g160','dm1_mid4_uniform_mix1_g160']
items={}
for arm in arms:
 model=root/'outputs'/arm/'model'
 w,=model.glob('weights/*/student_rank0.pt')
 files=[w,w.parent/'shapes_rank0.json',w.parent/'complete.json']+[model/n for n in ('config.json','status.json','data_manifest.json','source_sha256.json','rank0.jsonl')]
 for f in files:
  assert f.is_file()
  items[str(f.relative_to(root))]=f.stat().st_size
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

