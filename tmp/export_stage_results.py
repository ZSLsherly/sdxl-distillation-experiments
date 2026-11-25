"""Package quantitative records and lossless comparison sheets; raw images stay remote."""
import json,zipfile,hashlib,importlib.metadata
from pathlib import Path
from PIL import Image,ImageDraw
root=Path('/data1/shenglin.zeng/teacher_turning_20261026/stage_intervention')
assert json.loads((root/'complete.json').read_text())['cases']==32
names=['student','replace1','replace2','replace3','replace4','teacher']
gallery=root/'gallery';gallery.mkdir(exist_ok=True)
sources={name:hashlib.sha256((root.parent/name).read_bytes()).hexdigest() for name in ['stage_intervention_core.py','sdxl_stage_intervention.py','guard_stage_intervention.py']}
(root/'provenance.json').write_text(json.dumps({'source_sha256':sources,'assets':json.loads((root.parent/'stage_assets/manifest.json').read_text()),
 'versions':{name:importlib.metadata.version(name) for name in ['torch','diffusers','transformers','torchvision']}},indent=2))
for case in sorted(root.glob('case_*.json')):
 d=json.loads(case.read_text());i=d['prompt_index'];canvas=Image.new('RGB',(6*256,2*256+36),'white');draw=ImageDraw.Draw(canvas)
 for j,name in enumerate(names):
  im=Image.open(root/f'{i:03d}_{name}.png').convert('RGB');assert im.size==(1024,1024)
  canvas.paste(im.resize((256,256),Image.Resampling.LANCZOS),(j*256,36))
  canvas.paste(im.crop((128,128,384,384)),(j*256,292))
  draw.text((j*256+8,10),name,fill='black')
 canvas.save(gallery/f'{i:03d}.png')
with zipfile.ZipFile(root.parent/'stage_intervention_results.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in root.glob('*'):
  if f.suffix in ['.json','.log']:z.write(f,f.name)
 for f in gallery.glob('*.png'):z.write(f,'gallery/'+f.name)
 for name in sources:z.write(root.parent/name,'code/'+name)
 # Full-resolution images for four fixed prompts, without selecting by outcome.
 for i in range(96,100):
  for name in names:z.write(root/f'{i:03d}_{name}.png','full/'+f'{i:03d}_{name}.png')
print('PACKAGED', (root.parent/'stage_intervention_results.zip').stat().st_size,flush=True)
