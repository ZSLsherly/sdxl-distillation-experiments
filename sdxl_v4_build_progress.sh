set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
from pathlib import Path
import json,statistics
from PIL import Image,ImageDraw
root=Path('outputs/gan_stage_v4');launch=json.loads((root/'launch.json').read_text());groups={}
for v in ['joint4','gan4']:
 p=Path('outputs')/launch[v]['run'];dirs=sorted((p/'eval').glob('*'));initial=next(d for d in dirs if '_g000000_' in d.name);after=next(d for d in dirs if '_g000002_' in d.name)
 rows={}
 for tag,d in [('before',initial),('after',after)]:
  rows[tag]=[]
  for f in d.glob('rank*.json'):
   for r in json.loads(f.read_text())['samples']:r['path']=str(d/r['image']);rows[tag].append(r)
 groups[v]=rows
keys=[]
for keyword in ['night, tranquil','Wood Effect']:
 keys.append(next(r['original_index'] for r in groups['joint4']['before'] if keyword in r['prompt']))
canvas=Image.new('RGB',(960,1160),'white');draw=ImageDraw.Draw(canvas)
for i,key in enumerate(keys):
 for c,(v,tag,label) in enumerate([('joint4','before','Before (G0)'),('joint4','after','GAN+DMD group (G2)'),('gan4','after','GAN-only group (G2)')]):
  r=next(r for r in groups[v][tag] if r['original_index']==key);im=Image.open(r['path']).convert('RGB')
  draw.text((c*320+5,i*580+5),label,fill='black');canvas.paste(im.resize((312,312)),(c*320+4,i*580+25));canvas.paste(im.crop((384,384,640,640)),(c*320+32,i*580+340))
canvas.save(root/'progress_g2.png')
summary={}
for v,data in groups.items():
 before={r['original_index']:r for r in data['before']};after=data['after']
 summary[v]={'student_updates':2,'student_GAN_updates':0,'images':len(after),'texture_energy_mean_relative_change':statistics.mean(r['laplacian_variance_diagnostic_only']/before[r['original_index']]['laplacian_variance_diagnostic_only']-1 for r in after)}
 for split in ['train','holdout']:
  rows=[r for r in after if r['split']==split]
  summary[v][split]={kind:{branch:statistics.mean(r['D'][kind][branch] for r in rows) for branch in ['total','dino','rgb']} for kind in ['real','fake']}
(root/'progress_g2_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
PY
