cd /data5/shenglin.zeng/a50k_20261124 || exit 1
/data5/shenglin.zeng/envs/crossdistill/bin/python - <<'PY'
import json,statistics
from pathlib import Path
from PIL import Image,ImageDraw
from datetime import datetime
out=Path('reports/pair_g616');out.mkdir(exist_ok=True,parents=True)
panels=[];stats=[]
for run in ('A50K_singlepass','A50K_no_middle_singlepass'):
 d=Path('outputs')/run
 rows=[json.loads(l) for l in (d/'rank0.jsonl').read_text().splitlines() if l.endswith('}')]
 gs=[r for r in rows if r['phase']=='student_done' and 596<=r['g']<=616]
 assert len(gs)==21
 def tm(r):return datetime.strptime(r['timestamp'],'%Y%m%dT%H%M%S%z')
 stats.append((run,{k:round(statistics.mean(r[k] for r in gs[1:]),6) for k in ('pcm_loss','middle_dm_loss','terminal_dm_loss','grad_norm','cycle_seconds')},(tm(gs[-1])-tm(gs[0])).total_seconds()/20))
 ev,=list((d/'eval').glob('*_g000616_periodic'))
 cases={}
 for p in ev.glob('rank*.json'):
  for r in json.loads(p.read_text())['samples']:
   if r['split']=='holdout':cases[int(r['original_index'])]=(r['prompt'],r['image'],ev/r['image'])
 panels.append(cases)
print(stats);a,b=panels;assert a.keys()==b.keys() and len(a)==4
canvas=Image.new('RGB',(4*260,2*295),'white');draw=ImageDraw.Draw(canvas)
for col,i in enumerate(sorted(a)):
 assert a[i][:2]==b[i][:2] and 'seed102' in a[i][1]
 for row,c in enumerate((a,b)):
  img=Image.open(c[i][2]).convert('RGB').resize((256,256),Image.Resampling.LANCZOS)
  x,y=col*260,row*295;canvas.paste(img,(x,y+25));draw.text((x+2,y+4),f'{"A" if row==0 else "No middle DM"} | ID {i} | G616',fill='black')
canvas.save(out/'G616_holdout.jpg',quality=94)
PY
sha256sum reports/pair_g616/G616_holdout.jpg
