set -euo pipefail
root=/data1/shenglin.zeng/teacher_turning_20261026
cd "$root"
export LD_LIBRARY_PATH="$root/runtime/lib:${LD_LIBRARY_PATH:-}"
env/bin/python - <<'PY'
from PIL import Image,ImageDraw
from pathlib import Path
import json,zipfile
p=Path('stage_smoke');print((p/'worker0_complete.json').read_text())
for r in json.loads((p/'metrics_rank0.json').read_text()):
 print(r['variant'],'DINO',r['dino_distance_to_teacher'],'LPIPS',r['lpips_full_to_teacher'],'PATCH',r['lpips_patch_to_teacher'])
canvas=Image.new('RGB',(1536,548),'white');draw=ImageDraw.Draw(canvas)
for j,name in enumerate(['student','replace1','replace2','replace3','replace4','teacher']):
 im=Image.open(p/f'096_{name}.png');canvas.paste(im.resize((256,256)),(j*256,36));canvas.paste(im.crop((128,128,384,384)),(j*256,292));draw.text((j*256+8,10),name,fill='black')
canvas.save(p/'comparison.png')
with zipfile.ZipFile('stage_smoke_results.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in p.iterdir():
  if f.suffix in ['.json','.log'] or f.name=='comparison.png':z.write(f,f.name)
PY
