set -eu
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
from pathlib import Path
from PIL import Image, ImageDraw
import json
p=Path('outputs/coco10k_g160_20261020')
methods=['sdxl','dmd2','dmd_only','gan4','joint4','joint34']
im=Image.new('RGB',(6*384,2*416),'white')
d=ImageDraw.Draw(im)
for row,idx in enumerate([100,101]):
    for col,m in enumerate(methods):
        d.text((col*384+8,row*416+8),f'{m} / COCO #{idx}',fill='black')
        src=Image.open(p/'probe_images'/m/f'{idx:05d}.png')
        im.paste(src.resize((384,384)),(col*384,row*416+32))
im.save(p/'preflight_preview.jpg',quality=92)
for name in ['verify.exit','run.exit','worker_pids.txt','run.log']:
    f=p/name
    if f.exists():print(name, f.read_text()[-1000:])
for f in sorted(p.glob('worker_?.log')):
    print(f.name,f.read_text()[-700:])
PY
