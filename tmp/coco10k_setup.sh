set -eu
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python -m pip install --no-deps clean-fid==0.1.35
python - <<'PY'
from pathlib import Path
import zipfile,json,inspect,torch
from cleanfid import fid,features,resize
r=Path.cwd()
z=zipfile.ZipFile(r/'datasets/dmd2/coco10k.zip')
print('META_FILES',[n for n in z.namelist() if not n.endswith('.jpg')])
print('CSV',z.read('coco10k/subset.csv')[:1300].decode())
for parent in ['gan_stage_parallel','gan_stage_joint4_4gpu','gan_stage_joint34_4gpu','dmd_control']:
    launch=json.loads((r/'outputs'/parent/'launch.json').read_text())
    for rec in launch['runs']:
        if parent=='gan_stage_parallel' and rec['variant']=='joint4':continue
        p=r/'outputs'/rec['run']
        ds=[d for d in (p/'weights').glob('*g000160*') if (d/'complete.json').exists()]
        print('SELECT',rec['variant'],[(str(d),json.loads((d/'complete.json').read_text())) for d in ds])
        print('SHAPES_SAMPLE',list(json.loads((ds[-1]/'shapes_rank0.json').read_text()).items())[:2])
print('FEATURE_BUILDER',inspect.getsource(features.build_feature_extractor))
print('RESIZER',inspect.getsource(resize.build_resizer))
print('MEMORY',Path('/proc/meminfo').read_text().splitlines()[:3])
PY
