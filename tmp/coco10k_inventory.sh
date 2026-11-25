set -eu
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
df -h /data5/shenglin.zeng
python - <<'PY'
import json,os,zipfile,importlib.util
from pathlib import Path
r=Path.cwd()
for parent in ['gan_stage_parallel','gan_stage_joint4_4gpu','gan_stage_joint34_4gpu','dmd_control']:
    launch=json.loads((r/'outputs'/parent/'launch.json').read_text())
    for rec in launch['runs']:
        if parent=='gan_stage_parallel' and rec['variant']=='joint4':continue
        p=r/'outputs'/rec['run']
        print('RUN',rec['variant'],str(p))
        print('CONFIG', (p/'config.json').read_text())
        for f in p.rglob('*'):
            if f.is_file() and (f.suffix in ['.pt','.safetensors'] or f.name in ['DONE','complete.json','summary.json','metadata.json','manifest.json']):
                print('ARTIFACT',str(f.relative_to(r)),f.stat().st_size)
print('MODELS',[(str(p.relative_to(r)),p.stat().st_size) for p in (r/'models/baselines/dmd2').glob('*')])
z=r/'datasets/dmd2/coco10k.zip'
print('COCO_ZIP',z.exists(),z.stat().st_size if z.exists() else None)
if z.exists():
    with zipfile.ZipFile(z) as a:print('ZIP_MEMBERS',len(a.namelist()),a.namelist()[:15])
for base in ['models/metrics','src','datasets/dmd2']:
    print('DIR',base,[p.name for p in (r/base).iterdir()])
for mod in ['torch','diffusers','clean_fid','cleanfid','torch_fidelity','scipy','safetensors','open_clip']:
    spec=importlib.util.find_spec(mod)
    print('PACKAGE',mod,str(spec.origin) if spec else None)
PY
