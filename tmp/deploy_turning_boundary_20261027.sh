set -euo pipefail
source /data5/shenglin.zeng/env.sh
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
root=/data5/shenglin.zeng/sdxl_crossdistill
dest="$root/outputs/turning_boundary_ablation_20261027"
mkdir -p "$dest/code"
python -u - <<'PY'
import tarfile,json,hashlib
from pathlib import Path
root=Path('/data5/shenglin.zeng/sdxl_crossdistill');dest=root/'outputs/turning_boundary_ablation_20261027'
archive=root/'turning_boundary_ablation_20261027.tar.gz'
with tarfile.open(archive) as f:
 for m in f.getmembers():
  assert not m.name.startswith('/') and '..' not in Path(m.name).parts
 f.extractall(dest/'code',filter='data')
code=dest/'code/sdxl_research'
for name in ['normalized_u','raw_angle']:
 p=code/'configs'/f'turning_{name}_g160.json';d=json.loads(p.read_text())
 d['turning_profile']=str(code/'turning_profiles'/f'{name}_512.json')
 p.write_text(json.dumps(d,indent=2))
(dest/'deployment.json').write_text(json.dumps({'status':'deployed_not_started','archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
 'framework':'boundary DM, PCM + random middle + terminal; fake:G=2; no GAN','code':str(code)},indent=2))
print('DEPLOYED',code,flush=True)
PY
cd "$dest/code/sdxl_research"
python -m unittest test_turning_sampling -v
python smoke_turning_boundary_cpu.py
