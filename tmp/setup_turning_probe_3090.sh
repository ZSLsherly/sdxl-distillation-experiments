set -euo pipefail
root=/data1/shenglin.zeng/teacher_turning_20261026
cd "$root"
mkdir -p outputs/turning_boundary_ablation_20261027/code
python3 -u - <<'PY'
import tarfile,json
from pathlib import Path
root=Path.cwd();out=root/'outputs/turning_boundary_ablation_20261027'
with tarfile.open(root/'turning_boundary_ablation_20261027.tar.gz') as tar:
 for m in tar.getmembers():assert not m.name.startswith('/') and '..' not in Path(m.name).parts
 tar.extractall(str(out/'code'))
code=out/'code/sdxl_research'
for arm in ['normalized_u','raw_angle']:
 p=code/'configs'/('turning_'+arm+'_g160.json');d=json.loads(p.read_text())
 d['turning_profile']=str(code/'turning_profiles'/(arm+'_512.json'));p.write_text(json.dumps(d,indent=2))
print('CODE_READY',code)
PY
