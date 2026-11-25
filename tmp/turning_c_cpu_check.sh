set -euo pipefail
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
test ! -e sdxl_turning_c_20261020
mkdir sdxl_turning_c_20261020
tar -xzf turning_c_20261020.tar.gz -C sdxl_turning_c_20261020
python -m compileall -q sdxl_turning_c_20261020
python -m unittest discover -s sdxl_turning_c_20261020 -p 'test_*.py'
python - <<'PY'
import json,hashlib,difflib
from pathlib import Path
p=Path('sdxl_turning_c_20261020')
base=json.loads((p/'base_source_sha256.json').read_text())
for name,expected in base.items():
    actual=hashlib.sha256((Path('sdxl_parallel')/name).read_bytes()).hexdigest()
    assert actual==expected,(name,'baseline source changed')
for name in ['batching.py','pixel.py','objectives.py']:
    assert (p/name).read_bytes()==(Path('sdxl_parallel')/name).read_bytes()
for name in ['train.py','training_state.py']:
    print('DIFF',name)
    print(''.join(difflib.unified_diff((Path('sdxl_parallel')/name).read_text().splitlines(True),(p/name).read_text().splitlines(True))))
print('TURNING_C_CPU_CHECK_PASS')
(p/'cpu_check_pass.json').write_text(json.dumps({
    'archive_sha256':hashlib.sha256(Path('turning_c_20261020.tar.gz').read_bytes()).hexdigest(),
    'checks':'original batching tests plus turning distribution, packing invariance and unchanged random inputs'
}))
PY
