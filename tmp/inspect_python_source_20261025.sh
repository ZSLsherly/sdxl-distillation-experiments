source /data5/shenglin.zeng/env.sh
CUDA_VISIBLE_DEVICES='' python - <<'PY'
import sys,sysconfig
from pathlib import Path
print('PREFIX',sys.prefix,'BASE_PREFIX',sys.base_prefix,'EXECUTABLE',Path(sys.executable).resolve())
print('SITE',sysconfig.get_path('purelib'))
p=Path(sys.prefix)/'pyvenv.cfg'
if p.exists():print(p.read_text())
PY
