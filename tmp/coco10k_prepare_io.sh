set -eu
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
from pathlib import Path
import os,inspect
from cleanfid.inception_torchscript import InceptionV3W
p=Path('/proc/3837070')
if p.exists():
    print('IO', (p/'io').read_text())
    for fd in (p/'fd').iterdir():
        try:
            dest=os.readlink(fd)
            if 'shenglin.zeng' in dest:print('OPEN_FILE',dest,(p/'fdinfo'/fd.name).read_text().splitlines()[:2])
        except FileNotFoundError:pass
print('INCEPTION_SOURCE',inspect.getsource(InceptionV3W.__init__))
PY
