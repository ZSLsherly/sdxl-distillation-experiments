set -eu
r=/data1/shenglin.zeng/dmd2_matched_10k_20261119
base=/data1/shenglin.zeng/sdxl_eval_uniform_20261025
"$base/env/bin/python" - <<'PY'
from pathlib import Path
import os,signal,sysconfig
r=Path('/data1/shenglin.zeng/dmd2_matched_10k_20261119')
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:args=p.read_bytes().split(b'\0')
 except OSError:continue
 if args and args[0].decode(errors='ignore')==str(r/'env/bin/python') and args[1:4]==[b'-m',b'pip',b'install']:
  os.kill(int(p.parent.name),signal.SIGTERM);print('STOP_OWN_DEP_INSTALL',p.parent.name)
target=r/'env/lib/python3.12/site-packages/reuse_sdxl_base.pth'
target.write_text(sysconfig.get_path('purelib')+'\n')
print('REUSE_PACKAGE_PATH',sysconfig.get_path('purelib'))
PY
"$r/env/bin/python" -c 'import torch,diffusers,transformers,accelerate,numpy;print(torch.__version__,diffusers.__version__,numpy.__version__)'
"$r/env/bin/python" -m pip install 'peft==0.17.1' 'matplotlib==3.10.7' 'imageio==2.38.2'
