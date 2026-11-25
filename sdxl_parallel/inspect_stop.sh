set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import json,os
from pathlib import Path
active=Path('outputs/active_experiment.json')
m=json.loads((active if active.exists() else Path('outputs/gan_stage_parallel/launch.json')).read_text())
print('LAUNCH',json.dumps(m))
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
  if 'sdxl_parallel/train.py' in args and '--run' in args and args[args.index('--run')+1] in {r['run'] for r in m['runs']}:
   print('PROCESS',p.name,'UID',p.stat().st_uid,'PGID',os.getpgid(int(p.name)),' '.join(args))
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
for r in m['runs']:
 p=Path('outputs')/r['run']
 if (p/'status.json').exists():print('STATUS',r['variant'],(p/'status.json').read_text())
PY
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
