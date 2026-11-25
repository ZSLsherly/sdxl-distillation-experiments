set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
if test -f logs/v4_validation.exit; then cat logs/v4_validation.exit; fi
python - <<'PY'
from pathlib import Path
p=Path('outputs/v4_validation/latest_runs.txt')
if p.exists():
 for name in p.read_text().splitlines():
  run=Path('outputs')/name;log=Path('logs')/('v4_'+run.name+'.log')
  print(name)
  if log.exists():print('\n'.join(log.read_text(errors='replace').splitlines()[-10:]))
  if (run/'status.json').exists():print((run/'status.json').read_text())
 for f in Path('logs').glob('v4_comparison_*.log'):
  print(f.name,'\n'.join(f.read_text(errors='replace').splitlines()[-4:]))
PY
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
