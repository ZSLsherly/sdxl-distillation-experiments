set -eu
. /root/sdxl_research/env.sh
python /root/autodl-tmp/sdxl_research/tmp/finish_migration.py
/root/sdxl_research/launch.sh --config /root/sdxl_research/code/sdxl_research/configs/boundary_global_noise_validation.json --gpus 0,1 --run path_check_only > /root/autodl-tmp/sdxl_runs/logs/launch_dry_run.txt
cat /root/autodl-tmp/sdxl_runs/logs/launch_dry_run.txt
python - <<'PY'
import sys,os,json
from pathlib import Path
r=Path('/root/sdxl_research')
assert Path(sys.prefix)==r/'env' and Path(sys.base_prefix)==r/'runtime'
assert (r/'outputs').resolve()==Path('/root/autodl-tmp/sdxl_runs/outputs')
for key in ['env','runtime','code','models','datasets']:
 assert (Path('/root/autodl-tmp/sdxl_research')/key).resolve()==r/key
print('FINAL_PATH_CHECK_OK')
PY
df -h / /root/autodl-tmp
