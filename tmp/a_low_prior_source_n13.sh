set -eu
. /root/sdxl_research/env.sh
r="$SDXL_ROOT/outputs/turning_BC_paired_quota_20261102/uniform_control"
python - <<'PY'
from pathlib import Path
import json
p=Path('/root/sdxl_research/outputs/turning_BC_paired_quota_20261102/uniform_control')
c=json.loads((p/'config.json').read_text());print({k:c.get(k) for k in ['objective','variant','schedule','pcm_weight','student_lr','fake_lr','fake_updates','batch','micro_batch','seed','fake_track_pilot','fake_student_mix','turning_profile','pair_fake_stages','fake_weight_max','data_dir','eval_every_samples']})
print('SOURCE',list(x.name for x in (p/'source').glob('*.py')))
print((p/'source/turning_sampling.py').read_text()[-1000:])
print('BASE_RAW_CONFIG_EXISTS',Path('/root/sdxl_research/code/sdxl_research/configs/uniform_control_a800_g160.json').exists())
PY
ps -eo pid,comm,args | awk '$2 ~ /python|torchrun/ && /run_|train.py|evaluate.py/ {print substr($0,1,200)}'
