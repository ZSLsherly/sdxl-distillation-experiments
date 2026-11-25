set -eu
. /root/sdxl_research/env.sh
export CUDA_VISIBLE_DEVICES=''
python - <<'PY'
import importlib.util,pathlib,shutil
for k in ['cleanfid','scipy','diffusers','torchvision','transformers','PIL','torch']:print(k,bool(importlib.util.find_spec(k)))
p=pathlib.Path('/root/sdxl_research');print('base',list(x.name for x in (p/'models/sdxl-base-1.0').iterdir()));print('work disk',shutil.disk_usage('/root/autodl-tmp'));print((p/'outputs/turning_BC_paired_quota_20261102/uniform_control/status.json').read_text());print('root software',list((p/'models').iterdir()))
PY
