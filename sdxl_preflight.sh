set -eu
source /data5/shenglin.zeng/env.sh
root="$CROSSDISTILL_ROOT/sdxl_crossdistill"
mkdir -p "$root"/{src,logs,outputs,datasets,models}
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
df -h "$root"
python - <<'PY'
import torch,diffusers,inspect,requests,json
from pathlib import Path
from torch.distributed._composable.fsdp import fully_shard,MixedPrecisionPolicy
print('RUNTIME',torch.__version__,diffusers.__version__)
print('FSDP2',inspect.signature(fully_shard),inspect.signature(MixedPrecisionPolicy))
repo='stabilityai/stable-diffusion-xl-base-1.0'
r=requests.get('https://hf-mirror.com/api/models/'+repo,timeout=60);r.raise_for_status();info=r.json()
print('REVISION',info['sha']);print('FILES',[x['rfilename'] for x in info['siblings']])
Path('/data5/shenglin.zeng/sdxl_crossdistill/model_info.json').write_text(json.dumps(info,indent=2))
for f in ['scheduler/scheduler_config.json','unet/config.json']:
 r=requests.get('https://hf-mirror.com/'+repo+'/resolve/'+info['sha']+'/'+f,timeout=60);r.raise_for_status();print(f,r.text)
from diffusers import StableDiffusionXLPipeline
print('ENCODE',inspect.signature(StableDiffusionXLPipeline.encode_prompt))
print('ADDED TIME IDS',inspect.signature(StableDiffusionXLPipeline._get_add_time_ids))
PY
