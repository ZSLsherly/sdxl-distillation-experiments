set -eu
/data5/shenglin.zeng/envs/crossdistill/bin/python - <<'PY'
import json,math,struct
from pathlib import Path
p=Path('/data5/shenglin.zeng/sdxl_crossdistill/models/sdxl-base-1.0/unet/diffusion_pytorch_model.fp16.safetensors')
with p.open('rb') as f:
 n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n))
P=sum(math.prod(v['shape']) for k,v in h.items() if k!='__metadata__')
print(json.dumps({'UNet_tensor_elements':P,'one_full_resume_state_GiB':24*P/1024**3,'one_BF16_snapshot_GiB':2*P/1024**3,'all_planned_checkpoints_GiB':116*P/1024**3,'with_10_percent_overhead_GiB':127.6*P/1024**3}))
PY
