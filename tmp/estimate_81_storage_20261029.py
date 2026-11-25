import json,math,shutil,struct
from pathlib import Path
root=Path('/data3/shenglin.zeng/sdxl_turning_ablation_20261028')
weight=root/'models/sdxl-base-1.0/unet/diffusion_pytorch_model.fp16.safetensors'
with weight.open('rb') as f:
    length=struct.unpack('<Q',f.read(8))[0]
    header=json.loads(f.read(length))
n=sum(math.prod(value['shape']) for key,value in header.items() if key!='__metadata__')
# Two FP32 models plus two Adam moments for each model: 24 bytes/parameter.
# Two arms, each retaining G2 + G160 full states and five BF16 snapshots.
full=24*n
snap=2*n
total=4*full+10*snap
free=shutil.disk_usage(root).free
result={'UNet_tensor_elements':n,'full_resume_state_GiB':full/1024**3,
        'student_snapshot_GiB':snap/1024**3,'two_arms_all_planned_checkpoints_GiB':total/1024**3,
        'reserve_with_10_percent_overhead_GiB':total*1.1/1024**3,'available_GiB':free/1024**3,
        'note':'Model and optimizer tensor size estimate; actual files measured after validation.'}
(root/'storage_estimate.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
if free < total*1.1:raise RuntimeError('Insufficient space for planned checkpoints')
