"""Verify completed inference weights on CPU without interrupting the other arm."""
import json,sys,hashlib,tarfile
from pathlib import Path
import torch
from diffusers import UNet2DConditionModel
from run_quota_bc import verify
torch.set_num_threads(1)
root=Path('/root/sdxl_research')
out=root/'outputs/turning_BC_paired_quota_20261102'
arm=sys.argv[1];assert arm in ('B','C','uniform_control')
run=out/arm
dest=Path(verify(run))
file=dest/'student_rank0.pt'
state=torch.load(file,map_location='cpu',weights_only=True,mmap=True)
config=json.loads((root/'models/sdxl-base-1.0/unet/config.json').read_text())
with torch.device('meta'):
    model=UNet2DConditionModel.from_config(config)
model.load_state_dict(state,strict=True,assign=True)
assert all(torch.isfinite(t).all().item() for t in state.values())
result={'arm':arm,'G':160,'fake':320,'file':str(file),'bytes':file.stat().st_size,
        'tensor_count':len(state),'parameter_count':sum(p.numel() for p in model.parameters()),
        'strict_model_load':'passed','all_tensors_finite':True,'validation':'CPU load only; no checkpoint GPU inference rerun yet'}
(out/f'checkpoint_validation_{arm}.json').write_text(json.dumps(result,indent=2))
with tarfile.open(out/f'{arm}_completed_evidence.tar.gz','w:gz') as tar:
    for name in ['config.json','status.json','source_sha256.json','data_manifest.json','rank0.jsonl']:
        tar.add(run/name,arcname=f'{arm}/{name}')
    for g in [0,40,80,118,120,160]:
        for folder in (run/'eval').glob(f'*_g{g:06d}_*'):
            tar.add(folder,arcname=f'{arm}/eval/{folder.name}')
    tar.add(out/f'checkpoint_validation_{arm}.json',arcname=f'checkpoint_validation_{arm}.json')
print(json.dumps(result,indent=2))
