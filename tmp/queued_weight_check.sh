set -eu
. /root/sdxl_research/env.sh
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python - <<'PY'
import ast,hashlib,json,os,torch
from pathlib import Path
from diffusers import UNet2DConditionModel
root=Path(os.environ['SDXL_ROOT']); c=root/'tmp/bc_coco_winner_20261102/code'
path=c/'train.py';s=path.read_text()
assert hashlib.sha256(path.read_bytes()).hexdigest()=='0f270a794e8cb38a5cb5fd1c142e24c1323c712b0dafe956f21e503ac0d20b98'
needle='        if rank == 0:\n            write_json(dest / "complete.json",'
block="""            if reason=='end' and args.fake_track_pilot and args.variant=='dmd_only' and world==1:
                state=torch.load(dest/'student_rank0.pt',map_location='cpu',mmap=True,weights_only=True)
                with torch.device('meta'):
                    cpu_unet=UNet2DConditionModel.from_config(json.loads((base/'unet/config.json').read_text()))
                cpu_unet.load_state_dict(state,strict=True,assign=True)
                if not all(torch.isfinite(v).all().item() for v in state.values()):
                    raise FloatingPointError('Nonfinite saved student weights')
                write_json(dest/'cpu_validation.json',{'strict_load':True,'finite':True,'tensors':len(state)})
"""
assert s.count(needle)==1 and not (root/'outputs/bc_winner_late_mix_20261102/late_mix').exists()
updated=s.replace(needle,needle.replace('            write_json',block+'            write_json'));compile(updated,str(path),'exec')
dest=next((root/'outputs/turning_BC_paired_quota_20261102/B/weights').glob('*/student_rank0.pt')).parent
import textwrap
record=[]
env=dict(torch=torch,UNet2DConditionModel=UNet2DConditionModel,json=json,base=root/'models/sdxl-base-1.0',dest=dest,write_json=lambda path,data:record.append(data))
stmt=ast.parse(textwrap.dedent(block)).body[0]
exec(compile(ast.Module(body=stmt.body,type_ignores=[]),'<checkpoint verification>','exec'),env)
assert record==[{'strict_load':True,'finite':True,'tensors':1680}]
stage=path.with_suffix('.audit-new');stage.write_text(updated);stage.replace(path)
print('Existing B G160 checkpoint tested with deployed acceptance:',record)
PY
