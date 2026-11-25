import json,os,time,subprocess,sys,hashlib,math
from pathlib import Path
from launch import gpu_check
root=Path(os.environ['SDXL_ROOT'])
inputs=Path(os.environ['SDXL_WORK'])/'bc_texture_inputs'
train=root/'outputs/h20_C_mix1_terminal_GAN_G160'
out=train.parent/'h20_C_mix1_terminal_GAN_G160_coco10k'
def read(p):return json.loads(p.read_text())
while not (train/'status.json').exists() or read(train/'status.json').get('status') not in ('complete','stopped'):
    time.sleep(60)
s=read(train/'status.json')
assert (s['status'],s['student_updates'],s['fake_updates'],s['D_updates'])==('complete',160,320,160)
for n in (40,80,120,160):
    dirs=list((train/'eval').glob(f'*_g{n:06d}_periodic/refinement'))
    assert len(dirs)==1 and len(list(dirs[0].glob('*.png')))==24,n
check,=list((train/'weights').glob('*/complete.json'))
assert read(check)['student_updates']==160 and read(check)['world_size']==1
ck=check.parent
import torch
from diffusers import UNet2DConditionModel
w=torch.load(ck/'student_rank0.pt',weights_only=True,map_location='cpu',mmap=True)
with torch.device('meta'):model=UNet2DConditionModel.from_config(read(root/'models/sdxl-base-1.0/unet/config.json'))
model.load_state_dict(w,strict=True,assign=True)
assert all(torch.isfinite(t).all() for t in w.values())
del model,w
m=read(Path('coco_reference_manifest.json'));assert len(m['samples'])==m['n']==10000 and len(m['visual_indices'])==128
assert m['csv_sha256']=='fc1efb7dd0f7d561a999125340fc4b10632d28f2547c17b174170514889c8599'
with (ck/'student_rank0.pt').open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
arm='C_mix1_terminal_GAN'
m['models']={arm:dict(checkpoint=str(ck),world_size=1,student_updates=160,config=read(train/'config.json'),shards=[dict(name='student_rank0.pt',sha256=sha)])}
out.mkdir(exist_ok=True)
mp=out/'manifest.json'
if mp.exists():assert read(mp)==m
else:mp.write_text(json.dumps(m))
(out/'features').mkdir(exist_ok=True)
link=out/'features/real'
if not link.exists():link.symlink_to(inputs/'reference_features/real',target_is_directory=True)
assert (inputs/'public_inputs_ready.json').exists()
while True:
    try:gpu_check([0],75);break
    except RuntimeError:time.sleep(30)
opts=['--root',str(inputs),'--out',str(out)]
subprocess.run([sys.executable,'evaluate.py','worker',*opts,'--methods',arm,'--batch','4'],check=True)
subprocess.run([sys.executable,'evaluate.py','metrics',*opts],check=True)
score=read(out/'complete.json')['scores'][arm]
assert score['n']==10000 and all(math.isfinite(score[k]) for k in ('FID','Patch_FID'))
print('COCO10K_COMPLETE',score,flush=True)
