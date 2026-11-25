import ast
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
src=root/'sdxl_research'
for p in src.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
base=json.loads((src/'configs/uniform_joint4.json').read_text())
variants={
 'baseline':{},'rgb_only':{'discriminator':'rgb'},'logistic_gan':{'gan_loss':'logistic'},
 'symmetric_flip':{'gan_augment':'both'},'full_decode_crop':{'gan_view':'full'},
 'critic_student_cycle':{'fake_policy':'student_cycle'},'critic_match_student':{'fake_policy':'match_student'},
 'last_step_gan_only':{'student_stages':[4],'dmd_stages':[],'gan_stages':[4]},
}
dest=src/'configs/modules';dest.mkdir(exist_ok=True)
for name,changes in variants.items():
 (dest/f'{name}.json').write_text(json.dumps({**base,**changes},indent=2),encoding='utf-8')
print('PREPARED',len(variants),'standalone experiment configs; nothing launched')
