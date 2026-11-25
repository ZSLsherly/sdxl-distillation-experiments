source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import torch,hashlib,json
from pathlib import Path
d=torch.load('datasets/prepared.pt',weights_only=True,map_location='cpu')
h=[hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest() for x in d['latents']]
matches=[(i,j) for i in range(96) for j in range(112,120) if h[i]==h[j]]
captions=[str(x).strip().lower() for x in d['pcm_prompts']]
same_caption=[(i,j) for i in range(96) for j in range(112,120) if captions[i]==captions[j]]
p=Path('outputs/gan_stage_pilot_v2');p.mkdir(exist_ok=True)
r={'exact_latent_train_holdout_duplicates':matches,'exact_caption_train_holdout_duplicates':same_caption,'train_indices':list(range(96)),'holdout_indices':list(range(112,120)),'scope':'exact cached tensor and canonical caption duplicates only; not perceptual near-duplicate detection'}
(p/'data_overlap_audit.json').write_text(json.dumps(r,indent=2))
print(json.dumps({k:v for k,v in r.items() if not k.endswith('indices')}))
assert not matches and not same_caption,'Select fresh holdout before training'
PY
