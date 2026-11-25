set -eu
root=/data5/shenglin.zeng/sdxl_crossdistill
date -Is
python3 - <<'PY'
import json
from pathlib import Path
root=Path('/data5/shenglin.zeng/sdxl_crossdistill')
runs=['dmd_control/dmd_only_20261018T141942_4gpu_mb16','gan_stage_joint4_4gpu/joint4_20261019T122237_4gpu_mb16','gan_stage_joint34_4gpu/joint34_20261019T195324_4gpu_mb16','joint4_uniform_10k_2gpu_mb16_20261023/train','boundary_dmd_10k_a100_20261025_v1']
for name in runs:
 p=root/'outputs'/name;print('\nRUN',name)
 if not (p/'config.json').exists():print('CONFIG_MISSING');continue
 d=json.loads((p/'config.json').read_text())
 print('CONFIG',json.dumps({k:d.get(k) for k in ['variant','objective','schedule','pcm_boundary','dmd_t_max','pcm_t_range','student_dmd_t_range','turning_sampling','data_dir','train_count','batch','world_size','student_lr','max_student_updates','fake_updates','gan_stages','dmd_stages']}))
 for f in ['status.json','workflow_error.json']:
  if (p/f).exists():print(f,(p/f).read_text()[:1800])
 completed=[]
 for f in (p/'weights').glob('*/complete.json'):
  m=json.loads(f.read_text());completed.append((m.get('student_updates',m.get('gcount',-1)),str(f)))
 if completed:print('LATEST_COMMITTED',sorted(completed)[-1])
 for directory in [p/'source',p/'code',p.parent/'code/sdxl_research']:
  for f in [directory/'batching.py',directory/'train.py']:
   if f.exists():
    lines=f.read_text().splitlines();print('SOURCE',f)
    for i,line in enumerate(lines):
     if any(k in line for k in ['turning','randint(lo','lo, hi =','phase == "fake" else (20','def schedule_settings','dmd_t_max=']):print(i+1,line.strip())
PY
