source /data5/shenglin.zeng/env.sh
python - <<'PY'
import json,statistics
from pathlib import Path
root=Path('/data5/shenglin.zeng/sdxl_crossdistill/outputs/hparam_pilot')
for name in ['lr5e7_pcm10','lr25e8_pcm10']:
 p=root/name;s=json.loads((p/'summary.json').read_text());rows=[]
 for rank in range(4):rows+=json.loads((p/f'validation_after_rank{rank}.json').read_text())
 print(name,'train_minutes',s['train_seconds']/60,'boundary_mse',statistics.mean(r['boundary_relative_mse'] for r in rows),'latent_std',statistics.mean(r['latent_std'] for r in rows),'finite',all(r['finite'] for r in rows))
PY
