ps -p 386899 -o pid,etime,args
tail -8 /data5/shenglin.zeng/tools/eval_transfer_20261025/clip_all_stage_20261113.log
python3 - <<'PY'
import json
from pathlib import Path
p=Path('/data5/shenglin.zeng/checkpoints/h20_gan_20261104/sdxl_runs/outputs/h20_C_mix1_terminal_GAN_G160_coco10k/manifest.json')
m=json.loads(p.read_text());print('H20_MODELS',[(k,v.get('world_size')) for k,v in m['models'].items()])
PY
