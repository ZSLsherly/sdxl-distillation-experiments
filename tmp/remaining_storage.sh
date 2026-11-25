set -e
source /data5/shenglin.zeng/env.sh
du -x -B1 --max-depth=1 /data5/shenglin.zeng
du -x -B1 --max-depth=1 /data5/shenglin.zeng/sdxl_crossdistill
du -x -B1 --max-depth=2 /data5/shenglin.zeng/sdxl_crossdistill/outputs | sort -n
du -x -B1 --max-depth=2 /data5/shenglin.zeng/outputs | sort -n
du -x -B1 --max-depth=1 /data5/shenglin.zeng/sdxl_crossdistill/models
du -x -B1 --max-depth=1 /data5/shenglin.zeng/sdxl_crossdistill/datasets
python - <<'PY'
import json
from pathlib import Path
root=Path('/data5/shenglin.zeng/sdxl_crossdistill')
q=root/'outputs/gan_stage_joint34/joint34_20261018T091442_4gpu_mb16/weights'
last=sorted(p for p in q.iterdir() if p.is_dir())[-1]
print('EXAMPLE_CHECKPOINT',str(last))
for p in sorted(last.iterdir()):print('FILE',p.name,p.stat().st_size)
shapes=json.loads((last/'shapes_rank0.json').read_text())
total=0
for meta in shapes.values():
 n=1
 for d in meta['shape']:n*=d
 total+=n
print('STATE_ELEMENT_COUNT',total,'BF16_BYTES',total*2)
print('LARGE_OUTPUT_FILES')
for parent in [root/'outputs',Path('/data5/shenglin.zeng/outputs')]:
 for p in parent.rglob('*'):
  if p.is_file() and not p.is_symlink() and p.stat().st_size>1024**3 and 'gan_stage_parallel/' not in str(p) and 'gan_stage_joint34/' not in str(p):
   print(p.stat().st_size,str(p))
PY
