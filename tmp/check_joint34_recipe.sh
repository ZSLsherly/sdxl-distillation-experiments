set -eu
cd /data5/shenglin.zeng/sdxl_crossdistill/outputs/gan_stage_joint34_4gpu/joint34_20261019T195324_4gpu_mb16
sha256sum source/train.py source/batching.py source/pixel.py
python3 - <<'PY'
import json,collections
from pathlib import Path
rows=[json.loads(x) for x in Path('rank0.jsonl').read_text().splitlines()];g=[r for r in rows if r['phase']=='student_done']; print('actual',collections.Counter((r['stage'],r['gan'],r['dmd']) for r in g),len(g));print('D/fake rows',sum(r['phase']=='fake_D' for r in rows));s=Path('source/train.py').read_text().splitlines()
for i,line in enumerate(s,1):
 if 'knots[i + 1] if' in line or 'stage = 4 if args.benchmark' in line or 'use_gan =' in line or 'adversarial = -disc' in line or 'generated = rollout_chunks' in line or 'pcm / plan.accumulation' in line:print(i,line.strip())
PY
