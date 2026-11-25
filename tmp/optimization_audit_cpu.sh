source /data5/shenglin.zeng/env.sh
export CUDA_VISIBLE_DEVICES=''
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import json, statistics, inspect
from pathlib import Path
from torch.distributed._composable.fsdp import FSDPModule, fully_shard
print('FSDP_API', inspect.signature(fully_shard), [n for n in dir(FSDPModule) if any(k in n for k in ['reshard','prefetch','unshard','sync'])])
for run in ['gan_stage_joint4_4gpu/joint4_20261019T122237_4gpu_mb16','gan_stage_joint34_4gpu/joint34_20261019T195324_4gpu_mb16','gan_stage_parallel/joint4_20261017T232317_mb16']:
 p=Path('outputs')/run
 rows=[json.loads(x) for x in (p/'rank0.jsonl').read_text().splitlines()]
 gs=[r for r in rows if r['phase']=='student_done' and r['g']>3]
 fs=[r for r in rows if r['phase']=='fake_D' and r['fake_update']>15]
 print(json.dumps({'run':run,'updates':len(gs),'cycle_median':statistics.median(r['cycle_seconds'] for r in gs),'cycle_mean':statistics.mean(r['cycle_seconds'] for r in gs),'student_mean':statistics.mean(r['seconds'] for r in gs),'fake_D_mean':statistics.mean(r['seconds'] for r in fs),'allocated_peak':max(r['peak_allocated_GiB'] for r in gs)}))
PY
