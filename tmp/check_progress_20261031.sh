set -eu
root=/data3/shenglin.zeng/sdxl_turning_ablation_20261028
"$root/env/bin/python" - <<'PY'
import json,math,statistics
from pathlib import Path
p=Path('/data3/shenglin.zeng/sdxl_turning_ablation_20261028/outputs/turning_batch64_g160_4gpu_teacher_shard_20261029/normalized_u_g160')
for rank in range(4):
 rows=[json.loads(s) for s in (p/f'rank{rank}.jsonl').read_text().splitlines()]
 students=[x for x in rows if x.get('phase')=='student_done']
 fakes=[x for x in rows if x.get('phase')=='fake']
 keys=['grad_norm','pcm_loss','middle_dm_loss','terminal_dm_loss','cycle_seconds','peak_allocated_GiB']
 print(json.dumps({'rank':rank,'last':{k:students[-1].get(k) for k in ['g','timestamp']+keys},'recent10_mean':{k:statistics.mean(x[k] for x in students[-10:]) for k in keys},'nonfinite':sum(not math.isfinite(x[k]) for x in students for k in keys),'fake_nonfinite':sum(not math.isfinite(x[k]) for x in fakes for k in ['fake_loss','fake_grad_norm'])}))
print('snapshots',[(str(f.relative_to(p)),f.stat().st_size) for f in p.glob('**/*.safetensors')])
print('eval_latest',[x.name for x in sorted((p/'eval').iterdir())[-3:]])
PY
tail -n 4 "$root/outputs/turning_batch64_g160_4gpu_teacher_shard_20261029/normalized_u_g160.log"
df -h "$root"
