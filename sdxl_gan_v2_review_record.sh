source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
mkdir -p outputs/reviews
python - <<'PY'
import json,hashlib
from pathlib import Path
p=Path('outputs/gan_stage_pilot')
counts={}
for rank in range(4):
    f=p/'dmd'/f'rank{rank}.jsonl'
    rows=[json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []
    counts[str(rank)]={'student_updates':sum(r['phase']=='student' for r in rows),'fake_updates':sum(r['phase']!='student' for r in rows),'last_iteration':rows[-1]['iteration'] if rows else None}
stop=json.loads((p/'review_stop.json').read_text());stop['interrupted_A_progress']=counts
(p/'review_stop.json').write_text(json.dumps(stop,indent=2))
print('V1_INTERRUPTED_PROGRESS',json.dumps(counts))
q=Path('outputs/gan_stage_pilot_v2/preflight/summary.json')
if q.exists():
    s=json.loads(q.read_text());print('V2_PREFLIGHT',json.dumps({k:s[k] for k in ['status','student_updates','fake_updates','pixel_discriminator_updates','D_stage_counts','GAN_G_stage_counts','peak_allocated_GiB']}))
PY
if test -f logs/gan_v2_preflight.exit; then echo "PREFLIGHT_EXIT $(cat logs/gan_v2_preflight.exit)"; fi
tail -n 5 logs/gan_v2_preflight.log
