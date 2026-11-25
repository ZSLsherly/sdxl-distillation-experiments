set -eu
. /root/sdxl_research/env.sh
cd "$SDXL_ROOT/outputs"
date -Iseconds
python - <<'PY'
from pathlib import Path
import json,datetime
def get(p):return json.loads(p.read_text())
def train(p):
    if not (p/'rank0.jsonl').exists():return
    rows=[]
    for l in (p/'rank0.jsonl').open():
        try:r=json.loads(l)
        except ValueError:continue
        if r['phase']=='student_done':rows.append(r)
    if rows:
        def dt(r):return datetime.datetime.strptime(r['timestamp'],'%Y%m%dT%H%M%S%z')
        print(str(p),len(rows),round((dt(rows[-1])-dt(rows[0])).total_seconds()+rows[0]['cycle_seconds']),rows[-1]['timestamp'])
j=Path('A_pcm10_g160')
print((j/'queue.log').read_text()[-650:])
for p in [Path('c_mix1_pcm_dm4_matched_time/PCM_DM4_G160'),j/'A_pcm10',Path('bc_winner_late_mix_20261102/coco10k_A')]:
    if (p/'status_0.json').is_file():
        r=get(p/'status_0.json');print(str(p),{k:r.get(k) for k in ['done','target','seconds','images_per_second','updated_unix']},'complete',(p/'complete.json').exists())
train(Path('turning_BC_paired_quota_20261102/uniform_control'))
train(j/'model')
PY
