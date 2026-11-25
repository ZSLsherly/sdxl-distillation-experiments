set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import json,hashlib,math,statistics
from pathlib import Path
root=Path.cwd();launch=json.loads((root/'outputs/active_experiment.json').read_text());run=root/'outputs'/launch['runs'][0]['run']
config=json.loads((run/'config.json').read_text());logs=[[json.loads(s) for s in (run/f'rank{i}.jsonl').read_text().splitlines()] for i in range(4)]
students=[[r for r in rows if r['phase']=='student_done'] for rows in logs]
common=min(map(len,students));checks=[]
for j in range(common):
 rs=[rows[j] for rows in students];g=rs[0]['g'];indices=[i for r in rs for i in r['original_indices_rank']]
 assert all(r['g']==g and r['effective_batch']==64 and r['accumulation']==1 for r in rs)
 assert len(indices)==len(set(indices))==64
 assert all(math.isfinite(r[k]) for r in rs for k in ['grad_norm','dmd_loss','gan_loss','pcm_loss'])
 checks.append({k:rs[0][k] for k in ['g','stage','dmd','gan','grad_norm','cycle_seconds','seconds','dmd_loss','gan_loss','pcm_loss']})
hashes=json.loads((run/'source_sha256.json').read_text())
source_match={f:hashlib.sha256((root/'sdxl_parallel'/f).read_bytes()).hexdigest()==sha for f,sha in hashes.items()}
weights=sorted((run/'weights').glob('*/complete.json'))
total_bytes=sum(f.stat().st_size for f in (run/'weights').rglob('*.pt'))
r1=[{k:r[k] for k in ['fake_update','R1_raw','D_grad_norm','seconds']} for r in logs[0] if r['phase']=='fake_D' and r['R1_active']]
record=dict(run=str(run),config=config,source_match=source_match,common_student_updates=common,student_updates=checks,r1_updates=r1,checkpoint_count=len(weights),checkpoint_total_bytes=total_bytes,last_status=json.loads((run/'status.json').read_text()))
dest=root/'outputs/gan_stage_joint34/code_review_runtime.json';dest.write_text(json.dumps(record,indent=2))
print(json.dumps({k:v for k,v in record.items() if k!='config'},indent=2))
PY
free -h
