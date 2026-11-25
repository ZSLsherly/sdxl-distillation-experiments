source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import json,hashlib,math
from pathlib import Path
m=json.loads(Path('outputs/active_experiment.json').read_text());p=Path('outputs')/m['runs'][0]['run']
if not (p/'config.json').exists():print('MODEL_LOADING');raise SystemExit
c=json.loads((p/'config.json').read_text())
assert c['variant']=='joint34' and c['gan_stages']==[3,4] and c['discriminator_enabled']
assert c['effective_batch']==64 and c['fake_updates']==5 and c['world_size']==4
assert c['save_every_updates']==40 and c['student_lr']==c['fake_lr']==1e-6 and not c['resume']
print('CONFIG_VERIFIED')
old=Path('outputs/gan_stage_joint34/joint34_20261018T091442_4gpu_mb16')
evs=sorted((p/'eval').glob('*_g000000_initial'));olds=sorted((old/'eval').glob('*_g000000_initial'))
if evs and olds:
 files=list(evs[0].glob('*.png'))
 print('INITIAL_IMAGES',len(files),'IDENTICAL_TO_PREVIOUS',sum(hashlib.sha256(f.read_bytes()).digest()==hashlib.sha256((olds[0]/f.name).read_bytes()).digest() for f in files))
students=[]
for rank in range(4):
 f=p/f'rank{rank}.jsonl';rs=[json.loads(s) for s in f.read_text().splitlines()] if f.exists() else []
 fs=[r for r in rs if r['phase']=='fake_D'];gs=[r for r in rs if r['phase']=='student_done']
 assert all(math.isfinite(r['fake_loss']) and math.isfinite(r['D_hinge']) for r in fs)
 assert all(r['dmd'] and r['gan']==(r['stage'] in (3,4)) and math.isfinite(r['grad_norm']) for r in gs)
 print('RANK',rank,'FAKE_D',len(fs),'G',len(gs))
 if gs:students.append(gs[0])
if len(students)==4:
 ids=[i for r in students for i in r['original_indices_rank']];assert len(ids)==len(set(ids))==64
 print('FIRST_UPDATE',json.dumps(students[0]));print('STARTUP_VERIFIED')
print('STATUS',(p/'status.json').read_text() if (p/'status.json').exists() else 'pending')
PY
