source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
from pathlib import Path
import json,statistics,datetime,inspect,importlib.metadata
active=json.loads(Path('outputs/active_experiment.json').read_text())
paths=[Path('outputs/gan_stage_joint4_4gpu/joint4_20261019T122237_4gpu_mb16'),Path('outputs')/active['runs'][0]['run']]
def dt(s):return datetime.datetime.strptime(s,'%Y%m%dT%H%M%S%z')
for p in paths:
 rows=[json.loads(s) for s in (p/'rank0.jsonl').read_text().splitlines()];gs=[r for r in rows if r['phase']=='student_done'];fs=[r for r in rows if r['phase']=='fake_D'];fm={r['fake_update']:r for r in fs}
 print('RUN',str(p),'G',len(gs))
 for label,rr in [('early',gs[:12]),('recent',gs[-12:])]:
  print(label,'n',len(rr),'cycle_mean',statistics.mean(r['cycle_seconds'] for r in rr),'student_mean',statistics.mean(r['seconds'] for r in rr),'fake5_mean',statistics.mean(sum(fm[i]['seconds'] for i in range(5*(r['g']-1)+1,5*r['g']+1)) for r in rr))
 gaps={'eval':[],'save_eval':[],'other':[]}
 for a,b in zip(gs,gs[1:]):
  key='save_eval' if a['g']%40==0 else 'eval' if a['g']%2==0 else 'other'
  gaps[key].append((dt(b['timestamp'])-dt(a['timestamp'])).total_seconds()-b['cycle_seconds'])
 print('POST_UPDATE_GAPS',{k:{'n':len(v),'mean':statistics.mean(v),'min':min(v),'max':max(v)} for k,v in gaps.items() if v})
 st=json.loads((p/'status.json').read_text());print('FINAL_TAIL_SECONDS',(dt(st['timestamp'])-dt(gs[-1]['timestamp'])).total_seconds() if st['status']=='complete' else None)
for p in Path('outputs').glob('gan_stage*/**/benchmark.json'):
 try:b=json.loads(p.read_text())
 except Exception:continue
 if b.get('profile_rank0_seconds'):print('ARCHIVED_PROFILE',str(p),json.dumps(b))
for n in ['torch','diffusers','transformers','xformers','deepspeed']:
 try:print('VERSION',n,importlib.metadata.version(n))
 except importlib.metadata.PackageNotFoundError:print('VERSION',n,'not installed')
from diffusers.models.attention_processor import Attention,AttnProcessor2_0
print('DEFAULT_ATTENTION',type(Attention(query_dim=32,heads=4,dim_head=8).processor).__name__)
print('SDPA_IN_PROCESSOR','scaled_dot_product_attention' in inspect.getsource(AttnProcessor2_0))
PY
