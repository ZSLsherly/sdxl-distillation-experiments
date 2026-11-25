set -eu
. /root/sdxl_research/env.sh
export CUDA_VISIBLE_DEVICES=''
python - <<'PY'
import json,hashlib,math,collections
from pathlib import Path
root=Path('/root/sdxl_research');run=root/'outputs/turning_BC_interval_quota_20261101/B'
files=['train.py','boundary.py','turning_sampling.py','batching.py','objectives.py','training_state.py','run_quota_bc.py']
print('SOURCE_HASHES',json.dumps({n:hashlib.sha256((root/'code/sdxl_research'/n).read_bytes()).hexdigest() for n in files}))
config=json.loads((run/'config.json').read_text());print('CONFIG',json.dumps({k:config.get(k) for k in ['objective','variant','batch','micro_batch','rollout_batch','fake_updates','fake_offload','offload_student_optimizer','turning_sampling','student_dmd_t_range','checkpoint_mode','save_every_updates','teacher_cfg','seed']}))
source=run/'source'
print('LIVE_VS_SNAPSHOT',json.dumps({n:((source/n).exists() and (source/n).read_bytes()==(root/'code/sdxl_research'/n).read_bytes()) for n in files if n!='run_quota_bc.py'}))
rows=[json.loads(s) for s in (run/'rank0.jsonl').read_text().splitlines()]
by=collections.defaultdict(list)
for r in rows:
 if r['phase']=='boundary_turning_sample':by[r['stage']].extend(r['timesteps'])
print('OBSERVED_TIMES',json.dumps({k:{'n':len(v),'high_n':sum(900<=t<=979 for t in v),'high_fraction':sum(900<=t<=979 for t in v)/len(v),'min':min(v),'max':max(v)} for k,v in by.items()}))
gs=[r for r in rows if r['phase']=='student_done'];fs=[r for r in rows if r['phase']=='fake']
print('UPDATES',len(gs),len(fs),'LATEST',json.dumps(gs[-1] if gs else {}))
print('FINITE',all(math.isfinite(r[k]) for r in gs for k in ['grad_norm','pcm_loss','middle_dm_loss','terminal_dm_loss']),all(math.isfinite(r[k]) for r in fs for k in ['fake_grad_norm','fake_loss']))
print('WEIGHTS_SO_FAR',list(str(p) for p in (run/'weights').glob('*')))
PY
