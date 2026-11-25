set -eu
. /root/sdxl_research/env.sh
export CUDA_VISIBLE_DEVICES=''
export PYTHONPATH="$SDXL_ROOT/code/sdxl_research"
python - <<'PY'
import json,hashlib,math
from pathlib import Path
from turning_sampling import TurningSampler
from boundary import phase_ranges,critic_stage
from batching import BatchPlan
root=Path('/root/sdxl_research');run=root/'outputs/turning_BC_matched_quota_20261101/B'
c=json.loads((run/'config.json').read_text());assert c['match_fake_prior']
s=TurningSampler(Path(c['turning_profile']),979);ranges=phase_ranges([999,749,499,249,-1],10,979);plan=BatchPlan(64,1,16)
rows=[json.loads(x) for x in (run/'rank0.jsonl').read_text().splitlines()];counts={'fake':0,'student':0};examples=[]
for r in rows:
 if r['phase']=='fake_turning_sample':
  k=r['stage'];assert k==critic_stage(r['fake_update'])
  expected=s.sample(plan.rank_positions(r['fake_update'],0),c['seed']+500000+1000*k,time_range=ranges[k])
  assert expected==r['timesteps'];counts['fake']+=len(expected)
  examples.append({'fake_update':r['fake_update'],'stage':k,'n':len(expected),'high_n':sum(900<=x<=979 for x in expected)})
 elif r['phase']=='boundary_turning_sample':
  k=r['stage'];expected=s.sample(plan.positions(r['g'],r['micro'],0),c['seed']+1000*k,time_range=ranges[k])
  assert expected==r['timesteps'];counts['student']+=len(expected)
gs=[r for r in rows if r['phase']=='student_done'];fs=[r for r in rows if r['phase']=='fake']
assert all(math.isfinite(r[k]) for r in gs for k in ['grad_norm','pcm_loss','middle_dm_loss','terminal_dm_loss'])
assert all(math.isfinite(r[k]) for r in fs for k in ['fake_grad_norm','fake_loss'])
for name in ['train.py','turning_sampling.py','boundary.py']:
 assert (run/'source'/name).read_bytes()==(root/'code/sdxl_research'/name).read_bytes()
r={'same_stage_same_distribution':True,'independent_seed_offset':500000,'stage_cycle_preserved':True,'replayed_samples':counts,'fake_samples':examples,'completed_G':len(gs),'completed_fake':len(fs),'latest_G':{k:gs[-1][k] for k in ['g','cycle_seconds','grad_norm','peak_allocated_GiB']} if gs else None,'scope':c['turning_sampling']['scope']}
(run.parent/'sampling_alignment_audit.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
PY
