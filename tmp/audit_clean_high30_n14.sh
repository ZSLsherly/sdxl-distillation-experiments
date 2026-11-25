set -eu
. /root/sdxl_research/env.sh
python - <<'PY'
from pathlib import Path
import json,os,math,hashlib,collections
r=Path(os.environ['SDXL_ROOT']);j=r/'outputs/A_clean_global_high30_g160_20261114';c=r/'tmp'/j.name/'code'
checks=json.loads((j/'baseline_source_hashes.json').read_text())
assert all(hashlib.sha256((c/n).read_bytes()).hexdigest()==h for n,h in checks.items())
print('SOURCE_MATCH',len(checks))
print('STATUS',json.loads((j/'model/status.json').read_text()))
text=(j/'model/rank0.jsonl').read_text();rows=[json.loads(l) for l in text[:text.rfind('\n')].splitlines()]
groups=collections.defaultdict(list)
for x in rows:
 if x['phase'] in ['boundary_turning_sample','fake_turning_sample']:
  prefix='student' if x['phase']=='boundary_turning_sample' else 'fake'
  groups[(prefix,x['stage'])]+=x['timesteps']
for k,ts in sorted(groups.items()):
 print('TIMESTEPS',k,dict(n=len(ts),high_count=sum(t>=900 for t in ts),high_fraction=sum(t>=900 for t in ts)/len(ts),minimum=min(ts),maximum=max(ts)))
for phase in ['student','fake']:
 ts=[t for (p,k),v in groups.items() if p==phase for t in v]
 print('POOLED',phase,dict(n=len(ts),high=sum(t>=900 for t in ts),fraction=sum(t>=900 for t in ts)/len(ts)))
gs=[x for x in rows if x['phase']=='student_done'];fs=[x for x in rows if x['phase']=='fake']
for x in gs[-3:]:print('G',{k:x.get(k) for k in ['g','stage','pcm_loss','middle_dm_loss','terminal_dm_loss','grad_norm','cycle_seconds']})
assert all(math.isfinite(x[k]) for x in gs for k in ['pcm_loss','middle_dm_loss','terminal_dm_loss','grad_norm'])
assert all(math.isfinite(x[k]) for x in fs for k in ['fake_loss','fake_grad_norm'])
print('FINITE',len(gs),len(fs),'CLIPPED',sum(x['grad_norm']>1 for x in gs))
print('FAKE_LAST',[(x['fake_update'],x['fake_stage'],x['fake_loss']) for x in fs[-4:]])
data=dict(status=json.loads((j/'model/status.json').read_text()),source_matches=True,
 sampling={f'{k[0]}_{k[1]}':dict(n=len(v),high_count=sum(t>=900 for t in v)) for k,v in groups.items()},
 student_done=len(gs),fake_done=len(fs),finite=True,student_clipped=sum(x['grad_norm']>1 for x in gs))
(j/'live_audit.json').write_text(json.dumps(data,indent=2))
PY
