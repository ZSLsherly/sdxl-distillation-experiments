"""Read-only audit of the running experiment; CPU only, no training changes."""
import collections, hashlib, json, math, statistics, sys
from pathlib import Path
import torch
from batching import BatchPlan
from boundary import phase_ranges, critic_stage
from turning_sampling import TurningSampler

torch.set_num_threads(1)
root = Path('/root/sdxl_research')
code = root/'code/sdxl_research'
out = root/'outputs/turning_BC_paired_quota_20261102'
arm = sys.argv[1] if len(sys.argv)>1 else 'B'
assert arm in ('B','C','uniform_control')
run = out/arm
cfg = json.loads((run/'config.json').read_text())
hashes = json.loads((run/'source_sha256.json').read_text())
for name, expected in hashes.items():
    assert hashlib.sha256((run/'source'/name).read_bytes()).hexdigest() == expected
    live = Path(cfg['turning_profile']) if name == 'turning_distribution.json' else code/name
    assert hashlib.sha256(live.read_bytes()).hexdigest() == expected, name
a,b = [json.loads((code/f'configs/turning_{arm}_quota_a800_g160.json').read_text()) for arm in 'BC']
diff = {k:[a.get(k),b.get(k)] for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
assert set(diff)=={'turning_profile'}, diff
reference=a if arm=='B' else b
if arm=='uniform_control':reference=dict(b,turning_profile=None,match_fake_prior=False)
for key,value in reference.items():
    if key in ('schedule','data_dir'): continue
    assert cfg[key] == (str(root/value) if key=='turning_profile' and value else value), (key,cfg[key],value)
assert cfg['schedule']==[999,749,499,249,-1]
assert cfg['world_size']==1 and cfg['effective_batch']==64 and cfg['accumulation']==4
assert not cfg['discriminator_enabled']
data = torch.load(root/'datasets/parallel_10k_seed20261018/prepared.pt',mmap=True,weights_only=True,map_location='cpu')
manifest=json.loads((run/'data_manifest.json').read_text())
train_ids=set(data['original_indices'][:data['train_count']])
assert data['train_count']==10000
assert not train_ids.intersection(manifest['eval_original_indices'])
rows=[]
for line in (run/'rank0.jsonl').read_text().splitlines():
    try: rows.append(json.loads(line))
    except json.JSONDecodeError: break # writer may be between write and flush
gs=[r for r in rows if r['phase']=='student_done']
fs=[r for r in rows if r['phase']=='fake']
plan=BatchPlan(64,1,16)
ranges=phase_ranges(cfg['schedule'],10,979)
sampler=TurningSampler(Path(cfg['turning_profile']),979) if cfg['turning_profile'] else None
replayed=collections.Counter()
for r in rows:
    phase=r['phase']
    if phase=='fake_turning_sample':
        k=critic_stage(r['fake_update'],True,cfg['seed']);assert k==r['stage']
        expected=sampler.sample(plan.rank_positions(r['fake_update'],0),cfg['seed']+500000+1000*k,ranges[k])
    elif phase=='boundary_turning_sample':
        k=r['stage']
        expected=sampler.sample(plan.positions(r['g'],r['micro'],0),cfg['seed']+1000*k,ranges[k])
    else: continue
    assert r['timesteps']==expected, (phase,r)
    replayed[phase]+=len(expected)
for i,g in enumerate(gs):
    assert g['g']==i+1
    assert [r['fake_stage'] for r in fs[2*i:2*i+2]]==[g['stage'],4]
    assert all(math.isfinite(g[k]) for k in ('grad_norm','pcm_loss','middle_dm_loss','terminal_dm_loss'))
    assert len(g['original_indices_rank'])==64
    assert set(g['original_indices_rank']).issubset(train_ids)
assert all(math.isfinite(f[k]) for f in fs for k in ('fake_loss','fake_grad_norm'))
evals=[]
for p in sorted((run/'eval').glob('*/rank0.json')):
    e=json.loads(p.read_text());assert len(e['samples'])==8
    assert all((p.parent/s['image']).is_file() and math.isfinite(s['latent_std']) for s in e['samples'])
    evals.append({'g':e['student_updates'],'latent_std':[s['latent_std'] for s in e['samples']]})
result={'arm':arm,'source_files_verified':len(hashes),'BC_config_diff':diff,'training_count':10000,
        'holdout_overlap':0,'timesteps_exactly_replayed':dict(replayed),'G':len(gs),'fake':len(fs),
        'latest_student':gs[-1] if gs else None,'evaluation_checks':evals,
        'median_cycle_seconds':statistics.median(g['cycle_seconds'] for g in gs) if gs else None,
        'conclusion':'No material error found in checked runtime, data split, sampling or pairing; checkpoint loading is a separate check.'}
(out/f'overnight_audit_{arm}.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ('latest_student','evaluation_checks')},indent=2))
