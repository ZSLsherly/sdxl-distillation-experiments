"""Clean-global baseline, identical trainer; only student/fake re-noise prior changes."""
from pathlib import Path
import hashlib,json,os,shutil
r=Path(os.environ['SDXL_ROOT']);n='A_clean_global_high30_g160_20261114'
j=r/'outputs'/n;c=r/'tmp'/n/'code';base=r/'outputs/A_clean_global_g160_20261113'
assert not j.exists(),'Inspect existing job before preparing'
assert json.loads((base/'complete.json').read_text())['G']==160
j.mkdir(parents=True);c.mkdir(parents=True,exist_ok=True)
hashes=json.loads((base/'model/source_sha256.json').read_text())
for p in (base/'model/source').glob('*.py'):
 assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[p.name]
 shutil.copy2(p,c/p.name)
for name in ['launch.py','turning_batch64_workflow.py']:
 shutil.copy2(r/'tmp/A_clean_global_g160_20261113/code'/name,c/name)
cfg=json.loads((base/'effective_config.json').read_text())
assert cfg['pcm_weight']==1 and cfg['gan_weight']==0 and cfg['turning_profile'] is None
assert cfg['pair_fake_stages'] and cfg['fake_updates']==2 and cfg['batch']==64 and cfg['micro_batch']==16
assert cfg['max_student_updates']==160 and cfg['boundary_noise_max']==979 and not cfg.get('fake_student_mix',0)
saved=json.loads((base/'model/config.json').read_text())
assert saved['student_dmd_t_range']=={str(k):[20,979] for k in (2,3,4)}
assert saved['experiment_modules']['middle_target']=='clean_x0'
for k,v in cfg.items():
 if k=='schedule':assert saved[k]==v+[-1]
 elif k=='data_dir':assert Path(saved[k]).resolve()==(r/v).resolve()
 else:assert saved[k]==v,(k,saved[k],v)
profile=dict(name='clean_global_high30',metric='fixed high-noise interval quota',beta=.3,
 support=[20,979],source_sha256=None,smoothing='none',offset_rule='none',exponent=1.,
 importance_correction=False,timesteps=list(range(20,980)),prior_intervals=[[900,979]],
 interval_rule='quota_uniform',probabilities=[.3/80 if t>=900 else .7/880 for t in range(20,980)])
(c/'high30.json').write_text(json.dumps(profile,indent=2))
new=dict(cfg,turning_profile=str(c/'high30.json'),match_fake_prior=True)
assert {k for k in cfg if cfg[k]!=new[k]}=={'turning_profile','match_fake_prior'}
assert all(hashlib.sha256((c/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
for p in c.glob('*.py'):compile(p.read_text(),str(p),'exec')
(j/'effective_config.json').write_text(json.dumps(new,indent=2))
(j/'baseline_config.json').write_text(json.dumps(cfg,indent=2))
(j/'baseline_source_hashes.json').write_text(json.dumps(hashes,indent=2))
(j/'design.json').write_text(json.dumps(dict(baseline=str(base),training_source_changed=False,
 change='student/fake stages 2,3,4: 30% U[900,979] + 70% U[20,899]',
 PCM=1,GAN=0,mix=0,student='random middle clean DM plus terminal DM',
 fake='paired middle clean prediction plus terminal; two updates',
 G=160,F=320,initialization='original SDXL; fresh student/fake',evaluation='not queued'),indent=2))
print('PREPARED',n,'IDENTICAL_TRAINING_SOURCE',len(hashes))
