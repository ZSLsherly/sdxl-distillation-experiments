from pathlib import Path
import os,json,shutil,hashlib
root=Path(os.environ['SDXL_ROOT']); name='C_mix1_pcm_dm4_fake4_low30_g160'
code=root/'tmp'/name/'code'; job=root/'outputs'/name
code.mkdir(parents=True,exist_ok=True);job.mkdir(exist_ok=True)
base=root/'outputs/c_mix1_pcm_dm4_matched_time/model'
assert not (job/'model').exists()
for p in (root/'tmp/c_mix1_pcm_dm4_matched_time/code').glob('*.py'): shutil.copy2(p,code/p.name)
for p in (base/'source').glob('*.py'): shutil.copy2(p,code/p.name)
provenance={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (base/'source').glob('*.py')}
p=code/'boundary.py';s=p.read_text();start=s.index('def critic_stage(');end=s.index('\n\ndef ',start)
s=s[:start]+'''def critic_stage(update, paired=False, seed=20261017, updates_per_student=2):
    # This isolated experiment trains the fake only on the four-step clean endpoint.
    if not paired or updates_per_student != 2:
        raise ValueError('Terminal-only fake requires paired mode and two updates')
    return 4
'''+s[end:];p.write_text(s)
p=code/'train.py';s=p.read_text();s=s.replace("fake='random middle 2/3 then terminal 4'", "fake='two independent terminal 4 updates'")
needle="    if args.terminal_only_dm:\n        config['experiment_modules'].update"
assert needle in s
s=s.replace(needle,"    if args.terminal_only_dm:\n        config['fake_stage_policy']='terminal4_twice'\n        config['experiment_modules'].update",1);p.write_text(s)
cfg=json.loads((root/'outputs/bc_winner_late_mix_20261102/effective_config.json').read_text())
cfg.update(terminal_only_dm=True,matched_wall_seconds=None)
assert cfg['fake_updates']==2 and cfg['pcm_weight']==1 and cfg['fake_student_mix']==.01 and cfg['fake_mix_start_g']==1
assert cfg['batch']==64 and cfg['micro_batch']==16 and cfg['match_fake_prior'] and cfg['pair_fake_stages']
profile=json.loads((root/cfg['turning_profile']).read_text());profile.update(name='terminal_low30',prior_intervals=[[20,100]],beta=.3,interval_rule='quota_uniform')
profile.pop('terminal_interval_masses',None)
profile['probabilities']=[.3/81 if 20<=t<=100 else .7/879 for t in profile['timesteps']]
(code/'low30.json').write_text(json.dumps(profile,indent=2));cfg['turning_profile']=str(code/'low30.json')
(job/'effective_config.json').write_text(json.dumps(cfg,indent=2))
(job/'source_parent_hashes.json').write_text(json.dumps(provenance,indent=2))
for p in code.glob('*.py'):compile(p.read_text(),str(p),'exec')
print('PREPARED',code)
