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
# Restore exact baseline routing and loss; only terminal noise sampling changes.
p=code/'turning_sampling.py';s=p.read_text()
needle="        times=list(range(lo,hi+1))"
assert s.count(needle)==1
s=s.replace(needle,needle+"\n        if self.profile.get('terminal_low_only') and (lo,hi)==(20,979):\n            return times,self.profile['probabilities']")
p.write_text(s)
cfg=json.loads((root/'outputs/bc_winner_late_mix_20261102/effective_config.json').read_text())
cfg.update(terminal_only_dm=True,matched_wall_seconds=None)
assert cfg['fake_updates']==2 and cfg['pcm_weight']==1 and cfg['fake_student_mix']==.01 and cfg['fake_mix_start_g']==1
assert cfg['batch']==64 and cfg['micro_batch']==16 and cfg['match_fake_prior'] and cfg['pair_fake_stages']
profile=json.loads((root/cfg['turning_profile']).read_text());profile.update(name='terminal_low30',terminal_low_only=True,beta=.3,interval_rule='quota_uniform')
profile.pop('terminal_interval_masses',None)
profile['probabilities']=[.3/81 if 20<=t<=100 else .7/879 for t in profile['timesteps']]
(code/'low30.json').write_text(json.dumps(profile,indent=2));cfg['turning_profile']=str(code/'low30.json')
(job/'effective_config.json').write_text(json.dumps(cfg,indent=2))
(job/'source_parent_hashes.json').write_text(json.dumps(provenance,indent=2))
for p in code.glob('*.py'):compile(p.read_text(),str(p),'exec')
print('PREPARED',code)
