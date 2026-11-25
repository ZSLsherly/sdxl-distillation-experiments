"""Clone the completed clean-x0 control; expand only its score timestep support."""
from pathlib import Path
import hashlib
import json
import os
import shutil

root = Path(os.environ['SDXL_ROOT'])
name = 'A_clean_global_g160_20261113'
job = root/'outputs'/name
code = root/'tmp'/name/'code'
base = root/'outputs/A_mid_clean_x0'
assert not job.exists(), 'Inspect existing job before preparing again'
code.mkdir(parents=True, exist_ok=True)
job.mkdir(parents=True)
hashes = json.loads((base/'model/source_sha256.json').read_text())
source = base/'model/source'
for p in source.glob('*.py'):
    assert hashlib.sha256(p.read_bytes()).hexdigest() == hashes[p.name], p.name
    shutil.copy2(p, code/p.name)
cfg = json.loads((base/'effective_config.json').read_text())
assert cfg['pcm_weight']==1 and cfg['gan_weight']==0 and cfg['turning_profile'] is None
assert cfg['batch']==64 and cfg['micro_batch']==16 and cfg['fake_updates']==2
assert cfg['max_student_updates']==160 and cfg['checkpoint_mode']=='final-weights'
assert cfg['boundary_noise_max']==979 and cfg['pair_fake_stages']
assert not cfg.get('fake_student_mix', 0)
assert 'middle_target' in (code/'train.py').read_text()
path = code/'boundary.py'
s = path.read_text()
old = '''    ranges = {k: (knots[k] + margin if k < 4 else 20, knots[k-1] - margin)
              for k in (2, 3, 4)}'''
new = '''    # All supervised outputs are CLEAN x0, so all stages share full support.
    # Student and fake both consume this mapping; generation knots stay fixed.
    ranges = {k: (20, 979) for k in (2, 3, 4)}'''
assert s.count(old)==1
path.write_text(s.replace(old,new))
helpers = root/'tmp/A_pcm10_terminal_low30_g160_20261113/code'
for n in ['launch.py','turning_batch64_workflow.py']:
    shutil.copy2(helpers/n,code/n)
changed=[p.name for p in source.glob('*.py') if (code/p.name).read_bytes()!=p.read_bytes()]
assert changed==['boundary.py'], changed
for p in code.glob('*.py'):compile(p.read_text(),str(p),'exec')
(job/'effective_config.json').write_text(json.dumps(cfg,indent=2))
(job/'baseline_source_hashes.json').write_text(json.dumps(hashes,indent=2))
(job/'design.json').write_text(json.dumps(dict(baseline=str(base), changed_files=changed,
    old_ranges={2:[509,979],3:[259,979],4:[20,979]},
    new_ranges={2:[20,979],3:[20,979],4:[20,979]},
    student_and_fake='clean x0; paired middle+terminal', PCM=1, GAN=0, mix=0,
    initialization='original SDXL, independent fresh run', evaluation='not queued'),indent=2))
print('PREPARED',job, 'ONLY_TRAINING_SOURCE_CHANGE',changed)
