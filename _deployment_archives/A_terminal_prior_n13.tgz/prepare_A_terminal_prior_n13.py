"""Freeze the actual A trainer; change terminal re-noise sampling only."""
from pathlib import Path
import hashlib
import json
import os
import shutil

root = Path(os.environ['SDXL_ROOT'])
name = 'A_pcm10_terminal_low30_g160_20261113'
base = root/'outputs/A_pcm10_g160/model'
job = root/'outputs'/name
code = root/'tmp'/name/'code'
assert not (job/'model').exists(), 'Inspect any existing training before modifying code'
code.mkdir(parents=True, exist_ok=True)
job.mkdir(parents=True, exist_ok=True)
hashes = json.loads((base/'source_sha256.json').read_text())
for path in (base/'source').glob('*.py'):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[path.name], path.name
    shutil.copy2(path, code/path.name)
helpers = root/'tmp/A_pcm10_g160/code'
for name in ['launch.py','turning_batch64_workflow.py']:
    origin = helpers/name
    if not origin.exists():
        origin = root/'code/sdxl_research'/name
    if not origin.exists():
        origin = root/'tmp/bc_coco_winner_20261102/code'/name
    shutil.copy2(origin, code/name)

path = code/'turning_sampling.py'
source = path.read_text()
needle = '        if phase==\'student\' or (phase==\'fake\' and include_fake):'
assert source.count(needle) == 1
source = source.replace(needle, '''        if self.profile.get('terminal_only') and tuple(time_range or ()) != (20, 979):
            return randoms  # Preserve A's middle timestep RNG exactly.
''' + needle)
path.write_text(source)

cfg = json.loads((root/'outputs/A_pcm10_g160/effective_config.json').read_text())
saved = json.loads((base/'config.json').read_text())
for key, value in cfg.items():
    if key == 'schedule':
        assert saved[key] == value + [-1]
    elif key == 'data_dir':
        assert Path(saved[key]).resolve() == (root/value).resolve()
    else:
        assert saved.get(key) == value, (key, saved.get(key), value)
times = list(range(20, 980))
profile = dict(name='A_pcm10_terminal_low30', metric='fixed terminal timestep interval quota',
               beta=.3, support=[20,979], source_sha256=None, smoothing='none',
               offset_rule='none', exponent=1., importance_correction=False,
               terminal_only=True, timesteps=times, prior_intervals=[[20,100]],
               interval_rule='quota_uniform',
               probabilities=[.3/81 if t <= 100 else .7/879 for t in times])
(code/'terminal_low30.json').write_text(json.dumps(profile, indent=2))
new = dict(cfg, turning_profile=str(code/'terminal_low30.json'), match_fake_prior=True)
assert {k for k in cfg if cfg[k] != new[k]} == {'turning_profile','match_fake_prior'}
(job/'effective_config.json').write_text(json.dumps(new, indent=2))
(job/'baseline_config.json').write_text(json.dumps(cfg, indent=2))
(job/'baseline_source_hashes.json').write_text(json.dumps(hashes, indent=2))
for path in code.glob('*.py'):
    compile(path.read_text(), str(path), 'exec')
print('PREPARED', job)
