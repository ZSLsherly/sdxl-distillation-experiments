from pathlib import Path
import subprocess
b=Path('/data1/shenglin.zeng/dymd_prior_8_20261101')
for name in ['install.log','download_parallel.log','run.log']:
 p=b/name
 if p.exists():
  print(name)
  print('\n'.join(p.read_text(errors='replace').replace('\r','\n').splitlines()[-7:]))
for p in b.glob('*READY'): print(p.name)
for p in (b/'official/pf_wan/checkpoints/PF_Wan').glob('*.chunks.json'):
 import json
 print(p.name,len(json.loads(p.read_text()))*8,'MiB completed')
