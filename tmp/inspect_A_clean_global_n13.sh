set -eu
. /root/sdxl_research/env.sh
printf 'DISK\n'
df -h /root/autodl-tmp /root/sdxl_research
printf 'PROCESSES\n'
ps -eo pid,ppid,args | grep -E 'run_.*py|torchrun|train.py' | grep -v grep
python - <<'PY'
from pathlib import Path
import os,json
r=Path(os.environ['SDXL_ROOT'])
print('ROOT',r)
for rel in ['outputs/A_mid_clean_x0','tmp/A_mid_clean_x0/code','outputs/turning_BC_paired_quota_20261102/uniform_control/source','outputs/A_pcm10_g160/model/source']:
 p=r/rel; print('PATH',p,'EXISTS',p.exists())
 if p.exists():print([x.name for x in p.iterdir()][:30])
p=r/'outputs/turning_BC_paired_quota_20261102/uniform_control'
print('A_CONFIG', (p/'config.json').read_text())
for name in ['train.py','boundary.py','batching.py']:
 f=p/'source'/name
 if f.exists():
  s=f.read_text().splitlines()
  for i,line in enumerate(s):
   if any(x in line for x in ['phase_ranges(', 's = knots[boundary_dm.critic_stage','time_range=ranges','scheduler,generated','def random_inputs','def local_rollout','def rollout_boundary_and_final']):
    print(name,i+1,'\n'.join(s[max(0,i-2):i+4]))
PY
