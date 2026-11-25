set -eu
. /root/sdxl_research/env.sh
export CUDA_VISIBLE_DEVICES=''
python - <<'PY'
from pathlib import Path
import hashlib,json,os
r=Path(os.environ['SDXL_ROOT']);w=Path(os.environ['SDXL_WORK'])
for c in (r/'tmp/bc_coco_winner_20261102/code',r/'tmp/flash_gan_g160_20261102/code'):
 print('CODE',c)
 for name in ('train.py','boundary.py','fake_tracking.py','run_flash_after_winner.py','run_bc_coco_then_winner.py','evaluation_core.py','control_after_bc.py','evaluate.py','launch.py'):
  p=c/name
  if p.is_file(): print(name,hashlib.sha256(p.read_bytes()).hexdigest())
 t=(c/'train.py').read_text().splitlines()
 for i,line in enumerate(t,1):
  if "eval_original_indices" in line or 'len(idx)==8' in line:print(i,line.strip())
j=r/'outputs/bc_winner_late_mix_20261102';g=r/'outputs/flash_gan_g160_20261102'
for p in [j/'coco10k/status_0.json',j/'coco10k/scores.json',j/'coco10k_A/status_0.json',j/'late_mix/status.json',g/'recipe.json']:
 if p.is_file():print(p.name,p.read_text()[:370])
print('training dirs', (j/'late_mix').exists(),[str(p) for p in g.glob('gan_micro*')])
for d in (j/'coco10k',j/'coco10k_A'):
 m=json.loads((d/'manifest.json').read_text());print('manifest',d.name,list(m['models']),m['n'],len(m['visual_indices']),hashlib.sha256(json.dumps([m['samples'],m['visual_indices'],m['protocol']],sort_keys=True).encode()).hexdigest())
p=r/'datasets/parallel_10k_seed20261018/manifest.json'
m=json.loads(p.read_text()); print('DATA',len(m['eval_original_indices']),len(m['holdout_original_indices']),m['eval_original_indices'],m['holdout_original_indices'][:10])
PY
ps -o pid,ppid,stat,etime,cmd -p 88668,103394,117460,118220
