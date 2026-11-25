set -eu
. /root/sdxl_research/env.sh
python - <<'PY'
from pathlib import Path
r=Path('/root/sdxl_research')
for p in ['code/sdxl_research/configs/uniform_control_a800_g160.json','tmp/bc_coco_winner_20261102/code/evaluate.py','tmp/c_mix1_dm_pair_coco10k/code/queue_dm_pair_coco.py','outputs/bc_winner_late_mix_20261102/coco10k_C_mix1/manifest.json','outputs/bc_winner_late_mix_20261102/coco10k_C_mix1/features/real','outputs/turning_BC_paired_quota_20261102/uniform_control/source_sha256.json']:
 q=r/p;print(p,q.exists())
print('helpers',[(str(p),p.exists()) for p in r.glob('tmp/*/code/queue_dm_pair_coco.py')][:8])
PY
