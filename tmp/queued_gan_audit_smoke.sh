set -eu
. /root/sdxl_research/env.sh
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
c="$SDXL_ROOT/tmp/flash_gan_g160_20261102/code"
cd "$c"; export PYTHONPATH="$c"
python smoke_paired_gan_validation.py --flash --paired-mix --output /tmp/flash_queue_audit_smoke.json >/dev/null
python - <<'PY'
from pathlib import Path
import os,json
d=json.loads(Path('/tmp/flash_queue_audit_smoke.json').read_text()); print('paired GAN tiny CPU',d['passed'],d['G_updates'],d['fake_updates'],d['D_updates'])
from run_flash_after_winner import make_config,job,load,code
prior=Path(os.environ['SDXL_ROOT'])/'tmp/bc_coco_winner_20261102/code'
for a in ('B','C'):
    cfg=load(prior/'configs'/f'turning_{a}_quota_a800_g160.json');cfg.update(fake_track_pilot=True,fake_student_mix=.01,fake_mix_start_g=1)
    gan=make_config(a,load(job/'recipe.json'),4)
    print('effective option differences',a,sorted(k for k in set(gan)|set(cfg) if cfg.get(k)!=gan.get(k)))
PY
