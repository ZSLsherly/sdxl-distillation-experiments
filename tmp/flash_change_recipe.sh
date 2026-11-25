set -eu
. /root/sdxl_research/env.sh
cd "$SDXL_ROOT/tmp/flash_gan_g160_20261102/code"
export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python - <<'PY'
from run_flash_after_winner import job,prior,make_config,load,write
from launch import flags
from train import arguments
from pathlib import Path
import os,sys
pid=int((job/'queue.pid').read_text())
os.kill(pid,0)
assert not (prior/'training_complete.json').exists() and not (job/'effective_config.json').exists()
new={'both_middle':False,'student_mix':0.01}
for winner in ('B','C'):
    cfg=make_config(winner,new,4)
    sys.argv=['train','--run','validate',*flags(cfg)]
    a=arguments()
    assert (a.variant,a.pair_fake_stages,a.both_middle,a.flash_terminal_gan,a.fake_student_mix,a.fake_mix_start_g,a.fake_track_pilot)==('joint4',True,False,True,.01,1,True)
    assert (a.max_student_updates,a.fake_updates,a.batch,a.micro_batch)==(160,2,64,4)
    print('PARAMETERS PASSED',winner)
write(job/'recipe.json',new)
assert load(job/'recipe.json')==new
print('EFFECTIVE QUEUE RECIPE',pid,new,'pending')
PY
python smoke_paired_gan_validation.py --flash --paired-mix --output flash_paired_mix_cpu.json > flash_paired_mix_cpu.log
python -c "import json; r=json.load(open('flash_paired_mix_cpu.json')); assert r['passed'] and (r['G_updates'],r['fake_updates'],r['D_updates'])==(4,8,4); print('CPU passed: 4G / 8 fake / 4 D')"
