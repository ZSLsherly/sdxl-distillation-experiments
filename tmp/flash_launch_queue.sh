set -eu
. /root/sdxl_research/env.sh
c="$SDXL_ROOT/tmp/flash_gan_g160_20261102/code"
j="$SDXL_ROOT/outputs/flash_gan_g160_20261102"
mkdir -p "$j"
cd "$c"
export PYTHONPATH="$c" CUDA_VISIBLE_DEVICES=''
python - <<'PY'
import json,os,sys
from pathlib import Path
from run_flash_after_winner import make_config,job,code,work
from launch import flags
from train import arguments
recipe={'both_middle':False,'student_mix':0.01}
dest=job/'recipe.json'
if dest.exists():assert json.loads(dest.read_text())==recipe
else:dest.write_text(json.dumps(recipe,indent=2))
for winner in ('B','C'):
    cfg=make_config(winner,recipe,4)
    sys.argv=['train','--run','validate',*flags(cfg)];a=arguments()
    assert a.max_student_updates==160 and a.pair_fake_stages and a.flash_terminal_gan and a.fake_student_mix==recipe['student_mix']
    assert a.both_middle==recipe['both_middle'] and a.batch==64 and a.micro_batch==4
    assert Path(a.sam_path,'model.safetensors').is_file() and a.gan_weight==.005
print('CPU_SMOKE',json.loads((code/'flash_cpu_smoke.json').read_text())['passed'])
PY
if test -e "$j/queue.pid" && kill -0 "$(cat "$j/queue.pid")" 2>/dev/null; then echo ALREADY_ACTIVE; exit 0; fi
nohup python -u run_flash_after_winner.py >>"$j/queue.log" 2>&1 </dev/null &
echo $! >"$j/queue.pid"
sleep 2
ps -p "$(cat "$j/queue.pid")" -o pid,stat,cmd
cat "$j/recipe.json"; cat "$j/queue.log"
