set -eu
. /root/sdxl_research/env.sh
d=/root/autodl-tmp/h20_direct_after_coco_20261103
cd "$d"
export PYTHONPATH="$d:/root/autodl-tmp/h20_C_mix1_formal_20261103"
CUDA_VISIBLE_DEVICES='' python - <<'PY'
from pathlib import Path
import hashlib,json,os,shlex,sys
import train
for file in ('short_command.txt','formal_command.txt'):
    cmd=shlex.split(Path(file).read_text())
    sys.argv=['train.py']+cmd[cmd.index('--root'):]
    a=train.arguments()
    assert a.direct_terminal_gan and not a.flash_terminal_gan
    assert (a.micro_batch,a.batch,a.vae_batch,a.fake_student_mix,a.fake_mix_start_g)==(16,64,1,.01,1)
    print(file,a.max_student_updates,a.benchmark,a.vae_batch)
baseline=Path(os.environ['SDXL_ROOT'])/'outputs/h20_C_mix1_terminal_GAN_G160'
hashes=json.loads((baseline/'source_sha256.json').read_text())
for name,digest in hashes.items():assert hashlib.sha256((baseline/'source'/name).read_bytes()).hexdigest()==digest,name
print('matched baseline source files',len(hashes))
PY
ps -p "$(cat queue.pid)" -o pid,etime,stat,args
tail -2 queue.log
