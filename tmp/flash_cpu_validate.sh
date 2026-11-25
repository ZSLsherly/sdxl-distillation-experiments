set -eu
. /root/sdxl_research/env.sh
c="$SDXL_ROOT/tmp/flash_gan_g160_20261102/code"
mkdir -p "$c"
tar -xzf "$SDXL_ROOT/tmp/bc_coco_winner_code.tar.gz" -C "$c"
tar -xzf "$SDXL_ROOT/tmp/flash_gan_changes.tar.gz" -C "$c"
cd "$c"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH="$c"
python -m py_compile train.py boundary.py
python -m unittest test_paired_stages test_fake_tracking
python smoke_boundary_cpu.py --flash --output "$c/flash_cpu_smoke.json" | tail -4
python - <<'PY'
import torch,os
from pathlib import Path
from sam_pixel import SAMDiscriminator
from stage_policy import make_stage_plan
from train import arguments
from launch import flags
import json,sys
torch.set_num_threads(2)
model=SAMDiscriminator(Path(os.environ['SDXL_WORK'])/'flash_gan_inputs/sam-vit-base')
print('PRETRAINED SAM LOADED ON CPU', len(model.backbone.layers),model.size)
for name in ('B','C'):
    cfg=json.loads(Path(f'configs/turning_{name}_quota_a800_g160.json').read_text())
    cfg.update(variant='joint4',discriminator='sam',gan_view='whole',gan_augment='none',flash_terminal_gan=True,both_middle=True,gan_weight=.005,micro_batch=4)
    sys.argv=['train','--run','cpu_check',*flags(cfg)]
    a=arguments()
    assert a.both_middle and a.flash_terminal_gan and a.batch==64 and a.max_student_updates==160 and a.fake_updates==2
    print('VALIDATED CONFIG',name)
PY
