set -eu
src=/data5/shenglin.zeng/sdxl_crossdistill/models/baselines/dmd2/unet.safetensors
tail -c +5132533761 "$src" > /data5/shenglin.zeng/sdxl_crossdistill/tmp/ref_sem_xfer_20261104/model_tail.bin
