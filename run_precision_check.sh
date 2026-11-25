set -eu
source /data5/shenglin.zeng/env.sh
cd "$CROSSDISTILL_ROOT"
torchrun --standalone --nproc_per_node=4 fsdp_precision_check.py > logs/fsdp_precision_check.log 2>&1
tail -8 logs/fsdp_precision_check.log
tail -3 logs/segmented_download.log
