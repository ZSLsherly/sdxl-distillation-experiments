set -eu
source /data5/shenglin.zeng/env.sh
root="$CROSSDISTILL_ROOT/sdxl_crossdistill"
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --standalone --nnodes=1 --nproc-per-node=4 "$root/src/sdxl_fsdp_probe.py"
