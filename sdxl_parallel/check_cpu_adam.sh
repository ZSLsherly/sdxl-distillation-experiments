set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node=2 sdxl_parallel/check_cpu_adam.py > outputs/gan_stage_joint34/cpu_adam_check.log 2>&1
cat outputs/gan_stage_joint34/cpu_adam_check.log
