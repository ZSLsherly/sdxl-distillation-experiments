set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node=4 sdxl_parallel/check_exit_sync.py
