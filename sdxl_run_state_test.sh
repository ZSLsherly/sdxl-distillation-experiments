set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN OMP_NUM_THREADS=2
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 src/sdxl_state_test.py > logs/state_test_v1.log 2>&1
cat logs/state_test_v1.log
