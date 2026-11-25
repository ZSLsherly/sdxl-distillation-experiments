set -euo pipefail
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python -m py_compile sdxl_parallel/train.py sdxl_parallel/training_state.py
python -m unittest discover -s sdxl_parallel -p test_batching.py
export CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1
torchrun --standalone --nproc_per_node=4 sdxl_parallel/test_training_state_gpu.py "outputs/reviews/resume_test_$(date +%Y%m%dT%H%M%S)"
df -h /data5/shenglin.zeng
