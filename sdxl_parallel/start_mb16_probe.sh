cd /data5/shenglin.zeng/sdxl_crossdistill
test "$(cat logs/parallel_mb4_vaecp.exit)" = 0 || exit 1
nohup bash -c 'source /data5/shenglin.zeng/env.sh; CUDA_VISIBLE_DEVICES=2,3 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1 torchrun --standalone --nproc_per_node=2 sdxl_parallel/train.py --run parallel_probe/mb16_vae4 --variant joint4 --batch 64 --micro-batch 16 --rollout-batch 16 --benchmark --benchmark-batch 64 --profile > logs/parallel_mb16_vae4.log 2>&1; echo $? > logs/parallel_mb16_vae4.exit' </dev/null >/dev/null 2>&1 &
echo BENCH16_PID=$!
