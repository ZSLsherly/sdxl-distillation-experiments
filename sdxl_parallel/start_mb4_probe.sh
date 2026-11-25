cd /data5/shenglin.zeng/sdxl_crossdistill
nohup bash -c 'source /data5/shenglin.zeng/env.sh; CUDA_VISIBLE_DEVICES=2,3 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1 torchrun --standalone --nproc_per_node=2 sdxl_parallel/train.py --run parallel_probe/mb4_vaecp --variant joint4 --batch 64 --micro-batch 4 --rollout-batch 8 --benchmark --benchmark-batch 64 --profile > logs/parallel_mb4_vaecp.log 2>&1; echo $? > logs/parallel_mb4_vaecp.exit' </dev/null >/dev/null 2>&1 &
echo BENCH4_PID=$!
