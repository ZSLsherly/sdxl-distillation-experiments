cd /data5/shenglin.zeng/sdxl_crossdistill
touch outputs/parallel_probe/mb8_adamcpu/STOP_REQUEST
nohup bash -c '
  cd /data5/shenglin.zeng/sdxl_crossdistill
  while [ ! -f logs/parallel_mb8_adamcpu.exit ]; do sleep 2; done
  test "$(cat logs/parallel_mb8_adamcpu.exit)" = 0 || exit 1
  source /data5/shenglin.zeng/env.sh
  CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1 torchrun --standalone --nproc_per_node=2 sdxl_parallel/train.py --run parallel_probe/mb32_adamcpu --variant joint4 --batch 64 --micro-batch 32 --rollout-batch 32 --offload-student-optimizer --benchmark --benchmark-batch 64 --profile > logs/parallel_mb32_adamcpu.log 2>&1
  echo $? > logs/parallel_mb32_adamcpu.exit
' </dev/null >/dev/null 2>&1 &
echo PROBE32_LAUNCH_PID=$!
