#!/usr/bin/env bash
# Run only after selecting an available GPU pair / stopping the old experiment.
# MICRO_BATCH=4 bash sdxl_parallel/run_pair.sh
set -euo pipefail
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1
micro_batch=${MICRO_BATCH:-16}
rollout_batch=${ROLLOUT_BATCH:-16}
max_updates=${MAX_UPDATES:-160}
data_dir=${DATA_DIR:-datasets/parallel_10k_seed20261018}
run_stamp=$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S)
mkdir -p logs
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 sdxl_parallel/train.py \
  --run "gan_stage_parallel/joint4_${run_stamp}_mb${micro_batch}" --variant joint4 \
  --data-dir "$data_dir" --batch 64 --micro-batch "$micro_batch" --rollout-batch "$rollout_batch" \
  --offload-student-optimizer --max-student-updates "$max_updates" --save-every-updates 0 --snapshot-seconds 1800 --eval-every-samples 128 --hours 24 > "logs/parallel_joint4_${run_stamp}.log" 2>&1 &
joint_pid=$!
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 sdxl_parallel/train.py \
  --run "gan_stage_parallel/gan4_${run_stamp}_mb${micro_batch}" --variant gan4 \
  --data-dir "$data_dir" --batch 64 --micro-batch "$micro_batch" --rollout-batch "$rollout_batch" \
  --offload-student-optimizer --max-student-updates "$max_updates" --save-every-updates 0 --snapshot-seconds 1800 --eval-every-samples 128 --hours 24 > "logs/parallel_gan4_${run_stamp}.log" 2>&1 &
gan_pid=$!
mkdir -p outputs/gan_stage_parallel
python - "$run_stamp" "$joint_pid" "$gan_pid" "$micro_batch" "$rollout_batch" "$max_updates" "$data_dir" <<'PY'
import json, sys
from pathlib import Path
stamp,joint,gan,micro,rollout,updates,data=sys.argv[1:]
Path('outputs/gan_stage_parallel/launch.json').write_text(json.dumps({
  'timestamp':stamp, 'microbatch_per_gpu':int(micro), 'rollout_batch':int(rollout),
  'effective_batch':64, 'max_student_updates':int(updates), 'save_every_updates':0, 'snapshot_seconds':1800,
  'eval_every_updates':2, 'data_dir':data,
  'runs':[{'variant':v,'run':f'gan_stage_parallel/{v}_{stamp}_mb{micro}',
           'gpus':gpus,'launcher_pid':int(pid)} for v,gpus,pid in
          [('joint4',[0,1],joint),('gan4',[2,3],gan)]]},indent=2))
PY
joint_rc=0
gan_rc=0
wait "$joint_pid" || joint_rc=$?
wait "$gan_pid" || gan_rc=$?
echo "joint4 exit=$joint_rc; gan4 exit=$gan_rc"
test "$joint_rc" = 0 && test "$gan_rc" = 0
