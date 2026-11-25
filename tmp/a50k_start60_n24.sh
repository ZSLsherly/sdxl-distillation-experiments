set -eu
R=/data5/shenglin.zeng/a50k_20261124
cd "$R"
mkdir -p logs cache/TMPDIR models
test -e models/sdxl-base-1.0 || ln -s /data5/shenglin.zeng/sdxl_crossdistill/models/sdxl-base-1.0 models/sdxl-base-1.0
export TMPDIR="$R/cache/TMPDIR" CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=4
PY=/data5/shenglin.zeng/envs/crossdistill/bin/python
export PYTHONPATH="$R/code"
"$PY" -c 'from launch import gpu_check; print(gpu_check([0,1,2,3],36)); import train'
nohup bash -c '
set -eu
exec 9>bootstrap.lock
flock -n 9
py=/data5/shenglin.zeng/envs/crossdistill/bin/python
if ! test -f logs/restore_test_pass; then
 "$py" -m torch.distributed.run --standalone --nproc_per_node=4 code/test_training_state_gpu.py "cache/test_restore_$$" > logs/restore_test.log 2>&1
 touch logs/restore_test_pass
fi
while ! test -f datasets/已核验传输.txt; do sleep 10; done
exec "$py" -u code/queue50.py
' > logs/queue.log 2>&1 < /dev/null &
echo "QUEUE_PID=$!"
