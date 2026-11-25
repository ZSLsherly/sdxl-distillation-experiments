set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN OMP_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1
test ! -f logs/gan_v2_preflight.pid
python -m py_compile src/sdxl_gan_train_v2.py src/sdxl_gan_policy_v2.py src/sdxl_gan_audit_v2.py
python src/sdxl_gan_policy_v2.py
nohup bash -c 'torchrun --standalone --nproc_per_node=4 src/sdxl_gan_train_v2.py --run gan_stage_pilot_v2/preflight --variant joint34_matched --student-updates 3 --preflight > logs/gan_v2_preflight.log 2>&1; rc=$?; if test "$rc" = 0; then python src/sdxl_gan_audit_v2.py outputs/gan_stage_pilot_v2/preflight --preflight >> logs/gan_v2_preflight.log 2>&1; rc=$?; fi; echo "$rc" > logs/gan_v2_preflight.exit' < /dev/null > logs/gan_v2_preflight.launch.log 2>&1 &
echo $! > logs/gan_v2_preflight.pid
echo V2_PREFLIGHT_PID=$!
