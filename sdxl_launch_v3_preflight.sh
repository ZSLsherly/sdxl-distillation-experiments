set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python -m py_compile src/sdxl_gan_train_v3.py src/sdxl_pixel_v3.py
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN OMP_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1
nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 src/sdxl_gan_train_v3.py --run gan_stage_v3/preflight --variant joint4 --preflight --batch 4 > logs/gan_v3_preflight.log 2>&1; echo $? > logs/gan_v3_preflight.exit' < /dev/null > logs/gan_v3_preflight_launcher.log 2>&1 &
echo PREFLIGHT_PID_$!
