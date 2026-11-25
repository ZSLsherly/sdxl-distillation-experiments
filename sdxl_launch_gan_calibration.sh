source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN OMP_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1
mkdir -p outputs/gan_stage_pilot logs
test -f outputs/hparam_pilot/lr5e7_pcm10/export/unet/diffusion_pytorch_model.safetensors || exit 21
test ! -f logs/gan_calibration.exit || exit 22
nohup bash -c 'torchrun --standalone --nproc_per_node=4 src/sdxl_gan_train.py --run gan_stage_pilot/calibration --calibrate-only > logs/gan_calibration.log 2>&1; echo $? > logs/gan_calibration.exit' < /dev/null > logs/gan_calibration.launch.log 2>&1 &
echo CALIBRATION_LAUNCH_PID=$!
