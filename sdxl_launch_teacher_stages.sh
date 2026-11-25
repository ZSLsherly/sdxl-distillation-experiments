source /data5/shenglin.zeng/env.sh
root=/data5/shenglin.zeng/sdxl_crossdistill
cd "$root"
export CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1
nohup bash -c 'torchrun --standalone --nnodes=1 --nproc-per-node=4 src/sdxl_teacher_stages.py; echo $? > logs/teacher_stages.exit' > logs/teacher_stages.log 2>&1 < /dev/null &
echo $! > logs/teacher_stages.pid
cat logs/teacher_stages.pid
