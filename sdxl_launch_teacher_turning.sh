source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
nohup bash -c '
for i in $(seq 1 360);do
 if grep -qx "TRIAL_BEGIN lr25e8_pcm1" logs/hparam_pipeline.log;then
  export CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1
  torchrun --standalone --nnodes=1 --nproc-per-node=4 src/sdxl_teacher_turning.py
  status=$?
  if [ "$status" != 0 ];then echo "$status" > logs/teacher_turning.exit;exit "$status";fi
  python src/sdxl_summarize_turning.py
  status=$?
  echo "$status" > logs/teacher_turning.exit
  exit "$status"
 fi
 sleep 5
done
echo 124 > logs/teacher_turning.exit
' > logs/teacher_turning.log 2>&1 < /dev/null &
echo $! > logs/teacher_turning.pid
cat logs/teacher_turning.pid
