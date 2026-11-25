source /data5/shenglin.zeng/env.sh
root=/data5/shenglin.zeng/sdxl_crossdistill
cd "$root"
export CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1
nohup bash -c '
set -e
root=/data5/shenglin.zeng/sdxl_crossdistill
for i in $(seq 1 120); do
 if [ -f logs/teacher_stages.exit ]; then test "$(cat logs/teacher_stages.exit)" = 0; break;fi
 sleep 5
done
test "$(cat logs/teacher_stages.exit)" = 0
run_trial() {
 name=$1;lr=$2;weight=$3
 echo "TRIAL_BEGIN $name"
 torchrun --standalone --nnodes=1 --nproc-per-node=4 src/sdxl_hparam_train.py --run "hparam_pilot/$name" --updates 72 --lr-student "$lr" --lr-fake 5e-7 --pcm-weight "$weight" --fake-ratio 5 --crossover 749 --train-samples 96 > "logs/pilot_$name.log" 2>&1
 python src/sdxl_hparam_export.py --run "hparam_pilot/$name" > "logs/pilot_export_$name.log" 2>&1
 echo "TRIAL_COMPLETE $name"
}
run_trial lr5e7_pcm10 5e-7 10
run_trial lr25e8_pcm10 2.5e-7 10
run_trial lr5e7_pcm1 5e-7 1
run_trial lr25e8_pcm1 2.5e-7 1
echo PILOT_TRAINING_COMPLETE
' > logs/hparam_pipeline.log 2>&1 < /dev/null &
echo $! > logs/hparam_pipeline.pid
cat logs/hparam_pipeline.pid
