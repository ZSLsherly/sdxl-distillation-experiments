set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
nohup bash -c 'bash src/sdxl_run_v4_validation.sh > logs/v4_validation.log 2>&1; echo $? > logs/v4_validation.exit' < /dev/null > logs/v4_validation_launcher.log 2>&1 &
echo VALIDATION_PID_$!
