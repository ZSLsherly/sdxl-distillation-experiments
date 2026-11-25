set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
nohup python src/sdxl_validation_minimal_supervisor.py > logs/v4_minimal_supervisor.log 2>&1 < /dev/null &
echo MINIMAL_SUPERVISOR_PID_$!
