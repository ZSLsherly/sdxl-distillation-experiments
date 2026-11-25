set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
test ! -f logs/gan_stage_pilot.pid
test ! -f outputs/gan_stage_pilot/preflight/summary.json
test -f src/sdxl_gan_runner.sh
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
nohup bash src/sdxl_gan_runner.sh < /dev/null > logs/gan_stage_pilot.log 2>&1 &
echo $! > logs/gan_stage_pilot.pid
echo GAN_SERIES_LAUNCHED_PID=$!
