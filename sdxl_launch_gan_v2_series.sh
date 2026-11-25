set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
test "$(cat logs/gan_v2_preflight.exit)" = 0
test ! -f logs/gan_v2_series.pid
test ! -f outputs/gan_stage_pilot_v2/STOP_REQUEST
python -m py_compile src/sdxl_gan_evaluate_v2.py src/sdxl_gan_render_report_v2.py
nohup bash src/sdxl_gan_runner_v2.sh < /dev/null > logs/gan_v2_series.log 2>&1 &
echo $! > logs/gan_v2_series.pid
echo V2_SERIES_PID=$!
