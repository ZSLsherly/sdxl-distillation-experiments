set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
test -f logs/gan_stage_pilot.pid
test ! -f logs/gan_assessment.pid
python -m py_compile src/sdxl_gan_evaluate.py src/sdxl_gan_render_report.py
nohup bash src/sdxl_gan_finish.sh < /dev/null > logs/gan_assessment.log 2>&1 &
echo $! > logs/gan_assessment.pid
echo GAN_ASSESSMENT_QUEUED_PID=$!
