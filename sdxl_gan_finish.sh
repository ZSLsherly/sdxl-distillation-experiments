set -euo pipefail
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
trap 'rc=$?; echo "$rc" > logs/gan_assessment.exit' EXIT
echo WAITING_FOR_FOUR_TRAINING_TRIALS
while ! test -f logs/gan_stage_pilot.exit; do
  pid=$(cat logs/gan_stage_pilot.pid)
  if ! kill -0 "$pid" 2>/dev/null; then echo TRAINING_RUNNER_DISAPPEARED; exit 25; fi
  sleep 30
done
rc=$(cat logs/gan_stage_pilot.exit)
if test "$rc" != 0; then echo "TRAINING_FAILED_EXIT=$rc; assessment not started"; exit 26; fi
export OMP_NUM_THREADS=4
echo ASSESSMENT_START
python src/sdxl_gan_evaluate.py
python src/sdxl_gan_render_report.py
echo REPORT_COMPLETE
