set -eu
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
nohup bash -c '
  out=outputs/coco10k_g160_20261020
  while [ "$(cat "$out/verify.exit")" = running ]; do sleep 10; done
  if [ "$(cat "$out/verify.exit")" != 0 ]; then echo PREFLIGHT_FAILED > "$out/run.exit"; exit 1; fi
  bash sdxl_coco10k/run.sh > "$out/run.log" 2>&1
  echo $? > "$out/run.exit"
' </dev/null >/dev/null 2>&1 &
echo "$!" > outputs/coco10k_g160_20261020/runner_pid.txt
echo "EVALUATION_RUNNER_PID $!"
