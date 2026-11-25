cd /data5/shenglin.zeng/sdxl_crossdistill
if test -f logs/gan_calibration.exit; then
  echo "CALIBRATION_ALREADY_EXITED $(cat logs/gan_calibration.exit)"
else
  pids=$(pgrep -f '^(/data5/shenglin.zeng/envs/crossdistill/bin/python[^ ]* |python[^ ]* |/data5/shenglin.zeng/envs/crossdistill/bin/torchrun |torchrun ).*src/sdxl_gan_train.py --run gan_stage_pilot/calibration --calibrate-only' || true)
  if test -n "$pids"; then
    ps -o pid,user,args -p $(echo "$pids" | tr '\n' ' ')
    kill -TERM $pids
  fi
  echo 'Calibration interrupted for research and protocol review; no ablation training launched.' > outputs/gan_stage_pilot/review_checkpoint.txt
fi
pgrep -af '[s]dxl_gan_train.py' || true
