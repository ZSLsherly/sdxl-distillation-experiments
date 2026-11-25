source /data5/shenglin.zeng/env.sh
root=/data5/shenglin.zeng/sdxl_crossdistill
for i in $(seq 1 72);do
 if grep -qx PILOT_TRAINING_COMPLETE "$root/logs/hparam_pipeline.log";then
  python "$root/src/sdxl_audit_pilot.py"
  exit $?
 fi
 sleep 5
done
exit 124
