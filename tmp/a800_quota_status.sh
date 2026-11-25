. /root/sdxl_research/env.sh
date -Is
root="$SDXL_ROOT/outputs/turning_BC_interval_quota_20261101"
tail -n 4 "$SDXL_WORK/logs/quota_bc_queue.log"
for arm in B C; do
 test ! -f "$root/$arm.log" || { echo "ARM=$arm";tail -n 5 "$root/$arm.log"; }
 test ! -f "$root/$arm/status.json" || cat "$root/$arm/status.json"
done
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
