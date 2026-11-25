set -eu
for root in /data1/shenglin.zeng/sdxl_eval_uniform_20261025 /data5/shenglin.zeng/sdxl_crossdistill; do
  p="$root/outputs/turning_batch64_g160_20261027"
  if test -d "$p"; then
    echo "PROJECT=$p"
    df -h "$p" | tail -1
    find "$p" -maxdepth 2 -type f \( -name status.json -o -name workflow_complete.json \) -print
    pgrep -u "$(id -u)" -af '[t]urning_batch64_workflow.py' || true
  fi
done
