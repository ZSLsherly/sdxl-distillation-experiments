root=/root/autodl-tmp/sdxl_runs
p="$root/outputs/a800_B_batch_probe_20261101"
date -Is
tail -n 5 "$root/logs/a800_batch_probe.log"
for f in "$p"/mb*.log; do test ! -f "$f" || { echo "$f";tail -n 7 "$f"; };done
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
