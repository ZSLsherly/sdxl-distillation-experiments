set -eu
r=/data1/shenglin.zeng/coco_missing_20261111
date -Iseconds
tail -8 "$r/queue.log"
for n in A_pcm10_mix1 normalized_u_g160 raw_angle_g160; do
 test ! -f "$r/results/$n/status_0.json" || cat "$r/results/$n/status_0.json"
 test ! -f "$r/results/$n/worker.log" || tail -4 "$r/results/$n/worker.log"
done
nvidia-smi -i 4 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
ps -p "$(cat "$r/queue.pid")" -o pid,stat,etime,args
