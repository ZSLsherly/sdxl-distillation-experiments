cd /data5/shenglin.zeng/sdxl_crossdistill
for f in outputs/research_small_20261022/*.exit; do test -f "$f" && echo "$f $(cat "$f")"; done
tail -n 14 logs/research_small_state_20261022.log
nvidia-smi -i 0,3 --query-gpu=index,memory.used,utilization.gpu --format=csv
