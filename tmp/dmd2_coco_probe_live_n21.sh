E=/data1/shenglin.zeng/dmd2_matched_10k_20261119/coco10k_official_20261121
ps -eo pid,etime,stat,rss,args | grep -E '[p]ython.*(evaluate_dmd2|coco10k_official_20261121/code/run_queue)' || true
stat -c '%y %n' "$E/logs/prepare.log" "$E/logs/probe.log"
nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader
