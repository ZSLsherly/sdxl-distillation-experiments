set -e
R=/data1/shenglin.zeng/dmd2_matched_10k_20261119
E="$R/coco10k_official_20261121"
mkdir -p "$E/logs"
nohup "$R/env/bin/python" -u "$E/code/run_queue.py" "$E" >> "$E/logs/queue.log" 2>&1 < /dev/null &
echo "EVAL_QUEUE_PID=$!"
sleep 2
tail -n 6 "$E/logs/queue.log"

