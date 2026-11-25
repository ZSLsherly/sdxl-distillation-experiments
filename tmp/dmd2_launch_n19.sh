set -e
R=/data1/shenglin.zeng/dmd2_matched_10k_20261119
mkdir -p "$R/logs"
nohup "$R/env/bin/python" -u "$R/code/run_guard.py" --root "$R" > "$R/logs/queue.log" 2>&1 < /dev/null &
echo "GUARD_PID=$!"
sleep 3
cat "$R/logs/queue.log"
