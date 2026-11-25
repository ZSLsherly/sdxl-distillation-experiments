set -e
R=/data1/shenglin.zeng/dmd2_matched_10k_20261119
if ps -p 455398,455399,455400,455401,455222 -o stat= | grep -v '^Z' | grep -q '[^ ]'; then echo 'OWN_WORKERS_STILL_EXITING'; exit 1; fi
CUDA_VISIBLE_DEVICES='' "$R/env/bin/python" "$R/code/test_cpu.py"
mkdir -p "$R/failed_fsdp_forward_20261119"
[ ! -e "$R/runs/smoke_micro16" ] || mv "$R/runs/smoke_micro16" "$R/failed_fsdp_forward_20261119/"
[ ! -e "$R/logs/smoke_micro16.log" ] || mv "$R/logs/smoke_micro16.log" "$R/failed_fsdp_forward_20261119/"
[ ! -e "$R/logs/queue.log" ] || mv "$R/logs/queue.log" "$R/failed_fsdp_forward_20261119/"
rm -f "$R/STOP_REQUEST"
nohup "$R/env/bin/python" -u "$R/code/run_guard.py" --root "$R" > "$R/logs/queue.log" 2>&1 < /dev/null &
echo "GUARD_PID=$!"
sleep 2
cat "$R/logs/queue.log"

