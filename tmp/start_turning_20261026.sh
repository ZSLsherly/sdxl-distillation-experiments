set -euo pipefail
root=/data1/shenglin.zeng/teacher_turning_20261026
cd "$root"
test -f setup_complete.json
export LD_LIBRARY_PATH="$root/runtime/lib:${LD_LIBRARY_PATH:-}"
nvidia-smi --query-gpu=index,uuid,memory.free,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits
if test -f guard.pid && kill -0 "$(cat guard.pid)" 2>/dev/null; then
  state=$(ps -o stat= -p "$(cat guard.pid)")
  case "$state" in Z*) ;; *) echo 'Existing sampling guard is alive'; exit 1;; esac
fi
nohup nice -n 10 env/bin/python -u run_guard.py > guard.log 2>&1 < /dev/null &
echo $! > guard.pid
printf 'sampling_guard_pid ';cat guard.pid
