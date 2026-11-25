set -euo pipefail
root=/data1/shenglin.zeng/teacher_turning_20261026
cd "$root"
export LD_LIBRARY_PATH="$root/runtime/lib:${LD_LIBRARY_PATH:-}"
if test -f results512/guard.pid && kill -0 "$(cat results512/guard.pid)" 2>/dev/null; then echo 'Guard already alive';exit 1;fi
mkdir -p results512
nohup nice -n 10 env/bin/python -u guard_turning512.py --batch 8 --gpus 0,1,4 > results512/guard.log 2>&1 < /dev/null &
echo $! > results512/guard.pid
cat results512/guard.pid
