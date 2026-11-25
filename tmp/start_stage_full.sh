set -euo pipefail
root=/data1/shenglin.zeng/teacher_turning_20261026
cd "$root"
test -f stage_smoke/complete.json
export LD_LIBRARY_PATH="$root/runtime/lib:${LD_LIBRARY_PATH:-}"
if test -f stage_intervention/guard.pid && kill -0 "$(cat stage_intervention/guard.pid)" 2>/dev/null;then echo 'Guard already alive';exit 1;fi
mkdir -p stage_intervention
nohup nice -n 10 env/bin/python -u guard_stage_intervention.py > stage_intervention/guard.log 2>&1 < /dev/null &
echo $! > stage_intervention/guard.pid
cat stage_intervention/guard.pid
