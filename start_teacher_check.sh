set -eu
source /data5/shenglin.zeng/env.sh
cd "$CROSSDISTILL_ROOT"
nohup torchrun --standalone --nproc_per_node=1 teacher_forward_check.py > logs/teacher_forward_check.log 2>&1 < /dev/null &
printf 'TEACHER_PREFLIGHT_PID=%s\n' "$!"
