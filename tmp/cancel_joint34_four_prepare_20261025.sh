set -e
printf 'CANCEL PREPARATION ONLY\n'
pids=$(timeout 10 pgrep -u "$(id -u)" -f '^env/bin/python prepare_joint34_four_gpu_20261025.py$' || true)
for pid in $pids; do kill -TERM "$pid"; printf 'STOPPED_PREPARATION_PID=%s\n' "$pid"; done
printf 'NO FOUR GPU TRAINING WAS LAUNCHED\n'
