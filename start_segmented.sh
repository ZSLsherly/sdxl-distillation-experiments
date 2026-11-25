set -eu
ROOT=/data5/shenglin.zeng
if [ -r /proc/2974288/cmdline ] && tr '\0' ' ' < /proc/2974288/cmdline | grep -q '/data5/shenglin.zeng/download_models.py'; then kill -TERM 2974288; fi
sleep 2
source "$ROOT/env.sh"
nohup env PYTHONPATH="$ROOT/envs/download-tools" /data4/liushaozhen/miniforge3/bin/python "$ROOT/segmented_download.py" > "$ROOT/logs/segmented_download.log" 2>&1 < /dev/null &
printf 'SEGMENTED_PID=%s\n' "$!"
tail -5 "$ROOT/logs/install_deps.log"
