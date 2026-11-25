set -eu
root=/data5/shenglin.zeng/image_crossdistill
if [ -f "$root/logs/download.pid" ]; then
 pid=$(cat "$root/logs/download.pid")
 if [ -r "/proc/$pid/cmdline" ] && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -Fq 'src/image_download_model.py'; then
  kill -TERM "$pid"
  echo "Stopped own image model downloader PID $pid"
 else
  echo 'Downloader is no longer running'
 fi
fi
tail -n 10 "$root/logs/download.log"
du -sh "$root/models"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
