set -eu
root=/data5/shenglin.zeng/sdxl_crossdistill
for name in download data_download; do
 pid=$(cat "$root/logs/$name.pid")
 if [ -r "/proc/$pid/cmdline" ] && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -Eq 'src/sdxl_download_(model|data).py'; then
  kill -TERM "$pid"
  echo "Stopped own downloader $pid to enable byte-level resume"
 fi
done
