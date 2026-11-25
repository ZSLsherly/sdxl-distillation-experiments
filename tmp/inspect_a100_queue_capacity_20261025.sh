set -e
printf 'CPU MEMORY\n'
free -h
printf 'CGROUP\n'
cat /proc/self/cgroup
for p in /sys/fs/cgroup/user.slice /sys/fs/cgroup; do
 for f in memory.current memory.high memory.max memory.events; do
  if test -f "$p/$f"; then printf '%s/%s ' "$p" "$f"; cat "$p/$f"; fi
 done
done
printf 'GPU\n'
nvidia-smi --query-gpu=index,uuid,memory.free,utilization.gpu --format=csv,noheader,nounits
