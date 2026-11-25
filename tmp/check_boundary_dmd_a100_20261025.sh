set -eu
date -Is
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv
free -h
df -h /data5/shenglin.zeng
cat /proc/self/cgroup
for p in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory/memory.limit_in_bytes /sys/fs/cgroup/memory/memory.usage_in_bytes; do
  if test -f "$p"; then printf '%s ' "$p"; cat "$p"; fi
done
ps -eo pid,ppid,user,rss,etime,comm --sort=-rss | head -18
ls -lh /data5/shenglin.zeng/sdxl_crossdistill/datasets/parallel_10k_seed20261018/{prepared.pt,manifest.json}
for p in /sys/fs/cgroup/memory/user.slice /sys/fs/cgroup/memory/user.slice/user-1017.slice; do
  for f in memory.limit_in_bytes memory.usage_in_bytes memory.failcnt; do
    if test -f "$p/$f"; then printf '%s ' "$p/$f"; cat "$p/$f"; fi
  done
done
