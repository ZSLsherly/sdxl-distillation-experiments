set -e
printf 'GPU AFTER STOP\n'
timeout 12 nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
printf 'MEMORY LIMITS\n'
for p in /sys/fs/cgroup/user.slice/user-1009.slice/session-36114.scope /sys/fs/cgroup/user.slice/user-1009.slice /sys/fs/cgroup/user.slice; do
  printf '%s\n' "$p"
  for f in memory.current memory.high memory.max memory.swap.max memory.events cpu.max; do
    if test -f "$p/$f"; then printf '%s ' "$f"; cat "$p/$f"; fi
  done
done
printf 'TOOLS\n'
command -v py-spy || true
command -v gdb || true
printf 'MEMORY\n'
free -h
