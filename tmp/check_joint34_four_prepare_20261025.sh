set -e
printf 'PARENT MEMORY\n'
cat /sys/fs/cgroup/user.slice/memory.current /sys/fs/cgroup/user.slice/memory.high
printf 'OWN MEMORY\n'
cat /sys/fs/cgroup/user.slice/user-1009.slice/memory.current
printf 'LOAD\n'
cat /proc/loadavg
printf 'NEW RUN FILES\n'
ls -l /data1/shenglin.zeng/sdxl_eval_uniform_20261025/outputs/joint34_uniform_10k_4gpu_20261025
printf 'GPU PIDS\n'
timeout 10 nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
