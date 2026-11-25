set -eu
root=/data1/shenglin.zeng/teacher_turning_20261026/smoke_gpu0
date -Is
tail -8 "$root/worker.log"
find "$root" -maxdepth 1 -type f -printf '%f %s bytes\n'
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits
