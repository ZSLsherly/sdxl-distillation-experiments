set -eu
root=/data1/shenglin.zeng/teacher_turning_20261026
date -Is
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits
nvidia-smi --query-gpu=index,uuid,memory.free,utilization.gpu --format=csv,noheader,nounits
ps -eo pid,ppid,etime,stat,args | grep -E 'teacher_turning_20261026|python3 -|tarfile' | grep -v grep
du -sh "$root/env" "$root/runtime" "$root/models" 2>/dev/null || true
test ! -f "$root/setup_complete.json" || cat "$root/setup_complete.json"
