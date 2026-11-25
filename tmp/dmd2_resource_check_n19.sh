set -eu
hostname
date -Iseconds
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits
df -h /data* 2>/dev/null || true
for r in /data1/shenglin.zeng /data3/shenglin.zeng /data5/shenglin.zeng; do
 if test -d "$r"; then echo "$r"; ls "$r"; fi
done
command -v rclone || true
