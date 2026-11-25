date -u
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits
R=/data/shenglin.zeng/a50k_20261123
if test -f "$R/records/prepare_status.json"; then
 cat "$R/records/prepare_status.json"
 tail -4 "$R/records/build.log"
 if test -f "$R/datasets/parallel_50k_seed20261018/validation.json"; then cat "$R/datasets/parallel_50k_seed20261018/validation.json"; stat -c '%s %n' "$R/datasets/parallel_50k_seed20261018/prepared.pt"; fi
fi
