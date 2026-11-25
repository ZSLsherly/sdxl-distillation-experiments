set -eu
root=/data3/shenglin.zeng/sdxl_turning_ablation_20261028
p="$root/outputs/smoke_four_gpu_20261028"
date -Is
tail -n 4 "$p/workflow.log" 2>/dev/null || true
tail -n 6 "$p/normalized_u.log" 2>/dev/null || true
tail -n 2 "$p/normalized_u/rank0.jsonl" 2>/dev/null || true
nvidia-smi --id=1,2,4,5 --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --id=1,2,4,5 --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader,nounits
