set -eu
cd /data1/shenglin.zeng/sdxl_eval_uniform_20261025
date -Is
p=outputs/turning_boundary_no_offload_2gpu_20261027
if test -d "$p";then
 tail -n 1 "$p/workflow.jsonl" 2>/dev/null || true
 for arm in normalized_u raw_angle;do
  echo "ARM=$arm"
  tail -n 4 "$p/$arm.log" 2>/dev/null || true
  tail -n 2 "$p/$arm/rank0.jsonl" 2>/dev/null || true
 done
fi
nvidia-smi --id=4,7 --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --id=4,7 --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader,nounits
