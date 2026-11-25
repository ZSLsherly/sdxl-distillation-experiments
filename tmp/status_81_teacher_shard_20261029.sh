set -eu
root=/data3/shenglin.zeng/sdxl_turning_ablation_20261028
date -Is
tail -n 5 "$root/pipeline_teacher_shard_20261029.log" 2>/dev/null || true
p="$root/outputs/smoke_four_gpu_teacher_shard_20261029"
for f in "$p/workflow.log" "$p/normalized_u_teacher_check.log" "$p/normalized_u.log" "$root/outputs/batch64_teacher_shard_workflow.log";do
 if test -f "$f";then echo "LOG=$f";tail -n 6 "$f";fi
done
tail -n 2 "$p/normalized_u/rank0.jsonl" 2>/dev/null || true
for arm in normalized_u raw_angle;do
 for phase in validation g160;do
  file="$root/outputs/turning_batch64_g160_4gpu_teacher_shard_20261029/${arm}_${phase}/status.json"
  if test -f "$file";then echo "$arm $phase";cat "$file";fi
 done
done
nvidia-smi --id=1,2,4,5 --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --id=1,2,4,5 --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader,nounits
