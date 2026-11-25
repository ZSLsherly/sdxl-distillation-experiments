set -eu
root=/data3/shenglin.zeng/sdxl_turning_ablation_20261028
date -Is
tail -n 5 "$root/pipeline_expandable_retry1_20261029.log" 2>/dev/null || true
for f in "$root/outputs/smoke_four_gpu_expandable_retry1_20261029/workflow.log" "$root/outputs/smoke_four_gpu_expandable_retry1_20261029/normalized_u.log" "$root/outputs/batch64_workflow.log";do
 if test -f "$f";then echo "LOG=$f";tail -n 9 "$f";fi
done
tail -n 2 "$root/outputs/smoke_four_gpu_expandable_retry1_20261029/normalized_u/rank0.jsonl" 2>/dev/null || true
nvidia-smi --id=1,2,4,5 --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --id=1,2,4,5 --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader,nounits
