set -eu
root=/data3/shenglin.zeng/sdxl_turning_ablation_20261028
date -Is
tail -n 15 "$root/pipeline_20261029.log" 2>/dev/null || true
tail -n 15 "$root/pipeline_retry_20261029.log" 2>/dev/null || true
tail -n 18 "$root/pipeline_compat_20261029.log" 2>/dev/null || true
tail -n 22 "$root/pipeline_offline_20261029.log" 2>/dev/null || true
ls -lh "$root/transfers" | tail -7
cat "$root/storage_estimate.json" 2>/dev/null || true
for out in smoke_four_gpu_20261028 turning_batch64_g160_4gpu_20261029;do
 for f in "$root/outputs/$out/workflow.log" "$root/outputs/$out/normalized_u.log";do test ! -f "$f" || tail -n 8 "$f";done
done
tail -n 4 "$root/outputs/batch64_workflow.log" 2>/dev/null || true
nvidia-smi --id=1,2,4,5 --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --id=1,2,4,5 --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader,nounits
