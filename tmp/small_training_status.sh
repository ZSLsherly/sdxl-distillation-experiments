cd /data5/shenglin.zeng/sdxl_crossdistill
date -Is
for f in outputs/research_small_20261022/*.exit; do test -f "$f" && echo "$f $(cat "$f")"; done
for run in legacy_b8 legacy_offload_b8 block_offload_b8 bucket_offload_b8; do
 test -f "outputs/research_small_20261022/$run/status.json" && cat "outputs/research_small_20261022/$run/status.json"
 test -f "logs/research_small_20261022_${run}.log" && tail -n 4 "logs/research_small_20261022_${run}.log"
done
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
