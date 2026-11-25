set -eu
ROOT=/data1/shenglin.zeng/sdxl_eval_uniform_20261025
p="$ROOT/outputs/turning_boundary_no_offload_1gpu_probe_20261028"
date -Is
tail -n 6 "$p/workflow.log" 2>/dev/null || true
tail -n 12 "$p/normalized_u.log" 2>/dev/null || true
tail -n 2 "$p/normalized_u/rank0.jsonl" 2>/dev/null || true
nvidia-smi --id=4 --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --id=4 --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader,nounits
