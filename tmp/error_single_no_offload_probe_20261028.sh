set -eu
p=/data1/shenglin.zeng/sdxl_eval_uniform_20261025/outputs/turning_boundary_no_offload_1gpu_probe_20261028
grep -n -C 5 -E 'OutOfMemory|CUDA out of memory|Traceback|RuntimeError|ValueError|AssertionError|UPDATE|LOAD' "$p/normalized_u.log" | tail -70 || true
cat "$p/workflow.jsonl"
df -h /data1/shenglin.zeng | tail -1
