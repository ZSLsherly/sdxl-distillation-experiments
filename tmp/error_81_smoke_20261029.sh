set -eu
p=/data3/shenglin.zeng/sdxl_turning_ablation_20261028/outputs/smoke_four_gpu_20261028/normalized_u.log
grep -n -B 8 -A 2 -E 'OutOfMemoryError|out of memory|RuntimeError|ValueError' "$p" | head -75 || true
