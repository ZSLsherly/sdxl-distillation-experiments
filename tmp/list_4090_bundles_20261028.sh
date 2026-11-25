set -eu
root=/data1/shenglin.zeng/sdxl_eval_uniform_20261025
ls -lh "$root/transfers" 2>/dev/null || true
ls -lh "$root/runtime/bin/python" "$root/env/bin/python" 2>/dev/null || true
df -h "$root" | tail -1
