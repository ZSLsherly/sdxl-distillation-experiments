date
cat /proc/2492557/io 2>/dev/null || true
cat /proc/pressure/io
readlink -f /data1/shenglin.zeng/sdxl_eval_uniform_20261025/models/sdxl-base-1.0
ls -lh /data1/shenglin.zeng/sdxl_eval_uniform_20261025/models/sdxl-base-1.0/unet
ls -l /proc/2492557/fd 2>/dev/null | grep -E 'safetensors|\.py|\.bin|\.json' || true
command -v iostat >/dev/null && iostat -dx 1 2 | tail -20 || true
