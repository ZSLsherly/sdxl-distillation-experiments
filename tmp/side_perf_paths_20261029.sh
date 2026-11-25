set -eu
root=/data1/shenglin.zeng/teacher_turning_20261026
find "$root/models" -maxdepth 3 -name config.json
ls "$root/runtime/bin" | head
cat "$root/nccl_selfcheck.py" | head -25
