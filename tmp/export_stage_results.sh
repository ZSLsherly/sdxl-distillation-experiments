set -euo pipefail
root=/data1/shenglin.zeng/teacher_turning_20261026
cd "$root"
export LD_LIBRARY_PATH="$root/runtime/lib:${LD_LIBRARY_PATH:-}"
env/bin/python -u export_stage_results.py
