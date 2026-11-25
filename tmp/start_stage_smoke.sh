set -euo pipefail
root=/data1/shenglin.zeng/teacher_turning_20261026
cd "$root"
test -f stage_assets/verified.json
export LD_LIBRARY_PATH="$root/runtime/lib:${LD_LIBRARY_PATH:-}"
nice -n 10 env/bin/python -u guard_stage_intervention.py --smoke
