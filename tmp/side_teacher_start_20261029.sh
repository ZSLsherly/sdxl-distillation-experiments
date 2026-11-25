set -eu
root=/data1/shenglin.zeng/teacher_turning_20261026
export LD_LIBRARY_PATH="$root/runtime/lib:${LD_LIBRARY_PATH:-}"
export TMPDIR="$root/tmp"
nohup "$root/env/bin/python" -u "$root/tmp/side_teacher_guard_20261029.py" > "$root/tmp/side_teacher_guard_20261029.log" 2>&1 < /dev/null &
echo PID=$!
