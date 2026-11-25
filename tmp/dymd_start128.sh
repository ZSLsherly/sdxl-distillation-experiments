set -eu
B=/data1/shenglin.zeng/dymd_prior_8_20261101
"$B/env/bin/python" "$B/prepare_run.py"
"$B/env/bin/python" -m py_compile "$B/profile_official.py" "$B/controller.py" "$B/analyze.py"
"$B/env/bin/python" "$B/test_measurement.py"
nohup python3 "$B/controller.py" >> "$B/controller.log" 2>&1 < /dev/null &
echo $! > "$B/controller.pid"
cat "$B/controller.pid"
