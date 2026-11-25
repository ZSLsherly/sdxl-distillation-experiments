set -eu
B=/data1/shenglin.zeng/dymd_prior_8_20261101
"$B/env/bin/python" "$B/test_128_protocol.py" > "$B/test_128_protocol.log" 2>&1
cat "$B/test_128_protocol.log"
cat "$B/status.json"
