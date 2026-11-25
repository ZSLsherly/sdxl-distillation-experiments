set -eu
. /root/sdxl_research/env.sh
o="$SDXL_ROOT/outputs/c_mix1_dm234_20261103"
tail -3 "$o/queue.log"; tail -4 "$o/train.log" | cut -c1-180
cat "$o/model/status.json" 2>/dev/null || true
df -h "$o"|tail -1
