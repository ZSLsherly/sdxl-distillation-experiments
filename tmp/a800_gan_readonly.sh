set -eu
root=/root/sdxl_research
find "$root/models" -maxdepth 2 -name config.json -printf '%P\n'
cat "$root/outputs/turning_BC_matched_quota_20261101/B/status.json"
