set -eu
root=/data5/shenglin.zeng/sdxl_crossdistill
pgrep -u "$(id -u)" -af '[t]ransfer_81_assets_20261028.py' || true
tail -n 6 "$root/tmp/transfer_81_assets_20261029.log" 2>/dev/null || true
