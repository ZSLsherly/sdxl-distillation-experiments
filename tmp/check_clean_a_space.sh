set -eu
. /root/sdxl_research/env.sh
df -h /root "$SDXL_WORK" /dev/shm
cd "$SDXL_ROOT/outputs/turning_BC_paired_quota_20261102/uniform_control/source"
sed -n '/def local_rollout/,/def add_fake/p' boundary.py
grep -n -e 's = knots' -e 'knots\[k\]' -e 'boundary_t.*knots' train.py
ls -l /root |head -25

