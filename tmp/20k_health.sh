set -eu
. /root/sdxl_research/env.sh
w="$SDXL_WORK/datasets/20k_sources"; b="$SDXL_WORK/datasets/parallel_20k_seed20261018.building"; d="$SDXL_WORK/datasets/parallel_20k_seed20261018"
date '+%F %T %z'
grep -E '^SELECTED|^ENCODER_CHECK|^ENCODED|^READY|^Traceback|^AssertionError|^RuntimeError|^ValueError|^OSError' "$w/build.log" | tail -6
if test -e "$b/parts/encoding.json"; then cat "$b/parts/encoding.json"; fi
if test -e "$d/validation.json"; then cat "$d/validation.json"; fi
df -h "$w" | tail -1
