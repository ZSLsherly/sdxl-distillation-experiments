set -eu
. /root/sdxl_research/env.sh
w="$SDXL_WORK/datasets/20k_sources"
ls -lh "$w"/vae* || true
tail -6 "$w/download.log"
ls "$SDXL_WORK/datasets/parallel_20k_seed20261018.building/parts" 2>/dev/null | tail -3 || true
tail -6 "$w/build.log" 2>/dev/null || true
