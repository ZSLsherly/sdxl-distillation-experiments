set -eu
. /root/sdxl_research/env.sh
df -B1 --output=avail,target "$SDXL_WORK"
du -h --max-depth=2 "$SDXL_WORK" | sort -hr | head -30
ls -l "$SDXL_WORK/datasets/parallel_20k_seed20261018.building/parts/" | tail -4
