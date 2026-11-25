set -eu
. /root/sdxl_research/env.sh
p="$SDXL_ROOT/outputs/flash_gan_g160_20261102"
tail -6 "$p/queue.log"
for f in "$p"/gan_micro*/status.json; do test -e "$f" && head -c 400 "$f"; done
