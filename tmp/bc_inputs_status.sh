set -eu
. /root/sdxl_research/env.sh
tail -n 10 "$SDXL_WORK/bc_texture_inputs/public_download.log"
ls -lh "$SDXL_ROOT/tmp/bc_public_reference.tar" 2>/dev/null || true
