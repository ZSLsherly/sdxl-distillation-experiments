set -eu
. /root/sdxl_research/env.sh
p="$SDXL_ROOT/outputs/flash_gan_g160_20261102"
for l in "$p"/gan_micro*.log; do echo "$l"; grep -Ei -B16 'CUDA out of memory' "$l" | tail -19; done
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
