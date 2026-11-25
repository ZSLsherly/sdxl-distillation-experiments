set -eu
. /root/sdxl_research/env.sh
cd "$SDXL_ROOT/outputs/A_pcm10_terminal_patchgan"
tail -8 train.log | cut -c1-350
tail -4 queue.log
test ! -e model/status.json || cat model/status.json
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

