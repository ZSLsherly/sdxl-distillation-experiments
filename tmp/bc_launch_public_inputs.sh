set -eu
. /root/sdxl_research/env.sh
mkdir -p "$SDXL_WORK/bc_texture_inputs"
nohup nice -n 10 python -u "$SDXL_ROOT/tmp/bc_public_sdxl_inputs.py" > "$SDXL_WORK/bc_texture_inputs/public_download.log" 2>&1 < /dev/null &
echo PUBLIC_INPUT_PID=$!
