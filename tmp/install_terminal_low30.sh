set -eu
. /root/sdxl_research/env.sh
tar -xzf "$SDXL_ROOT/tmp/terminal_low30.tgz" -C "$SDXL_ROOT/tmp"
bash "$SDXL_ROOT/tmp/deploy_terminal_low30.sh"
