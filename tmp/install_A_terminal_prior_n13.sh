set -eu
. /root/sdxl_research/env.sh
tar -xzf "$SDXL_ROOT/tmp/A_terminal_prior_n13.tgz" -C "$SDXL_ROOT/tmp"
bash "$SDXL_ROOT/tmp/deploy_A_terminal_prior_n13.sh"
