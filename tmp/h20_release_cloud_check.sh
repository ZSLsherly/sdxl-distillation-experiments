set -eu
. /root/sdxl_research/env.sh
sha256sum "$SDXL_ROOT/datasets/parallel_10k_seed20261018/prepared.pt"
pwd; ls -a /root/autodl-tmp; ls /root/sdxl_research
