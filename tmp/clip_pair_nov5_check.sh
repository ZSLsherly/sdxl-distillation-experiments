set -eu
. /root/sdxl_research/env.sh
cd "$SDXL_WORK/semantic_pair_1024_20261105"
tail -4 worker.log | cut -c1-400
test ! -f results.json || echo COMPLETE

