set -eu
. /root/sdxl_research/env.sh
d="$SDXL_WORK/semantic_four_1024_20261103"
date '+%F %T %z'
tail -7 "$d/worker.log" | cut -c1-300
if test -f "$d/results.json";then python -c "import json,sys; r=json.load(open(sys.argv[1]));print(r['metrics']);print(r['differences']);print(r['replication_max_abs_diff'])" "$d/results.json"; fi
