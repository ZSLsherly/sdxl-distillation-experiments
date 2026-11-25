set -eu
. /root/sdxl_research/env.sh
cd "$SDXL_WORK/semantic_pair_1024_20261105"
for i in $(seq 1 55); do
 if test -s results.json;then python - <<'PY'
import json
s=json.load(open('results.json'))
print({k:(v['cosine_x100'],v['top1_correct'],v['top5_correct']) for k,v in s['metrics'].items()},s['differences'])
PY
 exit 0;fi
 kill -0 "$(cat worker.pid)" 2>/dev/null || { tail -15 worker.log;exit 1; }
 sleep 10
done
tail -1 worker.log

