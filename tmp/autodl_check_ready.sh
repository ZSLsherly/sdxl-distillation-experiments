set -eu
root=/root/autodl-tmp/sdxl_research
date -Is
cat "$root/setup.status" 2>/dev/null || true
cat "$root/preparation_complete.json" 2>/dev/null || true
ls -lh "$root/transfers"
tail -n 55 "$root/logs/setup.log"
ps -eo pid,etime,args | grep -E '[b]ash setup.sh|[t]rain.py|[t]orch.distributed.run' || true
df -h "$root"
