cd /data/shenglin.zeng/a50k_20261123/eval_A_milestones_20261125 || exit 1
date;tail -5 queue.log
for d in results/A*; do echo "$d"; test ! -f "$d/status_0.json" || head -c 200 "$d/status_0.json"; echo;test ! -f "$d/worker.log" || tail -2 "$d/worker.log" | cut -c1-220;test ! -f "$d/all_complete.json" || cat "$d/all_complete.json"; done
test ! -f summary.json || cat summary.json
