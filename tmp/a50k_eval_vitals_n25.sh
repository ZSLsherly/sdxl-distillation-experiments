cd /data/shenglin.zeng/a50k_20261123/eval_A_milestones_20261125 || exit 1
date -u
ps -o pid,stat,etime,pcpu,rss,wchan:20 -p 18825,18832,18833,18834,18835
free -h | head -2
for d in results/A*; do echo "$d"; grep -aE '^STRICT_STUDENT_LOADED|^PROGRESS|^ALL_METRICS_COMPLETE' "$d/worker.log" | tail -1 | cut -c1-200; done
tail -1 queue.log
