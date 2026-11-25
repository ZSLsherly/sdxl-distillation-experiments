cd /data/shenglin.zeng/a50k_20261123/eval_A_milestones_20261125 || exit 1
for pid in 18832 18833 18834 18835; do echo "$pid"; sed -n '/read_bytes/p' /proc/"$pid"/io 2>/dev/null; done
for i in sda sdd; do cat /sys/block/"$i"/stat; done
sleep 5
for pid in 18832 18833 18834 18835; do echo "$pid"; sed -n '/read_bytes/p' /proc/"$pid"/io 2>/dev/null; done
for i in sda sdd; do cat /sys/block/"$i"/stat; done
