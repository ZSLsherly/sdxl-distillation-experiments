cd /data5/shenglin.zeng/a50k_20261124
date
ps -u "$(id -un)" -o pid,ppid,stat,etime,wchan:24,comm | tail -26
ls -lh logs cache
ls -lh datasets/parallel_50k_seed20261018/prepared.pt*
df -h /data5 /
