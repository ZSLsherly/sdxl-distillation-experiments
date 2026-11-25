from backup_a800_nov5 import cloud,lab,run
with cloud() as s,lab() as d:
 print('CLOUD',run(s,"date -Iseconds; ps -eo pid,etime,pcpu,args | grep -E 'dd if=|read_side_n6' | grep -v grep; vmstat 1 2; df -h /root/autodl-tmp"))
 print('LAB',run(d,"date -Iseconds; ps -eo pid,etime,pcpu,args | grep -E 'pull6/pull.py|root@connect.nma1' | grep -v grep; du -sh /data5/shenglin.zeng/checkpoints/a800_archive_20261106; tail -n 6 /data5/shenglin.zeng/checkpoints/a800_archive_20261106/.pull6/log"))
