from backup_a800_nov5 import lab,cloud,run
with lab() as d:
 print(run(d,"tail -n 12 /data5/shenglin.zeng/checkpoints/a800_archive_20261106/.pull6/filezilla.log; ps -eo pid,etime,args | grep -E 'fz_backup.py|/usr/bin/fzsftp' | grep -v grep; find /data5/shenglin.zeng/checkpoints/a800_archive_20261106 -name '*.fzpart' -printf '%p %s\\n'"))
with cloud() as s:print(run(s,'df -h /root/autodl-tmp'))
