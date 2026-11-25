from backup_a800_nov5 import lab,run
with lab() as d:
 print(run(d,"tail -n 15 /data5/shenglin.zeng/checkpoints/a800_archive_20261106/.pull6/log; du -h --max-depth=1 /data5/shenglin.zeng/checkpoints/a800_archive_20261106; test ! -f /data5/shenglin.zeng/checkpoints/a800_archive_20261106/COMPLETE.json || cat /data5/shenglin.zeng/checkpoints/a800_archive_20261106/COMPLETE.json"))
