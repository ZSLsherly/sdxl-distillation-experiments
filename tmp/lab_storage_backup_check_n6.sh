set -eu
p=/data5/shenglin.zeng/checkpoints/a800_archive_20261106
ls -l "$p/COMPLETE.json" "$p/.pull6/filezilla.log" 2>/dev/null || true
tail -12 "$p/.pull6/filezilla.log" 2>/dev/null || true
ps -eo pid,args | grep -E '[f]z_backup.py|[f]zsftp' | head -10
du -sh "$p"
