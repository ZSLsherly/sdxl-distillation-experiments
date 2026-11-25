p=/data5/shenglin.zeng/checkpoints/a800_archive_20261109
find "$p" -type f -printf '%s %p\n' | tail -10
ps -eo etime,comm,args | grep '[s]ha256sum' | head -6
