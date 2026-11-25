set -eu
cd /data5/shenglin.zeng/checkpoints/a800_archive_20261105
tail -4 .pull5/log
du -h --max-depth=1 outputs | tail -6
test ! -f COMPLETE.json || cat COMPLETE.json
