set -eu
df -h /data5/shenglin.zeng
command -v rsync || true
command -v sshpass || true
test ! -x /data5/shenglin.zeng/tools/filezilla/app/usr/bin/fzsftp || echo FILEZILLA_PRESENT
ls -ld /data5/shenglin.zeng/checkpoints/a800_archive_20261114
ps -eo pid,args | grep -E 'filezilla_backup|fzsftp|backup_completed' | grep -v grep || true
test ! -e /data5/shenglin.zeng/checkpoints/a800_archive_20261114/outputs/A_clean_global_high30_g160_20261114_v2 || find /data5/shenglin.zeng/checkpoints/a800_archive_20261114/outputs/A_clean_global_high30_g160_20261114_v2 -type f -printf '%s %p\n'
