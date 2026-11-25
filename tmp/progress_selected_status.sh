p=/data5/shenglin.zeng/checkpoints/a800_archive_20261109/progress_selected
date -Iseconds
du -sh "$p"
find "$p" -maxdepth 1 -type f -printf '%s %f\n' | sort -nr | head -5
