p=/data5/shenglin.zeng/checkpoints
find "$p/a800_archive_20261106" -size +50M -printf '%s %p\n' | tail -40
find "$p/a800_archive_20261105" -name student_rank0.pt -printf '%s %p\n'
df -h "$p"
