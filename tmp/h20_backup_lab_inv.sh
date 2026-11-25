set -eu
date -Is;ls -ld /data5/shenglin.zeng /data5/shenglin.zeng/checkpoints
df -h /data5/shenglin.zeng
ls -d /data5/shenglin.zeng/checkpoints/*h20* 2>/dev/null || true
