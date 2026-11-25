set -eu
cd /data5/shenglin.zeng/checkpoints
df -h .
find . -name student_rank0.pt -printf '%p %s\n'

