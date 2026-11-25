set -eu
base=/data5/shenglin.zeng/checkpoints/a800_abc_g160_20261102
df -h "$base"
for a in A B C; do
 f="$base/$a/student_rank0.pt"
 if test -f "$f"; then stat -c '%n %s' "$f"; sha256sum "$f"; fi
done
find /data5/shenglin.zeng/checkpoints -maxdepth 4 -type f -name student_rank0.pt -printf '%p %s\n'
