set -eu
base=/data5/shenglin.zeng/checkpoints/a800_abc_g160_20261102
aux="$base/.abc_g160_transfer_20261102"
date '+%F %T %z'
wc -c "$base"/{A,B,C}/student_rank0.pt 2>/dev/null || true
for a in A B C; do if test -f "$base/$a/VERIFIED.txt"; then cat "$base/$a/VERIFIED.txt"; fi; done
find "$aux" -maxdepth 2 -type f \( -name '*.chunk' -o -name '*.part' \) -printf '%s\n' | awk '{s+=$1;n++} END {printf "chunk files=%d bytes=%.0f\n",n,s}'
if test -f "$aux/direct.log"; then tail -4 "$aux/direct.log"; fi
